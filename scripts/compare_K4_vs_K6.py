"""Train K=6 unsupervised HMM and inspect its state biology vs the saved K=4 model.

Saves results/models/unsupervised_hmm_K6.json and prints a per-state biology
report (top residues, hydrophobicity, DSSP enrichment, mean RSA, dwell time)
so we can decide whether K=6 has a clean enough interpretation to justify
switching the poster headline from K=4 to K=6.
"""

from __future__ import annotations

from collections import defaultdict

import json
import numpy as np

from _bootstrap import bootstrap

ROOT = bootstrap()

from protein_hmm.analysis.rsa import _ensure_dssp, per_residue_rsa
from protein_hmm.config import load_project_config
from protein_hmm.constants import (
    AMINO_ACIDS,
    KYTE_DOOLITTLE_HYDROPHOBICITY,
)
from protein_hmm.data.encoding import AminoAcidEncoder
from protein_hmm.data.loaders import load_split_records
from protein_hmm.models.discrete_hmm import DiscreteHMM
from protein_hmm.utils.io import write_json
from protein_hmm.utils.paths import by_K_dir, resolve_project_path


DSSP_LABELS = ("H", "E", "C")


def _train_k6(train_sequences, config) -> DiscreteHMM:
    base = {k: v for k, v in config.models["unsupervised"].items() if k != "num_states"}
    model = DiscreteHMM(num_states=6, **base)
    model.fit(train_sequences)
    return model


def _decode_test(model: DiscreteHMM, test_records, encoder) -> tuple[list[list[int]], list[str]]:
    state_paths = []
    label_seqs = []
    for r in test_records:
        if not r.labels:
            continue
        decoded = model.decode(encoder.encode(r.sequence), protein_id=r.protein_id, labels=r.labels)
        state_paths.append(list(decoded.states))
        label_seqs.append(r.labels)
    return state_paths, label_seqs


def _state_dssp_enrichment(state_paths, label_seqs, num_states):
    counts = np.ones((num_states, len(DSSP_LABELS)), dtype=float)
    label_to_idx = {l: i for i, l in enumerate(DSSP_LABELS)}
    for path, labels in zip(state_paths, label_seqs):
        if len(path) != len(labels):
            continue
        for s, l in zip(path, labels):
            if l in label_to_idx:
                counts[int(s), label_to_idx[l]] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def _per_state_rsa(model, test_records, encoder, sifts_dir, dssp_dir):
    samples = defaultdict(list)
    for r in test_records:
        meta = r.metadata
        try:
            sp_beg = int(meta.get("start"))
            sp_end = int(meta.get("end"))
        except (TypeError, ValueError):
            continue
        pdb_id = meta.get("pdb")
        chain_id = meta.get("chain")
        accession = meta.get("accession")
        if not (pdb_id and chain_id and accession):
            continue
        sifts_path = sifts_dir / f"{pdb_id}.xml.gz"
        if not sifts_path.exists():
            continue
        dssp_path = _ensure_dssp(pdb_id, dssp_dir)
        if dssp_path is None:
            continue
        try:
            rsa_values = per_residue_rsa(
                sifts_xml_path=sifts_path,
                dssp_path=dssp_path,
                accession=accession,
                chain_id=chain_id,
                sp_beg=sp_beg,
                sp_end=sp_end,
                expected_sequence=r.sequence,
            )
        except Exception:
            continue
        if rsa_values is None:
            continue
        states = list(model.decode(encoder.encode(r.sequence), protein_id=r.protein_id).states)
        for s, v in zip(states, rsa_values):
            if v is not None:
                samples[int(s)].append(float(v))
    return {s: (float(np.mean(v)), float(np.std(v)), len(v)) for s, v in samples.items()}


def main() -> None:
    config = load_project_config(ROOT)
    splits = load_split_records(resolve_project_path(config.data["processed_dir"], ROOT))
    encoder = AminoAcidEncoder()

    train_sequences = [encoder.encode(r.sequence) for r in splits["train"]]
    test_records = splits["test"]

    print("Training K=6 unsupervised HMM (4 restarts, max_iter 200)...")
    model = _train_k6(train_sequences, config)
    out_path = by_K_dir(6, "models", ROOT) / "unsupervised_hmm.json"
    model.save(out_path)
    print(f"Saved to {out_path}")
    print(f"Restart final LLs: {[round(x,1) for x in model.restart_log_likelihoods]}")
    print(f"Iterations: {model.training_history.iterations}, converged: {model.training_history.converged}")

    state_paths, label_seqs = _decode_test(model, test_records, encoder)
    enr = _state_dssp_enrichment(state_paths, label_seqs, num_states=6)

    sifts_dir = resolve_project_path("data/raw/pdb/sifts_xml", ROOT)
    dssp_dir = resolve_project_path("data/raw/dssp/files", ROOT)
    rsa = _per_state_rsa(model, test_records, encoder, sifts_dir, dssp_dir)

    if model.params is None:
        raise RuntimeError("No params")
    emit = model.params.emission_probs
    trans = model.params.transition_probs
    kd = np.asarray([KYTE_DOOLITTLE_HYDROPHOBICITY[a] for a in AMINO_ACIDS])
    stat = np.linalg.matrix_power(trans, 500)[0]

    summaries = []
    print("\n=== Per-state biology (K=6) ===")
    print(f"{'St':>3} {'occ':>5} {'KD':>6} {'RSA':>6} {'dwell':>6} {'P(H)':>5} {'P(E)':>5} {'P(C)':>5}  top residues")
    for k in range(6):
        e = emit[k]
        kd_val = float(e @ kd)
        top_indices = np.argsort(e)[::-1][:5]
        top = [AMINO_ACIDS[i] for i in top_indices]
        top_probs = [float(e[i]) for i in top_indices]
        dwell = 1.0 / max(1.0 - trans[k, k], 1e-9)
        rsa_mean = rsa.get(k, (float("nan"), 0.0, 0))[0]
        rsa_n = rsa.get(k, (0.0, 0.0, 0))[2]
        ph, pe, pc = enr[k]
        print(f"{k:>3} {stat[k]:>5.2f} {kd_val:>+6.2f} {rsa_mean:>6.2f} {dwell:>6.1f} {ph:>5.2f} {pe:>5.2f} {pc:>5.2f}  "
              f"{', '.join(f'{r}({p:.2f})' for r,p in zip(top, top_probs))}  (rsa_n={rsa_n})")

        summaries.append({
            "state": k,
            "occupancy": float(stat[k]),
            "kd": kd_val,
            "rsa": rsa_mean,
            "rsa_n": rsa_n,
            "dwell": dwell,
            "self_transition": float(trans[k, k]),
            "p_h": float(ph),
            "p_e": float(pe),
            "p_c": float(pc),
            "top_residues": list(zip(top, top_probs)),
        })

    payload = {"K": 6, "summaries": summaries}
    metrics_dir = by_K_dir(6, "metrics", ROOT)
    (metrics_dir / "state_summaries.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved K=6 summaries to {metrics_dir / 'state_summaries.json'}")


if __name__ == "__main__":
    main()
