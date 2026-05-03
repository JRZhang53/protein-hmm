"""Compute Q3, SOV, ARI, test LL/residue, and BIC for every comparator model
in the unified-evaluation table, so the poster table has no missing cells.

Outputs a JSON dump of all rows so the poster table can be filled directly
from the numbers.
"""

from __future__ import annotations

from collections import Counter
import json
import math

import numpy as np

from _bootstrap import bootstrap

ROOT = bootstrap()

from protein_hmm.analysis.metrics import (
    adjusted_rand_index,
    bic_score,
    q3_accuracy,
    segment_overlap_score,
)
from protein_hmm.config import load_project_config
from protein_hmm.data.encoding import AminoAcidEncoder
from protein_hmm.data.loaders import load_split_records
from protein_hmm.models.baselines import IIDCategoricalModel, ObservedMarkovChain
from protein_hmm.models.discrete_hmm import DiscreteHMM
from protein_hmm.models.semi_supervised_hmm import SemiSupervisedHMM
from protein_hmm.utils.io import write_json
from protein_hmm.utils.paths import by_K_dir, resolve_project_path


def _ari_vs_dssp(predicted_seqs: list[str], true_seqs: list[str]) -> float:
    """ARI between flattened predicted and true per-residue label strings."""
    flat_pred = "".join(predicted_seqs)
    flat_true = "".join(true_seqs)
    return adjusted_rand_index(list(flat_true), list(flat_pred))


def main() -> None:
    config = load_project_config(ROOT)
    splits = load_split_records(resolve_project_path(config.data["processed_dir"], ROOT))
    train_records = splits["train"]
    test_records = splits["test"]
    encoder = AminoAcidEncoder()

    train_seqs = [encoder.encode(r.sequence) for r in train_records]
    test_seqs = [encoder.encode(r.sequence) for r in test_records]
    train_residues = sum(len(s) for s in train_seqs)
    test_residues = sum(len(s) for s in test_seqs)
    train_labeled = [r for r in train_records if r.labels]
    test_labeled = [r for r in test_records if r.labels]
    test_label_seqs = [r.labels for r in test_labeled]
    test_seqs_labeled = [encoder.encode(r.sequence) for r in test_labeled]

    rows: list[dict] = []

    # 1. i.i.d. categorical
    iid = IIDCategoricalModel().fit(train_seqs)
    iid_train_ll = iid.score_many(train_seqs)
    iid_test_ll = iid.score_many(test_seqs)
    iid_params = 19
    rows.append({
        "model": "i.i.d. categorical",
        "test_log_likelihood_per_residue": iid_test_ll / test_residues,
        "q3": None, "sov": None, "ari": None,
        "bic": bic_score(iid_train_ll, iid_params, train_residues),
        "params": iid_params,
    })

    # 2. Observed Markov chain
    markov = ObservedMarkovChain().fit(train_seqs)
    markov_train_ll = markov.score_many(train_seqs)
    markov_test_ll = markov.score_many(test_seqs)
    # Initial 19 + 20*19 transition rows
    markov_params = 19 + 20 * 19
    rows.append({
        "model": "Observed Markov chain over residues",
        "test_log_likelihood_per_residue": markov_test_ll / test_residues,
        "q3": None, "sov": None, "ari": None,
        "bic": bic_score(markov_train_ll, markov_params, train_residues),
        "params": markov_params,
    })

    # 3. Residue-majority P(SS|aa)
    counts: dict[str, Counter] = {}
    for r in train_labeled:
        for aa, ss in zip(r.sequence, r.labels):
            counts.setdefault(aa, Counter())[ss] += 1
    residue_to_label = {aa: cnt.most_common(1)[0][0] for aa, cnt in counts.items()}
    fallback = Counter(c for r in train_labeled for c in r.labels).most_common(1)[0][0]
    pred = ["".join(residue_to_label.get(aa, fallback) for aa in r.sequence) for r in test_labeled]
    truth = [r.labels for r in test_labeled]
    rmaj_q3 = q3_accuracy("".join(truth), "".join(pred))
    rmaj_sov = float(np.mean([segment_overlap_score(t, p) for t, p in zip(truth, pred)]))
    rmaj_ari = _ari_vs_dssp(pred, truth)
    rows.append({
        "model": "Residue-majority P(SS|aa)",
        "test_log_likelihood_per_residue": None,
        "q3": rmaj_q3, "sov": rmaj_sov, "ari": rmaj_ari,
        "bic": None, "params": None,
    })

    # 4. Family-majority class
    family_label: dict[str, str] = {}
    for fam in {r.family for r in train_labeled}:
        cnt = Counter(c for r in train_labeled if r.family == fam for c in r.labels)
        family_label[fam] = cnt.most_common(1)[0][0]
    pred = [family_label.get(r.family, fallback) * r.length for r in test_labeled]
    fmaj_q3 = q3_accuracy("".join(truth), "".join(pred))
    fmaj_sov = float(np.mean([segment_overlap_score(t, p) for t, p in zip(truth, pred)]))
    fmaj_ari = _ari_vs_dssp(pred, truth)
    rows.append({
        "model": "Family-majority class",
        "test_log_likelihood_per_residue": None,
        "q3": fmaj_q3, "sov": fmaj_sov, "ari": fmaj_ari,
        "bic": None, "params": None,
    })

    # 5. Reference HMM K=3 (semi-supervised), already trained
    ref_path = resolve_project_path(config.experiments["outputs"]["model_dir"], ROOT) / "reference_hmm.json"
    ref = SemiSupervisedHMM.load(ref_path)
    ref_train_seqs = [encoder.encode(r.sequence) for r in train_labeled]
    ref_train_ll = ref.score_many(ref_train_seqs)
    ref_test_ll = ref.score_many(test_seqs_labeled)
    ref_params = ref.parameter_count()
    pred = [ref.predict_labels(s) for s in test_seqs_labeled]
    ref_q3 = q3_accuracy("".join(test_label_seqs), "".join(pred))
    ref_sov = float(np.mean([segment_overlap_score(t, p) for t, p in zip(test_label_seqs, pred)]))
    ref_ari = _ari_vs_dssp(pred, test_label_seqs)
    rows.append({
        "model": "Reference HMM K=3 (semi-supervised)",
        "test_log_likelihood_per_residue": ref_test_ll / sum(len(s) for s in test_seqs_labeled),
        "q3": ref_q3, "sov": ref_sov, "ari": ref_ari,
        "bic": bic_score(ref_train_ll, ref_params,
                         sum(len(s) for s in ref_train_seqs)),
        "params": ref_params,
    })

    # 6. Unsupervised HMM K=6
    K = int(config.models["unsupervised"]["num_states"])
    unsup = DiscreteHMM.load(by_K_dir(K, "models", ROOT) / "unsupervised_hmm.json")
    unsup_train_ll = unsup.score_many(train_seqs)
    unsup_test_ll = unsup.score_many(test_seqs)
    unsup_params = unsup.parameter_count()
    # Decode each test protein, map states to labels using train mapping
    train_paths = [list(unsup.decode(encoder.encode(r.sequence)).states) for r in train_labeled]
    state_to_label_counts = [Counter() for _ in range(unsup.num_states)]
    for path, r in zip(train_paths, train_labeled):
        for state, lbl in zip(path, r.labels):
            state_to_label_counts[state][lbl] += 1
    state_to_label = {
        k: state_to_label_counts[k].most_common(1)[0][0] if state_to_label_counts[k] else fallback
        for k in range(unsup.num_states)
    }
    test_paths = [list(unsup.decode(encoder.encode(r.sequence)).states) for r in test_labeled]
    pred = ["".join(state_to_label[s] for s in path) for path in test_paths]
    unsup_q3 = q3_accuracy("".join(test_label_seqs), "".join(pred))
    unsup_sov = float(np.mean([segment_overlap_score(t, p) for t, p in zip(test_label_seqs, pred)]))
    # ARI between raw decoded states and DSSP labels (no relabelling)
    flat_states = [s for path in test_paths for s in path]
    flat_truth = list("".join(test_label_seqs))
    unsup_ari = adjusted_rand_index(flat_truth, flat_states)
    rows.append({
        "model": f"Unsupervised HMM K={K} (this work)",
        "test_log_likelihood_per_residue": unsup_test_ll / test_residues,
        "q3": unsup_q3, "sov": unsup_sov, "ari": unsup_ari,
        "bic": bic_score(unsup_train_ll, unsup_params, train_residues),
        "params": unsup_params,
    })

    # Print poster-ready table
    print(f"{'model':45s}  {'LL/res':>9}  {'Q3':>6}  {'SOV':>6}  {'ARI':>6}  {'BIC':>10}  {'params':>7}")
    for r in rows:
        ll = f"{r['test_log_likelihood_per_residue']:>9.4f}" if r['test_log_likelihood_per_residue'] is not None else "       --"
        q3 = f"{r['q3']:>6.3f}" if r['q3'] is not None else "    --"
        sov = f"{r['sov']:>6.3f}" if r['sov'] is not None else "    --"
        ari = f"{r['ari']:>6.3f}" if r['ari'] is not None else "    --"
        bic = f"{r['bic']:>10,.0f}" if r['bic'] is not None else "        --"
        params = f"{r['params']:>7d}" if r['params'] is not None else "     --"
        print(f"{r['model']:45s}  {ll}  {q3}  {sov}  {ari}  {bic}  {params}")

    out_path = resolve_project_path("results/metrics/full_evaluation_table.json", ROOT)
    write_json(out_path, {"rows": rows, "train_residues": train_residues, "test_residues": test_residues})
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
