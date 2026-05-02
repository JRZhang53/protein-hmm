"""Compute mean per-residue solvent accessibility (RSA) per HMM latent state.

Walks the test split, downloads any missing DSSP files for the cached
SIFTS chains, reconstructs the per-residue RSA aligned to each
ProteinRecord.sequence, decodes the sequence with the saved K=4
unsupervised HMM, and aggregates RSA samples by latent state.

Outputs:
  results/metrics/state_rsa.json
  reports/figures/state_rsa.png
"""

from __future__ import annotations

from _bootstrap import bootstrap

ROOT = bootstrap()

from protein_hmm.analysis.rsa import (
    aggregate_state_rsa,
    per_residue_rsa,
    _ensure_dssp,
)
from protein_hmm.config import load_project_config
from protein_hmm.data.encoding import AminoAcidEncoder
from protein_hmm.data.loaders import load_split_records
from protein_hmm.models.discrete_hmm import DiscreteHMM
from protein_hmm.utils.io import ensure_dir, write_json
from protein_hmm.utils.paths import resolve_project_path
from protein_hmm.visualization.summary_plots import plot_state_property_bars


def main() -> None:
    config = load_project_config(ROOT)
    splits = load_split_records(resolve_project_path(config.data["processed_dir"], ROOT))
    test_records = splits["test"]
    if not test_records:
        raise RuntimeError("Test split is empty.")

    model = DiscreteHMM.load(
        resolve_project_path(config.experiments["outputs"]["model_dir"], ROOT) / "unsupervised_hmm.json"
    )
    encoder = AminoAcidEncoder()

    sifts_dir = resolve_project_path("data/raw/pdb/sifts_xml", ROOT)
    dssp_dir = ensure_dir(resolve_project_path("data/raw/dssp/files", ROOT))

    state_paths: list[list[int]] = []
    rsa_per_record: list[list[float | None]] = []

    skipped = {"missing_pdb_metadata": 0, "no_sifts": 0, "no_dssp": 0, "alignment_mismatch": 0}
    decoded_count = 0

    for record in test_records:
        meta = record.metadata
        pdb_id = meta.get("pdb")
        chain_id = meta.get("chain")
        accession = meta.get("accession")
        try:
            sp_beg = int(meta.get("start"))
            sp_end = int(meta.get("end"))
        except (TypeError, ValueError):
            skipped["missing_pdb_metadata"] += 1
            continue
        if not (pdb_id and chain_id and accession):
            skipped["missing_pdb_metadata"] += 1
            continue

        sifts_path = sifts_dir / f"{pdb_id}.xml.gz"
        if not sifts_path.exists():
            skipped["no_sifts"] += 1
            continue

        dssp_path = _ensure_dssp(pdb_id, dssp_dir)
        if dssp_path is None:
            skipped["no_dssp"] += 1
            continue

        try:
            rsa_values = per_residue_rsa(
                sifts_xml_path=sifts_path,
                dssp_path=dssp_path,
                accession=accession,
                chain_id=chain_id,
                sp_beg=sp_beg,
                sp_end=sp_end,
                expected_sequence=record.sequence,
            )
        except Exception:
            skipped["no_sifts"] += 1
            continue

        if rsa_values is None:
            skipped["alignment_mismatch"] += 1
            continue

        decoded = model.decode(encoder.encode(record.sequence), protein_id=record.protein_id)
        state_paths.append(list(decoded.states))
        rsa_per_record.append(rsa_values)
        decoded_count += 1
        if decoded_count % 25 == 0:
            print(f"[rsa] processed {decoded_count}/{len(test_records)} test proteins", flush=True)

    summary = aggregate_state_rsa(state_paths, rsa_per_record, num_states=model.num_states)
    state_labels = [f"S{index}" for index in range(model.num_states)]

    payload = {
        "num_states": model.num_states,
        "test_records_used": decoded_count,
        "skipped": skipped,
        "mean_rsa_per_state": summary["mean"],
        "median_rsa_per_state": summary["median"],
        "std_rsa_per_state": summary["std"],
        "n_residues_per_state": summary["n"],
    }
    metrics_path = resolve_project_path(config.experiments["outputs"]["metrics_dir"], ROOT) / "state_rsa.json"
    write_json(metrics_path, payload)

    figure_path = resolve_project_path(config.experiments["outputs"]["figure_dir"], ROOT) / "state_rsa.png"
    plot_state_property_bars(
        state_labels=state_labels,
        series={"Mean RSA": summary["mean"]},
        title="Solvent accessibility per latent state",
        ylabel="Mean RSA (0 = buried, 1 = exposed)",
        path=figure_path,
        diverging=False,
    )

    print(f"Wrote state RSA metrics to {metrics_path}")
    print(f"Wrote state RSA figure to {figure_path}")
    for state, mean in zip(state_labels, summary["mean"]):
        n = summary["n"][int(state[1:])]
        print(f"  {state}: mean RSA = {mean:.3f} (n={n} residues)")
    print(f"  test proteins used: {decoded_count}, skipped: {skipped}")


if __name__ == "__main__":
    main()
