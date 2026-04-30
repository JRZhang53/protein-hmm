"""Unified test-set evaluation across all models and baselines.

Produces a single rows-of-models metrics file containing per-residue
log-likelihood plus Q3 / SOV / ARI for:

  - i.i.d. categorical baseline (fit fresh)
  - first-order observed Markov chain (fit fresh)
  - global / family-majority / residue-majority annotation baselines
    (read from results/metrics/baselines.json, which evaluate_baselines.py
    produced)
  - unsupervised HMMs at every candidate K (read from
    results/metrics/model_selection.json so we don't refit)
  - constrained semi-supervised HMM (read from
    results/metrics/reference_metrics.json)

Run AFTER evaluate_baselines.py, run_model_selection.py, and
train_reference_hmm.py.
"""

from __future__ import annotations

from typing import Any

from _bootstrap import bootstrap

ROOT = bootstrap()

from protein_hmm.config import load_project_config
from protein_hmm.data.encoding import AminoAcidEncoder
from protein_hmm.data.loaders import load_split_records
from protein_hmm.models.baselines import IIDCategoricalModel, ObservedMarkovChain
from protein_hmm.utils.io import read_json, write_json
from protein_hmm.utils.paths import resolve_project_path


def _per_residue(score: float, residues: int) -> float:
    return float(score / residues) if residues else 0.0


def _encode(records, encoder):
    return [encoder.encode(record.sequence) for record in records]


def main() -> None:
    config = load_project_config(ROOT)
    metrics_dir = resolve_project_path(config.experiments["outputs"]["metrics_dir"], ROOT)
    splits = load_split_records(resolve_project_path(config.data["processed_dir"], ROOT))
    encoder = AminoAcidEncoder()

    train_sequences = _encode(splits["train"], encoder)
    val_sequences = _encode(splits["val"], encoder)
    test_sequences = _encode(splits["test"], encoder)
    test_residues = sum(len(seq) for seq in test_sequences)
    val_residues = sum(len(seq) for seq in val_sequences)

    rows: list[dict[str, Any]] = []

    iid_model = IIDCategoricalModel().fit(train_sequences)
    rows.append({
        "model": "iid_categorical",
        "kind": "baseline_lm",
        "test_log_likelihood_per_residue": _per_residue(iid_model.score_many(test_sequences), test_residues),
        "val_log_likelihood_per_residue": _per_residue(iid_model.score_many(val_sequences), val_residues),
    })

    markov_model = ObservedMarkovChain().fit(train_sequences)
    rows.append({
        "model": "observed_markov",
        "kind": "baseline_lm",
        "test_log_likelihood_per_residue": _per_residue(markov_model.score_many(test_sequences), test_residues),
        "val_log_likelihood_per_residue": _per_residue(markov_model.score_many(val_sequences), val_residues),
    })

    baseline_payload = read_json(metrics_dir / "baselines.json")
    for entry in baseline_payload["baselines"]:
        rows.append({
            "model": entry["name"],
            "kind": "baseline_annotation",
            "q3": entry["q3"],
            "sov": entry["sov"],
        })

    model_selection = read_json(metrics_dir / "model_selection.json")
    for entry in model_selection["results"]:
        evaluation = entry.get("annotation_evaluation", {})
        rows.append({
            "model": f"unsupervised_K{entry['num_states']}",
            "kind": "unsupervised_hmm",
            "num_states": entry["num_states"],
            "pseudocount": entry.get("pseudocount"),
            "train_log_likelihood_per_residue": entry["train_log_likelihood_per_residue"],
            "val_log_likelihood_per_residue": entry["val_log_likelihood_per_residue"],
            "test_log_likelihood_per_residue": entry["test_log_likelihood_per_residue"],
            "bic": entry["bic"],
            "q3": evaluation.get("q3"),
            "sov": evaluation.get("sov"),
            "ari": evaluation.get("ari"),
            "state_label_map": evaluation.get("state_label_map"),
        })

    reference_path = metrics_dir / "reference_metrics.json"
    if reference_path.exists():
        reference = read_json(reference_path)
        train_residues = sum(len(seq) for seq in train_sequences) or 1
        rows.append({
            "model": "reference_hmm_semi_supervised",
            "kind": "supervised_hmm",
            "num_states": 3,
            "train_log_likelihood_per_residue": _per_residue(reference.get("train_log_likelihood", 0.0), train_residues),
            "test_log_likelihood_per_residue": _per_residue(reference.get("test_log_likelihood", 0.0), test_residues),
            "q3": reference.get("q3"),
            "sov": reference.get("sov"),
        })

    payload = {
        "rows": rows,
        "label_distribution": baseline_payload["label_distribution"],
    }
    output_path = metrics_dir / "unified_evaluation.json"
    write_json(output_path, payload)
    print(f"Wrote unified evaluation to {output_path}")


if __name__ == "__main__":
    main()
