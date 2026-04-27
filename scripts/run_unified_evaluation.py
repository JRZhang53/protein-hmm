"""Unified test-set evaluation across all models and baselines.

Produces a single metrics file (and CSV via make_report_tables) containing
per-residue log-likelihood plus Q3/SOV/ARI for:

  - i.i.d. categorical baseline
  - first-order observed Markov chain
  - global / family-majority / residue-majority annotation baselines
  - unsupervised HMMs at every candidate K
  - constrained semi-supervised HMM (reference)

Run AFTER train_unsupervised, run_model_selection, train_reference_hmm.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from _bootstrap import bootstrap

ROOT = bootstrap()

from protein_hmm.analysis.evaluation import (
    annotation_baselines,
    evaluate_hmm_annotations,
    training_diagnostics,
)
from protein_hmm.analysis.metrics import bic_score, q3_accuracy, segment_overlap_score
from protein_hmm.config import load_project_config
from protein_hmm.data.encoding import AminoAcidEncoder
from protein_hmm.data.loaders import load_split_records
from protein_hmm.models.baselines import IIDCategoricalModel, ObservedMarkovChain
from protein_hmm.models.discrete_hmm import DiscreteHMM
from protein_hmm.models.semi_supervised_hmm import SemiSupervisedHMM
from protein_hmm.utils.io import write_json
from protein_hmm.utils.paths import resolve_project_path


def _per_residue(score: float, residues: int) -> float:
    return float(score / residues) if residues else 0.0


def _encode(records, encoder):
    return [encoder.encode(record.sequence) for record in records]


def main() -> None:
    config = load_project_config(ROOT)
    splits = load_split_records(resolve_project_path(config.data["processed_dir"], ROOT))
    encoder = AminoAcidEncoder()

    train_records = splits["train"]
    test_records = splits["test"]
    val_records = splits["val"]
    train_labeled = [record for record in train_records if record.labels]
    test_labeled = [record for record in test_records if record.labels]

    train_sequences = _encode(train_records, encoder)
    val_sequences = _encode(val_records, encoder)
    test_sequences = _encode(test_records, encoder)

    test_residues = sum(len(seq) for seq in test_sequences)
    val_residues = sum(len(seq) for seq in val_sequences)
    train_residues = sum(len(seq) for seq in train_sequences) or 1

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

    baseline_payload = annotation_baselines(train_records, test_records)
    for entry in baseline_payload["baselines"]:
        rows.append({
            "model": entry["name"],
            "kind": "baseline_annotation",
            "q3": entry["q3"],
            "sov": entry["sov"],
        })

    candidate_states = config.models["model_selection"]["candidate_states"]
    unsup_kwargs = {key: value for key, value in config.models["unsupervised"].items() if key != "num_states"}
    for num_states in candidate_states:
        model = DiscreteHMM(num_states=num_states, **unsup_kwargs)
        model.fit(train_sequences)
        params_count = model.parameter_count()
        train_ll = model.score_many(train_sequences)
        bic = bic_score(train_ll, params_count, train_residues)
        row: dict[str, Any] = {
            "model": f"unsupervised_K{num_states}",
            "kind": "unsupervised_hmm",
            "num_states": num_states,
            "train_log_likelihood_per_residue": _per_residue(train_ll, train_residues),
            "val_log_likelihood_per_residue": _per_residue(model.score_many(val_sequences), val_residues),
            "test_log_likelihood_per_residue": _per_residue(model.score_many(test_sequences), test_residues),
            "bic": bic,
            "training_diagnostics": training_diagnostics(model.training_history, train_residues),
            "restart_log_likelihoods": list(model.restart_log_likelihoods),
        }
        if train_labeled and test_labeled:
            evaluation = evaluate_hmm_annotations(
                model=model,
                encoder=encoder,
                mapping_records=train_labeled,
                evaluation_records=test_labeled,
            )
            row["q3"] = evaluation["q3"]
            row["sov"] = evaluation["sov"]
            row["ari"] = evaluation["ari"]
            row["state_label_map"] = evaluation["state_label_map"]
            row["state_label_enrichment"] = evaluation["state_label_enrichment"]
        rows.append(row)

    if train_labeled and test_labeled:
        train_label_seqs = [record.labels or "" for record in train_labeled]
        test_label_seqs = [record.labels or "" for record in test_labeled]
        train_seq_labeled = _encode(train_labeled, encoder)
        test_seq_labeled = _encode(test_labeled, encoder)
        reference = SemiSupervisedHMM(**config.models["reference"])
        reference.fit(train_seq_labeled, train_label_seqs)
        predicted = [reference.predict_labels(seq) for seq in test_seq_labeled]
        rows.append({
            "model": "reference_hmm_semi_supervised",
            "kind": "supervised_hmm",
            "num_states": reference.num_states,
            "train_log_likelihood_per_residue": _per_residue(
                reference.score_many(train_seq_labeled),
                sum(len(seq) for seq in train_seq_labeled) or 1,
            ),
            "test_log_likelihood_per_residue": _per_residue(
                reference.score_many(test_seq_labeled),
                sum(len(seq) for seq in test_seq_labeled) or 1,
            ),
            "q3": q3_accuracy("".join(test_label_seqs), "".join(predicted)),
            "sov": float(np.mean([
                segment_overlap_score(true, pred)
                for true, pred in zip(test_label_seqs, predicted)
            ])),
            "training_diagnostics": training_diagnostics(
                reference.training_history,
                sum(len(seq) for seq in train_seq_labeled) or 1,
            ),
        })

    payload = {
        "rows": rows,
        "label_distribution": baseline_payload["label_distribution"],
    }
    output_path = resolve_project_path(config.experiments["outputs"]["metrics_dir"], ROOT) / "unified_evaluation.json"
    write_json(output_path, payload)
    print(f"Wrote unified evaluation to {output_path}")


if __name__ == "__main__":
    main()
