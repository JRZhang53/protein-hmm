"""Evaluation helpers for annotation recovery and simple baselines."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np

from protein_hmm.analysis.metrics import adjusted_rand_index, q3_accuracy, segment_overlap_score
from protein_hmm.analysis.state_interpretation import state_label_enrichment
from protein_hmm.constants import AMINO_ACIDS, DSSP_LABELS
from protein_hmm.data.encoding import AminoAcidEncoder
from protein_hmm.types import ProteinRecord, TrainingHistory


def label_distribution(records: Iterable[ProteinRecord]) -> dict[str, float]:
    counts = Counter(label for record in records if record.labels for label in record.labels)
    total = sum(counts.values()) or 1
    return {label: counts.get(label, 0) / total for label in DSSP_LABELS}


def training_diagnostics(
    history: TrainingHistory,
    num_observations: int,
    per_observation_tol: float = 1e-4,
) -> dict[str, float | bool | int | None]:
    log_likelihoods = history.log_likelihoods
    last_delta = None
    relative_last_delta = None
    last_delta_per_observation = None
    if len(log_likelihoods) >= 2:
        last_delta = log_likelihoods[-1] - log_likelihoods[-2]
        relative_last_delta = abs(last_delta) / max(abs(log_likelihoods[-2]), 1.0)
        last_delta_per_observation = last_delta / max(num_observations, 1)

    return {
        "converged": history.converged,
        "iterations": history.iterations,
        "last_delta": last_delta,
        "relative_last_delta": relative_last_delta,
        "last_delta_per_observation": last_delta_per_observation,
        "per_observation_tol": per_observation_tol,
        "near_converged_per_observation": (
            False
            if last_delta_per_observation is None
            else abs(last_delta_per_observation) <= per_observation_tol + 1e-12
        ),
    }


def infer_state_to_label_map(
    decoded_states: list[list[int]],
    label_sequences: list[str],
    num_states: int,
) -> dict[int, str]:
    enrichment = state_label_enrichment(decoded_states, label_sequences, num_states=num_states, label_order=DSSP_LABELS)
    return {state: DSSP_LABELS[int(np.argmax(enrichment[state]))] for state in range(num_states)}


def evaluate_state_paths(
    decoded_states: list[list[int]],
    label_sequences: list[str],
    num_states: int,
    state_to_label: dict[int, str] | None = None,
) -> dict[str, Any]:
    if state_to_label is None:
        state_to_label = infer_state_to_label_map(decoded_states, label_sequences, num_states)
    predicted_labels = ["".join(state_to_label[state] for state in path) for path in decoded_states]
    flattened_states = [state for path in decoded_states for state in path]
    flattened_labels = [label for labels in label_sequences for label in labels]
    return {
        "ari": adjusted_rand_index(flattened_labels, flattened_states),
        "q3": q3_accuracy("".join(label_sequences), "".join(predicted_labels)),
        "sov": float(np.mean([segment_overlap_score(true, pred) for true, pred in zip(label_sequences, predicted_labels)]))
        if label_sequences
        else 0.0,
        "state_label_map": state_to_label,
        "state_label_enrichment": state_label_enrichment(
            decoded_states,
            label_sequences,
            num_states=num_states,
            label_order=DSSP_LABELS,
        ).tolist(),
    }


def evaluate_hmm_annotations(
    model: Any,
    encoder: AminoAcidEncoder,
    mapping_records: list[ProteinRecord],
    evaluation_records: list[ProteinRecord],
) -> dict[str, Any]:
    if not mapping_records:
        raise ValueError("At least one labeled mapping record is required.")
    if not evaluation_records:
        raise ValueError("At least one labeled evaluation record is required.")

    mapping_states: list[list[int]] = []
    mapping_labels: list[str] = []
    for record in mapping_records:
        if not record.labels:
            continue
        decoded = model.decode(encoder.encode(record.sequence), protein_id=record.protein_id, labels=record.labels)
        mapping_states.append(decoded.states)
        mapping_labels.append(record.labels)

    state_to_label = infer_state_to_label_map(mapping_states, mapping_labels, model.num_states)

    evaluation_states: list[list[int]] = []
    evaluation_labels: list[str] = []
    for record in evaluation_records:
        if not record.labels:
            continue
        decoded = model.decode(encoder.encode(record.sequence), protein_id=record.protein_id, labels=record.labels)
        evaluation_states.append(decoded.states)
        evaluation_labels.append(record.labels)

    metrics = evaluate_state_paths(
        evaluation_states,
        evaluation_labels,
        num_states=model.num_states,
        state_to_label=state_to_label,
    )
    metrics["mapping_split"] = "train"
    return metrics


def _label_counts(records: Iterable[ProteinRecord]) -> Counter[str]:
    return Counter(label for record in records if record.labels for label in record.labels)


def _majority_label(records: Iterable[ProteinRecord], fallback: str = "C") -> str:
    counts = _label_counts(records)
    if not counts:
        return fallback
    return counts.most_common(1)[0][0]


def _evaluate_label_predictions(
    records: list[ProteinRecord],
    predicted_sequences: list[str],
    name: str,
    mapping: dict[str, str] | str,
) -> dict[str, Any]:
    truths = [record.labels or "" for record in records]
    return {
        "name": name,
        "q3": q3_accuracy("".join(truths), "".join(predicted_sequences)) if truths else 0.0,
        "sov": float(np.mean([segment_overlap_score(true, pred) for true, pred in zip(truths, predicted_sequences)]))
        if truths
        else 0.0,
        "mapping": mapping,
    }


def majority_label_baseline(train_records: list[ProteinRecord], test_records: list[ProteinRecord]) -> dict[str, Any]:
    label = _majority_label(train_records)
    predictions = [label * record.length for record in test_records]
    return _evaluate_label_predictions(test_records, predictions, "global_majority_label", label)


def family_majority_label_baseline(train_records: list[ProteinRecord], test_records: list[ProteinRecord]) -> dict[str, Any]:
    global_label = _majority_label(train_records)
    family_records: dict[str, list[ProteinRecord]] = defaultdict(list)
    for record in train_records:
        family_records[record.family].append(record)
    family_to_label = {
        family: _majority_label(records, fallback=global_label)
        for family, records in family_records.items()
    }
    predictions = [family_to_label.get(record.family, global_label) * record.length for record in test_records]
    return _evaluate_label_predictions(test_records, predictions, "family_majority_label", family_to_label)


def residue_label_baseline(train_records: list[ProteinRecord], test_records: list[ProteinRecord]) -> dict[str, Any]:
    global_label = _majority_label(train_records)
    residue_counts: dict[str, Counter[str]] = {residue: Counter() for residue in AMINO_ACIDS}
    for record in train_records:
        if not record.labels:
            continue
        for residue, label in zip(record.sequence, record.labels):
            if residue in residue_counts:
                residue_counts[residue][label] += 1

    residue_to_label = {
        residue: (counts.most_common(1)[0][0] if counts else global_label)
        for residue, counts in residue_counts.items()
    }
    predictions = [
        "".join(residue_to_label.get(residue, global_label) for residue in record.sequence)
        for record in test_records
    ]
    return _evaluate_label_predictions(test_records, predictions, "residue_majority_label", residue_to_label)


def annotation_baselines(train_records: list[ProteinRecord], test_records: list[ProteinRecord]) -> dict[str, Any]:
    train_labeled = [record for record in train_records if record.labels]
    test_labeled = [record for record in test_records if record.labels]
    return {
        "label_distribution": {
            "train": label_distribution(train_labeled),
            "test": label_distribution(test_labeled),
        },
        "baselines": [
            majority_label_baseline(train_labeled, test_labeled),
            family_majority_label_baseline(train_labeled, test_labeled),
            residue_label_baseline(train_labeled, test_labeled),
        ],
    }
