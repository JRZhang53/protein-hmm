"""Evaluation metrics for unsupervised and supervised analyses."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from protein_hmm.constants import DSSP_COLLAPSE_MAP, DSSP_LABELS


def _as_labels(values: str | Iterable[int] | Iterable[str]) -> list[object]:
    if isinstance(values, str):
        return list(values)
    return list(values)


def _check_equal_length(left: list[object], right: list[object]) -> None:
    if len(left) != len(right):
        raise ValueError("Label sequences must have equal length.")


def _collapse_structural_labels(labels: list[object]) -> list[object]:
    collapsed: list[object] = []
    for label in labels:
        if isinstance(label, str):
            collapsed.append(DSSP_COLLAPSE_MAP.get(label.upper(), "C"))
        else:
            collapsed.append(label)
    return collapsed


def per_residue_accuracy(true_labels: str | Iterable[int] | Iterable[str], predicted_labels: str | Iterable[int] | Iterable[str]) -> float:
    true_list = _as_labels(true_labels)
    pred_list = _as_labels(predicted_labels)
    _check_equal_length(true_list, pred_list)
    if not true_list:
        return 0.0
    correct = sum(left == right for left, right in zip(true_list, pred_list))
    return correct / len(true_list)


def q3_accuracy(true_labels: str, predicted_labels: str) -> float:
    true_list = _collapse_structural_labels(_as_labels(true_labels))
    pred_list = _collapse_structural_labels(_as_labels(predicted_labels))
    return per_residue_accuracy(true_list, pred_list)


def _comb2(value: int) -> float:
    return 0.5 * value * (value - 1)


def adjusted_rand_index(
    true_labels: str | Iterable[int] | Iterable[str],
    predicted_labels: str | Iterable[int] | Iterable[str],
) -> float:
    true_list = _as_labels(true_labels)
    pred_list = _as_labels(predicted_labels)
    _check_equal_length(true_list, pred_list)
    n_items = len(true_list)
    if n_items < 2:
        return 1.0

    true_counts = Counter(true_list)
    pred_counts = Counter(pred_list)
    contingency = Counter(zip(true_list, pred_list))

    index = sum(_comb2(count) for count in contingency.values())
    true_sum = sum(_comb2(count) for count in true_counts.values())
    pred_sum = sum(_comb2(count) for count in pred_counts.values())
    total_pairs = _comb2(n_items)
    expected_index = (true_sum * pred_sum) / total_pairs if total_pairs else 0.0
    max_index = 0.5 * (true_sum + pred_sum)
    denominator = max_index - expected_index
    if denominator == 0.0:
        return 1.0
    return float((index - expected_index) / denominator)


@dataclass(frozen=True, slots=True)
class Segment:
    label: object
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def _segments(labels: list[object]) -> list[Segment]:
    if not labels:
        return []
    segments: list[Segment] = []
    start = 0
    current = labels[0]
    for index, label in enumerate(labels[1:], start=1):
        if label != current:
            segments.append(Segment(label=current, start=start, end=index))
            current = label
            start = index
    segments.append(Segment(label=current, start=start, end=len(labels)))
    return segments


def segment_overlap_score(true_labels: str, predicted_labels: str, label_order: tuple[str, ...] = DSSP_LABELS) -> float:
    true_list = _collapse_structural_labels(_as_labels(true_labels))
    pred_list = _collapse_structural_labels(_as_labels(predicted_labels))
    _check_equal_length(true_list, pred_list)

    true_segments = _segments(true_list)
    pred_segments = _segments(pred_list)
    total_length = sum(segment.length for segment in true_segments if segment.label in label_order)
    if total_length == 0:
        return 0.0

    score = 0.0
    for true_segment in true_segments:
        if true_segment.label not in label_order:
            continue
        overlapping = [
            pred_segment
            for pred_segment in pred_segments
            if pred_segment.label == true_segment.label
            and pred_segment.start < true_segment.end
            and pred_segment.end > true_segment.start
        ]
        if not overlapping:
            continue

        best = 0.0
        for pred_segment in overlapping:
            overlap = min(true_segment.end, pred_segment.end) - max(true_segment.start, pred_segment.start)
            union = max(true_segment.end, pred_segment.end) - min(true_segment.start, pred_segment.start)
            delta = min(
                union - overlap,
                overlap,
                true_segment.length / 2.0,
                pred_segment.length / 2.0,
            )
            best = max(best, ((overlap + delta) / union) * true_segment.length)
        score += best
    return float(score / total_length)


def bic_score(log_likelihood: float, num_parameters: int, num_observations: int) -> float:
    if num_observations <= 0:
        raise ValueError("num_observations must be positive.")
    return float(-2.0 * log_likelihood + num_parameters * np.log(num_observations))


def transition_matrix_frobenius(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape:
        raise ValueError("Transition matrices must have the same shape.")
    return float(np.linalg.norm(left - right, ord="fro"))
