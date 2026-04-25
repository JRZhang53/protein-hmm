"""Model comparison across protein families."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment

from protein_hmm.analysis.metrics import transition_matrix_frobenius
from protein_hmm.types import HMMParameters


def _emission_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pairwise Jensen-Shannon-like distance between two state x alphabet matrices."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape:
        raise ValueError("Emission matrices must have the same shape.")
    num_states = left.shape[0]
    distances = np.zeros((num_states, num_states), dtype=float)
    for i in range(num_states):
        for j in range(num_states):
            mean = 0.5 * (left[i] + right[j])
            kl_left = float(np.sum(left[i] * (np.log(np.clip(left[i], 1e-12, None)) - np.log(np.clip(mean, 1e-12, None)))))
            kl_right = float(np.sum(right[j] * (np.log(np.clip(right[j], 1e-12, None)) - np.log(np.clip(mean, 1e-12, None)))))
            distances[i, j] = 0.5 * (kl_left + kl_right)
    return distances


def hungarian_state_permutation(
    reference: HMMParameters,
    candidate: HMMParameters,
) -> np.ndarray:
    """Return a permutation array p such that ``candidate.*[p]`` aligns with ``reference``.

    Uses Hungarian assignment on Jensen-Shannon distance between emission rows.
    """
    distances = _emission_distance(reference.emission_probs, candidate.emission_probs)
    _, column_indices = linear_sum_assignment(distances)
    return np.asarray(column_indices, dtype=int)


def permute_parameters(params: HMMParameters, permutation: np.ndarray) -> HMMParameters:
    permutation = np.asarray(permutation, dtype=int)
    return HMMParameters(
        start_probs=params.start_probs[permutation],
        transition_probs=params.transition_probs[np.ix_(permutation, permutation)],
        emission_probs=params.emission_probs[permutation],
    )


def stationary_distribution(transition_probs: np.ndarray, max_iter: int = 5000, tol: float = 1e-12) -> np.ndarray:
    transition_probs = np.asarray(transition_probs, dtype=float)
    num_states = transition_probs.shape[0]
    distribution = np.full(num_states, 1.0 / num_states, dtype=float)
    for _ in range(max_iter):
        next_distribution = distribution @ transition_probs
        if np.max(np.abs(next_distribution - distribution)) < tol:
            distribution = next_distribution
            break
        distribution = next_distribution
    distribution = np.clip(distribution, 0.0, None)
    total = distribution.sum()
    if total <= 0.0:
        return np.full(num_states, 1.0 / num_states, dtype=float)
    return distribution / total


def expected_dwell_times(transition_probs: np.ndarray) -> np.ndarray:
    diagonal = np.clip(np.diag(np.asarray(transition_probs, dtype=float)), 0.0, 1.0 - 1e-9)
    return 1.0 / (1.0 - diagonal)


def transition_distance_matrix(
    models: Mapping[str, object],
    align_states: bool = True,
) -> tuple[list[str], np.ndarray]:
    """Pairwise Frobenius distance between transition matrices.

    When ``align_states=True``, candidate models are first permuted via
    Hungarian assignment on emissions so the comparison is not dominated by
    arbitrary state labelling.
    """
    families = sorted(models)
    matrix = np.zeros((len(families), len(families)), dtype=float)
    for row, family_i in enumerate(families):
        params_i = models[family_i].params
        if params_i is None:
            raise RuntimeError(f"Model {family_i} is not fit.")
        for col, family_j in enumerate(families):
            params_j = models[family_j].params
            if params_j is None:
                raise RuntimeError(f"Model {family_j} is not fit.")
            if align_states and row != col:
                permutation = hungarian_state_permutation(params_i, params_j)
                params_j_aligned = permute_parameters(params_j, permutation)
            else:
                params_j_aligned = params_j
            matrix[row, col] = transition_matrix_frobenius(
                params_i.transition_probs,
                params_j_aligned.transition_probs,
            )
    return families, matrix


def stationary_distance_matrix(models: Mapping[str, object]) -> tuple[list[str], np.ndarray]:
    """Pairwise L1 distance between sorted stationary distributions.

    Sorting makes the distance permutation-invariant without needing alignment.
    """
    families = sorted(models)
    distributions = []
    for family in families:
        params = models[family].params
        if params is None:
            raise RuntimeError(f"Model {family} is not fit.")
        distributions.append(np.sort(stationary_distribution(params.transition_probs)))
    matrix = np.zeros((len(families), len(families)), dtype=float)
    for row, left in enumerate(distributions):
        for col, right in enumerate(distributions):
            matrix[row, col] = float(np.sum(np.abs(left - right)))
    return families, matrix


def cross_family_likelihood_matrix(
    models: Mapping[str, object],
    sequences_by_family: Mapping[str, list[np.ndarray]],
) -> tuple[list[str], list[str], np.ndarray]:
    """Per-residue log-likelihood of model[i] on sequences from family[j].

    Returns ``(model_families, test_families, matrix)``.
    """
    test_families = sorted(sequences_by_family)
    model_families = sorted(models)
    matrix = np.zeros((len(model_families), len(test_families)), dtype=float)
    for row, family_i in enumerate(model_families):
        model = models[family_i]
        for col, family_j in enumerate(test_families):
            sequences = sequences_by_family[family_j]
            total_length = sum(len(sequence) for sequence in sequences) or 1
            matrix[row, col] = model.score_many(sequences) / total_length
    return model_families, test_families, matrix
