"""Baum-Welch learning for categorical HMMs."""

from __future__ import annotations

import numpy as np

from protein_hmm.inference.forward_backward import ForwardBackwardResult, forward_backward
from protein_hmm.types import HMMParameters, TrainingHistory
from protein_hmm.utils.random_state import get_rng


def _normalize(vector: np.ndarray) -> np.ndarray:
    total = float(np.sum(vector))
    if total <= 0.0:
        return np.full_like(vector, 1.0 / len(vector))
    return vector / total


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    zero_rows = row_sums.squeeze(axis=1) <= 0.0
    row_sums[zero_rows] = 1.0
    normalized = matrix / row_sums
    if np.any(zero_rows):
        normalized[zero_rows] = 1.0 / matrix.shape[1]
    return normalized


def initialize_random_parameters(
    num_states: int,
    alphabet_size: int,
    random_state: int | np.random.Generator | None = None,
) -> HMMParameters:
    rng = get_rng(random_state)
    start_probs = rng.dirichlet(np.ones(num_states))
    transition_probs = rng.dirichlet(np.ones(num_states), size=num_states)
    emission_probs = rng.dirichlet(np.ones(alphabet_size), size=num_states)
    return HMMParameters(
        start_probs=start_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
    )


def baum_welch(
    sequences: list[np.ndarray],
    num_states: int,
    alphabet_size: int,
    max_iter: int = 200,
    tol: float = 1e-3,
    pseudocount: float = 1e-3,
    random_state: int | np.random.Generator | None = None,
    initial_params: HMMParameters | None = None,
    state_masks: list[np.ndarray] | None = None,
    convergence: str = "per_observation",
) -> tuple[HMMParameters, TrainingHistory]:
    if not sequences:
        raise ValueError("At least one sequence is required for Baum-Welch.")
    if convergence not in {"per_observation", "absolute"}:
        raise ValueError("convergence must be 'per_observation' or 'absolute'.")
    encoded_sequences = [np.asarray(sequence, dtype=int) for sequence in sequences]
    if state_masks is not None and len(state_masks) != len(encoded_sequences):
        raise ValueError("state_masks must align with sequences.")

    total_observations = sum(len(sequence) for sequence in encoded_sequences) or 1

    params = initial_params.copy() if initial_params is not None else initialize_random_parameters(
        num_states=num_states,
        alphabet_size=alphabet_size,
        random_state=random_state,
    )
    history = TrainingHistory()

    for _ in range(max_iter):
        start_counts = np.zeros(num_states, dtype=float)
        transition_counts = np.zeros((num_states, num_states), dtype=float)
        emission_counts = np.zeros((num_states, alphabet_size), dtype=float)
        total_log_likelihood = 0.0

        for index, sequence in enumerate(encoded_sequences):
            mask = None if state_masks is None else state_masks[index]
            result: ForwardBackwardResult = forward_backward(
                start_probs=params.start_probs,
                transition_probs=params.transition_probs,
                emission_probs=params.emission_probs,
                observations=sequence,
                state_mask=mask,
            )
            total_log_likelihood += result.log_likelihood
            start_counts += result.posterior[0]
            if len(sequence) > 1:
                transition_counts += result.pairwise_posterior.sum(axis=0)
            np.add.at(emission_counts, (slice(None), sequence), result.posterior.T)

        updated_params = HMMParameters(
            start_probs=_normalize(start_counts + pseudocount),
            transition_probs=_normalize_rows(transition_counts + pseudocount),
            emission_probs=_normalize_rows(emission_counts + pseudocount),
        )
        history.log_likelihoods.append(float(total_log_likelihood))
        params = updated_params

        if len(history.log_likelihoods) >= 2:
            delta = history.log_likelihoods[-1] - history.log_likelihoods[-2]
            if convergence == "per_observation":
                metric = abs(delta) / total_observations
            else:
                metric = abs(delta)
            if metric < tol:
                history.converged = True
                break

    history.iterations = len(history.log_likelihoods)
    return params, history


def baum_welch_restarts(
    sequences: list[np.ndarray],
    num_states: int,
    alphabet_size: int,
    n_restarts: int = 1,
    max_iter: int = 200,
    tol: float = 1e-3,
    pseudocount: float = 1e-3,
    random_state: int | np.random.Generator | None = None,
    initial_params: HMMParameters | None = None,
    state_masks: list[np.ndarray] | None = None,
    convergence: str = "per_observation",
) -> tuple[HMMParameters, TrainingHistory, list[float]]:
    """Run Baum-Welch from multiple random initializations and keep the best.

    The first restart uses ``initial_params`` (or the supplied ``random_state``).
    Subsequent restarts draw fresh seeds from the same generator so that runs
    are deterministic given a single integer seed.
    """
    if n_restarts < 1:
        raise ValueError("n_restarts must be at least 1.")

    rng = get_rng(random_state)
    best_params: HMMParameters | None = None
    best_history: TrainingHistory | None = None
    final_likelihoods: list[float] = []

    for restart_index in range(n_restarts):
        if restart_index == 0:
            params = baum_welch(
                sequences=sequences,
                num_states=num_states,
                alphabet_size=alphabet_size,
                max_iter=max_iter,
                tol=tol,
                pseudocount=pseudocount,
                random_state=rng,
                initial_params=initial_params,
                state_masks=state_masks,
                convergence=convergence,
            )
        else:
            params = baum_welch(
                sequences=sequences,
                num_states=num_states,
                alphabet_size=alphabet_size,
                max_iter=max_iter,
                tol=tol,
                pseudocount=pseudocount,
                random_state=rng,
                state_masks=state_masks,
                convergence=convergence,
            )
        candidate_params, history = params
        final_ll = history.log_likelihoods[-1] if history.log_likelihoods else float("-inf")
        final_likelihoods.append(final_ll)
        if best_history is None or final_ll > best_history.log_likelihoods[-1]:
            best_params, best_history = candidate_params, history

    assert best_params is not None and best_history is not None
    return best_params, best_history, final_likelihoods
