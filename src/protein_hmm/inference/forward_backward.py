"""Log-space forward-backward inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LOG_FLOOR = 1e-300


@dataclass(slots=True)
class ForwardBackwardResult:
    log_likelihood: float
    log_alpha: np.ndarray
    log_beta: np.ndarray
    posterior: np.ndarray
    pairwise_posterior: np.ndarray


def logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    max_value = np.max(values, axis=axis, keepdims=True)
    shifted = values - max_value
    summed = np.sum(np.exp(shifted), axis=axis, keepdims=True)
    result = max_value + np.log(np.clip(summed, LOG_FLOOR, None))
    if axis is None:
        return result.reshape(())
    return np.squeeze(result, axis=axis)


def _to_log_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    return np.log(np.clip(probabilities, LOG_FLOOR, None))


def _state_log_mask(state_mask: np.ndarray | None, num_steps: int, num_states: int) -> np.ndarray:
    if state_mask is None:
        return np.zeros((num_steps, num_states), dtype=float)
    mask = np.asarray(state_mask)
    if mask.shape != (num_steps, num_states):
        raise ValueError("State mask must have shape (T, K).")
    if mask.dtype == bool:
        if np.any(np.all(~mask, axis=1)):
            raise ValueError("Every timestep must allow at least one state.")
        return np.where(mask, 0.0, -np.inf)
    return mask.astype(float)


def observation_log_likelihoods(observations: np.ndarray, emission_probs: np.ndarray) -> np.ndarray:
    observations = np.asarray(observations, dtype=int)
    if observations.ndim != 1:
        raise ValueError("Observations must be one-dimensional.")
    emission_probs = np.asarray(emission_probs, dtype=float)
    return _to_log_probabilities(emission_probs[:, observations]).T


def forward_backward(
    start_probs: np.ndarray,
    transition_probs: np.ndarray,
    emission_probs: np.ndarray,
    observations: np.ndarray,
    state_mask: np.ndarray | None = None,
) -> ForwardBackwardResult:
    observations = np.asarray(observations, dtype=int)
    num_steps = len(observations)
    num_states = np.asarray(start_probs).shape[0]
    if num_steps == 0:
        raise ValueError("Cannot run inference on an empty sequence.")

    log_start = _to_log_probabilities(start_probs)
    log_transition = _to_log_probabilities(transition_probs)
    log_observations = observation_log_likelihoods(observations, emission_probs)
    mask = _state_log_mask(state_mask, num_steps, num_states)

    log_alpha = np.full((num_steps, num_states), -np.inf, dtype=float)
    log_beta = np.full((num_steps, num_states), -np.inf, dtype=float)

    log_alpha[0] = log_start + log_observations[0] + mask[0]
    for step in range(1, num_steps):
        scores = log_alpha[step - 1][:, None] + log_transition
        log_alpha[step] = log_observations[step] + mask[step] + logsumexp(scores, axis=0)

    log_likelihood = float(logsumexp(log_alpha[-1], axis=0))
    if not np.isfinite(log_likelihood):
        raise ValueError("Sequence has zero probability under the model or mask.")

    log_beta[-1] = 0.0
    for step in range(num_steps - 2, -1, -1):
        scores = (
            log_transition
            + log_observations[step + 1][None, :]
            + mask[step + 1][None, :]
            + log_beta[step + 1][None, :]
        )
        log_beta[step] = logsumexp(scores, axis=1)

    log_posterior = log_alpha + log_beta - log_likelihood
    posterior = np.exp(log_posterior)
    posterior /= posterior.sum(axis=1, keepdims=True)

    pairwise_posterior = np.zeros((max(num_steps - 1, 0), num_states, num_states), dtype=float)
    for step in range(num_steps - 1):
        scores = (
            log_alpha[step][:, None]
            + log_transition
            + log_observations[step + 1][None, :]
            + mask[step + 1][None, :]
            + log_beta[step + 1][None, :]
            - log_likelihood
        )
        pairwise_posterior[step] = np.exp(scores)
        pairwise_posterior[step] /= np.clip(pairwise_posterior[step].sum(), LOG_FLOOR, None)

    return ForwardBackwardResult(
        log_likelihood=log_likelihood,
        log_alpha=log_alpha,
        log_beta=log_beta,
        posterior=posterior,
        pairwise_posterior=pairwise_posterior,
    )
