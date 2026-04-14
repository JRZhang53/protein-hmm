"""Viterbi decoding for discrete HMMs."""

from __future__ import annotations

import numpy as np

from protein_hmm.inference.forward_backward import _state_log_mask, _to_log_probabilities, observation_log_likelihoods


def viterbi_decode(
    start_probs: np.ndarray,
    transition_probs: np.ndarray,
    emission_probs: np.ndarray,
    observations: np.ndarray,
    state_mask: np.ndarray | None = None,
) -> tuple[list[int], float]:
    observations = np.asarray(observations, dtype=int)
    num_steps = len(observations)
    if num_steps == 0:
        raise ValueError("Cannot decode an empty sequence.")

    num_states = np.asarray(start_probs).shape[0]
    log_start = _to_log_probabilities(start_probs)
    log_transition = _to_log_probabilities(transition_probs)
    log_observations = observation_log_likelihoods(observations, emission_probs)
    mask = _state_log_mask(state_mask, num_steps, num_states)

    scores = np.full((num_steps, num_states), -np.inf, dtype=float)
    backpointers = np.zeros((num_steps, num_states), dtype=int)

    scores[0] = log_start + log_observations[0] + mask[0]
    for step in range(1, num_steps):
        transition_scores = scores[step - 1][:, None] + log_transition
        backpointers[step] = np.argmax(transition_scores, axis=0)
        scores[step] = (
            transition_scores[backpointers[step], np.arange(num_states)]
            + log_observations[step]
            + mask[step]
        )

    last_state = int(np.argmax(scores[-1]))
    best_score = float(scores[-1, last_state])
    path = [last_state]
    for step in range(num_steps - 1, 0, -1):
        path.append(int(backpointers[step, path[-1]]))
    path.reverse()
    return path, best_score
