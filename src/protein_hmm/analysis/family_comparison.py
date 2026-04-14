"""Model comparison across protein families."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from protein_hmm.analysis.metrics import transition_matrix_frobenius


def transition_distance_matrix(models: Mapping[str, object]) -> tuple[list[str], np.ndarray]:
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
            matrix[row, col] = transition_matrix_frobenius(
                params_i.transition_probs,
                params_j.transition_probs,
            )
    return families, matrix


def cross_family_likelihood_matrix(
    models: Mapping[str, object],
    sequences_by_family: Mapping[str, list[np.ndarray]],
) -> tuple[list[str], np.ndarray]:
    families = sorted(sequences_by_family)
    model_order = sorted(models)
    matrix = np.zeros((len(model_order), len(families)), dtype=float)
    for row, family_i in enumerate(model_order):
        model = models[family_i]
        for col, family_j in enumerate(families):
            sequences = sequences_by_family[family_j]
            total_length = sum(len(sequence) for sequence in sequences) or 1
            matrix[row, col] = model.score_many(sequences) / total_length
    return model_order, matrix
