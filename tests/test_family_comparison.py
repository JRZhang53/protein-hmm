from __future__ import annotations

import unittest

import numpy as np

from support import bootstrap

bootstrap()

from protein_hmm.analysis.family_comparison import (
    expected_dwell_times,
    hungarian_state_permutation,
    permute_parameters,
    stationary_distance_matrix,
    stationary_distribution,
    transition_distance_matrix,
)
from protein_hmm.types import HMMParameters


class _DummyModel:
    def __init__(self, params: HMMParameters) -> None:
        self.params = params


def _make_params(emission: np.ndarray, transition: np.ndarray) -> HMMParameters:
    num_states = emission.shape[0]
    return HMMParameters(
        start_probs=np.full(num_states, 1.0 / num_states),
        transition_probs=transition,
        emission_probs=emission,
    )


class FamilyComparisonTests(unittest.TestCase):
    def test_hungarian_recovers_permutation(self) -> None:
        emission = np.asarray([[0.9, 0.1], [0.2, 0.8]])
        transition = np.asarray([[0.7, 0.3], [0.4, 0.6]])
        original = _make_params(emission, transition)
        permuted = permute_parameters(original, np.asarray([1, 0]))
        recovered = hungarian_state_permutation(original, permuted)
        np.testing.assert_array_equal(recovered, np.asarray([1, 0]))

    def test_aligned_distance_is_zero_for_relabeled_copy(self) -> None:
        emission = np.asarray([[0.7, 0.3], [0.1, 0.9]])
        transition = np.asarray([[0.6, 0.4], [0.5, 0.5]])
        original = _make_params(emission, transition)
        permuted = permute_parameters(original, np.asarray([1, 0]))
        models = {"A": _DummyModel(original), "B": _DummyModel(permuted)}
        _, aligned = transition_distance_matrix(models, align_states=True)
        _, raw = transition_distance_matrix(models, align_states=False)
        self.assertAlmostEqual(aligned[0, 1], 0.0, places=8)
        self.assertGreater(raw[0, 1], 0.0)

    def test_stationary_distribution_is_normalized(self) -> None:
        transition = np.asarray([[0.7, 0.3], [0.4, 0.6]])
        distribution = stationary_distribution(transition)
        self.assertAlmostEqual(distribution.sum(), 1.0, places=8)
        np.testing.assert_allclose(distribution @ transition, distribution, atol=1e-6)

    def test_stationary_distance_matrix_is_symmetric_and_zero_on_diagonal(self) -> None:
        emission = np.asarray([[0.7, 0.3], [0.1, 0.9]])
        models = {
            "A": _DummyModel(_make_params(emission, np.asarray([[0.7, 0.3], [0.4, 0.6]]))),
            "B": _DummyModel(_make_params(emission, np.asarray([[0.5, 0.5], [0.2, 0.8]]))),
        }
        _, matrix = stationary_distance_matrix(models)
        self.assertAlmostEqual(matrix[0, 0], 0.0)
        self.assertAlmostEqual(matrix[1, 1], 0.0)
        self.assertAlmostEqual(matrix[0, 1], matrix[1, 0])

    def test_expected_dwell_times_match_geometric(self) -> None:
        transition = np.asarray([[0.9, 0.1], [0.5, 0.5]])
        dwell = expected_dwell_times(transition)
        np.testing.assert_allclose(dwell, np.asarray([10.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
