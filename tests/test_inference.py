from __future__ import annotations

import itertools
import unittest

import numpy as np

from support import bootstrap

bootstrap()

from protein_hmm.inference.baum_welch import baum_welch
from protein_hmm.inference.forward_backward import forward_backward
from protein_hmm.inference.viterbi import viterbi_decode


class InferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.start = np.asarray([0.6, 0.4], dtype=float)
        self.transition = np.asarray([[0.7, 0.3], [0.4, 0.6]], dtype=float)
        self.emission = np.asarray([[0.5, 0.5], [0.1, 0.9]], dtype=float)
        self.observations = np.asarray([0, 1, 1], dtype=int)

    def brute_force_log_likelihood(self) -> float:
        total = 0.0
        for path in itertools.product(range(2), repeat=len(self.observations)):
            prob = self.start[path[0]] * self.emission[path[0], self.observations[0]]
            for step in range(1, len(path)):
                prob *= self.transition[path[step - 1], path[step]]
                prob *= self.emission[path[step], self.observations[step]]
            total += prob
        return float(np.log(total))

    def brute_force_viterbi(self) -> tuple[list[int], float]:
        best_path = None
        best_prob = -1.0
        for path in itertools.product(range(2), repeat=len(self.observations)):
            prob = self.start[path[0]] * self.emission[path[0], self.observations[0]]
            for step in range(1, len(path)):
                prob *= self.transition[path[step - 1], path[step]]
                prob *= self.emission[path[step], self.observations[step]]
            if prob > best_prob:
                best_prob = prob
                best_path = list(path)
        assert best_path is not None
        return best_path, float(np.log(best_prob))

    def test_forward_backward_matches_bruteforce_likelihood(self) -> None:
        result = forward_backward(self.start, self.transition, self.emission, self.observations)
        self.assertAlmostEqual(result.log_likelihood, self.brute_force_log_likelihood(), places=8)
        np.testing.assert_allclose(result.posterior.sum(axis=1), np.ones(len(self.observations)))

    def test_viterbi_matches_bruteforce_path(self) -> None:
        path, score = viterbi_decode(self.start, self.transition, self.emission, self.observations)
        expected_path, expected_score = self.brute_force_viterbi()
        self.assertEqual(path, expected_path)
        self.assertAlmostEqual(score, expected_score, places=8)

    def test_baum_welch_monotonic_history(self) -> None:
        sequences = [
            np.asarray([0, 0, 0, 1, 1], dtype=int),
            np.asarray([1, 1, 1, 0, 0], dtype=int),
            np.asarray([0, 0, 1, 1, 1], dtype=int),
            np.asarray([1, 1, 0, 0, 0], dtype=int),
        ]
        params, history = baum_welch(
            sequences=sequences,
            num_states=2,
            alphabet_size=2,
            max_iter=6,
            tol=1e-8,
            random_state=3,
        )
        self.assertGreaterEqual(history.iterations, 1)
        for previous, current in zip(history.log_likelihoods, history.log_likelihoods[1:]):
            self.assertGreaterEqual(current + 1e-8, previous)
        np.testing.assert_allclose(params.transition_probs.sum(axis=1), np.ones(2))
        np.testing.assert_allclose(params.emission_probs.sum(axis=1), np.ones(2))


if __name__ == "__main__":
    unittest.main()
