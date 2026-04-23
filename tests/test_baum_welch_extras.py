from __future__ import annotations

import unittest

import numpy as np

from support import bootstrap

bootstrap()

from protein_hmm.inference.baum_welch import baum_welch, baum_welch_restarts


def _synthetic_two_state_sequences(rng: np.random.Generator, n_sequences: int = 8, length: int = 60) -> list[np.ndarray]:
    transition = np.asarray([[0.92, 0.08], [0.10, 0.90]])
    emission = np.asarray([[0.80, 0.05, 0.10, 0.05], [0.05, 0.10, 0.05, 0.80]])
    sequences: list[np.ndarray] = []
    for _ in range(n_sequences):
        state = int(rng.integers(2))
        observations = np.empty(length, dtype=int)
        for step in range(length):
            observations[step] = int(rng.choice(emission.shape[1], p=emission[state]))
            if step + 1 < length:
                state = int(rng.choice(2, p=transition[state]))
        sequences.append(observations)
    return sequences


class BaumWelchExtrasTests(unittest.TestCase):
    def test_per_observation_convergence_terminates(self) -> None:
        sequences = _synthetic_two_state_sequences(np.random.default_rng(1))
        _, history = baum_welch(
            sequences=sequences,
            num_states=2,
            alphabet_size=4,
            max_iter=200,
            tol=1e-5,
            random_state=2,
            convergence="per_observation",
        )
        self.assertTrue(history.converged)
        self.assertLess(history.iterations, 200)

    def test_restarts_pick_best_likelihood(self) -> None:
        sequences = _synthetic_two_state_sequences(np.random.default_rng(3))
        _, best_history, restart_lls = baum_welch_restarts(
            sequences=sequences,
            num_states=2,
            alphabet_size=4,
            n_restarts=4,
            max_iter=80,
            tol=1e-4,
            random_state=11,
        )
        self.assertEqual(len(restart_lls), 4)
        self.assertAlmostEqual(best_history.log_likelihoods[-1], max(restart_lls), places=8)


if __name__ == "__main__":
    unittest.main()
