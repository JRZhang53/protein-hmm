from __future__ import annotations

import unittest

import numpy as np

from support import bootstrap

bootstrap()

from protein_hmm.models.baselines import IIDCategoricalModel, ObservedMarkovChain


def _categorical_sequences(rng: np.random.Generator, n_sequences: int = 3, length: int = 20) -> list[np.ndarray]:
    return [rng.integers(0, 4, size=length) for _ in range(n_sequences)]


class BaselineModelTests(unittest.TestCase):
    def test_iid_score_many_matches_sum_of_score(self) -> None:
        sequences = _categorical_sequences(np.random.default_rng(0))
        model = IIDCategoricalModel(alphabet_size=4).fit(sequences)
        self.assertAlmostEqual(
            model.score_many(sequences),
            sum(model.score(sequence) for sequence in sequences),
            places=8,
        )

    def test_observed_markov_score_many_matches_sum_of_score(self) -> None:
        sequences = _categorical_sequences(np.random.default_rng(1))
        model = ObservedMarkovChain(alphabet_size=4).fit(sequences)
        self.assertAlmostEqual(
            model.score_many(sequences),
            sum(model.score(sequence) for sequence in sequences),
            places=8,
        )


if __name__ == "__main__":
    unittest.main()
