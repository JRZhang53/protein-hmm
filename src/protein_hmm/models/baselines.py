"""Simple baseline models for comparison against the latent HMM."""

from __future__ import annotations

import numpy as np

from protein_hmm.inference.forward_backward import LOG_FLOOR
from protein_hmm.utils.io import read_json, write_json


class IIDCategoricalModel:
    def __init__(self, alphabet_size: int = 20, pseudocount: float = 1e-3) -> None:
        self.alphabet_size = alphabet_size
        self.pseudocount = pseudocount
        self.probabilities: np.ndarray | None = None

    def fit(self, sequences: list[np.ndarray]) -> "IIDCategoricalModel":
        counts = np.full(self.alphabet_size, self.pseudocount, dtype=float)
        for sequence in sequences:
            sequence = np.asarray(sequence, dtype=int)
            counts += np.bincount(sequence, minlength=self.alphabet_size)
        self.probabilities = counts / counts.sum()
        return self

    def score(self, sequence: np.ndarray) -> float:
        if self.probabilities is None:
            raise RuntimeError("Model must be fit before scoring.")
        log_probs = np.log(np.clip(self.probabilities, LOG_FLOOR, None))
        return float(np.sum(log_probs[np.asarray(sequence, dtype=int)]))

    def score_many(self, sequences: list[np.ndarray]) -> float:
        return float(sum(self.score(sequence) for sequence in sequences))

    def save(self, path: str) -> None:
        if self.probabilities is None:
            raise RuntimeError("Model must be fit before saving.")
        write_json(path, {"alphabet_size": self.alphabet_size, "probabilities": self.probabilities.tolist()})

    @classmethod
    def load(cls, path: str) -> "IIDCategoricalModel":
        payload = read_json(path)
        model = cls(alphabet_size=int(payload["alphabet_size"]))
        model.probabilities = np.asarray(payload["probabilities"], dtype=float)
        return model


class ObservedMarkovChain:
    def __init__(self, alphabet_size: int = 20, pseudocount: float = 1e-3) -> None:
        self.alphabet_size = alphabet_size
        self.pseudocount = pseudocount
        self.start_probs: np.ndarray | None = None
        self.transition_probs: np.ndarray | None = None

    def fit(self, sequences: list[np.ndarray]) -> "ObservedMarkovChain":
        start_counts = np.full(self.alphabet_size, self.pseudocount, dtype=float)
        transition_counts = np.full((self.alphabet_size, self.alphabet_size), self.pseudocount, dtype=float)
        for sequence in sequences:
            sequence = np.asarray(sequence, dtype=int)
            if len(sequence) == 0:
                continue
            start_counts[int(sequence[0])] += 1.0
            for left, right in zip(sequence[:-1], sequence[1:]):
                transition_counts[int(left), int(right)] += 1.0

        self.start_probs = start_counts / start_counts.sum()
        self.transition_probs = transition_counts / transition_counts.sum(axis=1, keepdims=True)
        return self

    def score(self, sequence: np.ndarray) -> float:
        if self.start_probs is None or self.transition_probs is None:
            raise RuntimeError("Model must be fit before scoring.")
        sequence = np.asarray(sequence, dtype=int)
        if len(sequence) == 0:
            return 0.0
        log_start = np.log(np.clip(self.start_probs, LOG_FLOOR, None))
        log_transition = np.log(np.clip(self.transition_probs, LOG_FLOOR, None))
        total = float(log_start[int(sequence[0])])
        if len(sequence) > 1:
            total += float(np.sum(log_transition[sequence[:-1], sequence[1:]]))
        return total

    def score_many(self, sequences: list[np.ndarray]) -> float:
        return float(sum(self.score(sequence) for sequence in sequences))

    def save(self, path: str) -> None:
        if self.start_probs is None or self.transition_probs is None:
            raise RuntimeError("Model must be fit before saving.")
        write_json(
            path,
            {
                "alphabet_size": self.alphabet_size,
                "start_probs": self.start_probs.tolist(),
                "transition_probs": self.transition_probs.tolist(),
            },
        )

    @classmethod
    def load(cls, path: str) -> "ObservedMarkovChain":
        payload = read_json(path)
        model = cls(alphabet_size=int(payload["alphabet_size"]))
        model.start_probs = np.asarray(payload["start_probs"], dtype=float)
        model.transition_probs = np.asarray(payload["transition_probs"], dtype=float)
        return model
