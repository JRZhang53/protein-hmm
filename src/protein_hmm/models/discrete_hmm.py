"""Object-oriented wrapper around the discrete HMM algorithms."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from protein_hmm.inference.baum_welch import baum_welch
from protein_hmm.inference.forward_backward import forward_backward
from protein_hmm.inference.viterbi import viterbi_decode
from protein_hmm.types import DecodedSequence, HMMParameters, TrainingHistory
from protein_hmm.utils.io import read_json, write_json


class DiscreteHMM:
    def __init__(
        self,
        num_states: int,
        alphabet_size: int = 20,
        max_iter: int = 25,
        tol: float = 1e-3,
        pseudocount: float = 1e-3,
        random_state: int | None = None,
    ) -> None:
        self.num_states = num_states
        self.alphabet_size = alphabet_size
        self.max_iter = max_iter
        self.tol = tol
        self.pseudocount = pseudocount
        self.random_state = random_state
        self.params: HMMParameters | None = None
        self.training_history = TrainingHistory()

    def fit(
        self,
        sequences: Iterable[np.ndarray],
        state_masks: list[np.ndarray] | None = None,
        initial_params: HMMParameters | None = None,
    ) -> "DiscreteHMM":
        encoded_sequences = [np.asarray(sequence, dtype=int) for sequence in sequences]
        self.params, self.training_history = baum_welch(
            sequences=encoded_sequences,
            num_states=self.num_states,
            alphabet_size=self.alphabet_size,
            max_iter=self.max_iter,
            tol=self.tol,
            pseudocount=self.pseudocount,
            random_state=self.random_state,
            initial_params=initial_params,
            state_masks=state_masks,
        )
        return self

    def _require_params(self) -> HMMParameters:
        if self.params is None:
            raise RuntimeError("Model must be fit before inference.")
        return self.params

    def score(self, sequence: np.ndarray, state_mask: np.ndarray | None = None) -> float:
        params = self._require_params()
        return forward_backward(
            start_probs=params.start_probs,
            transition_probs=params.transition_probs,
            emission_probs=params.emission_probs,
            observations=np.asarray(sequence, dtype=int),
            state_mask=state_mask,
        ).log_likelihood

    def score_many(self, sequences: Iterable[np.ndarray]) -> float:
        return float(sum(self.score(sequence) for sequence in sequences))

    def posterior_marginals(self, sequence: np.ndarray, state_mask: np.ndarray | None = None) -> np.ndarray:
        params = self._require_params()
        return forward_backward(
            start_probs=params.start_probs,
            transition_probs=params.transition_probs,
            emission_probs=params.emission_probs,
            observations=np.asarray(sequence, dtype=int),
            state_mask=state_mask,
        ).posterior

    def decode(
        self,
        sequence: np.ndarray,
        protein_id: str = "sequence",
        labels: str | None = None,
        state_mask: np.ndarray | None = None,
    ) -> DecodedSequence:
        params = self._require_params()
        states, log_likelihood = viterbi_decode(
            start_probs=params.start_probs,
            transition_probs=params.transition_probs,
            emission_probs=params.emission_probs,
            observations=np.asarray(sequence, dtype=int),
            state_mask=state_mask,
        )
        return DecodedSequence(
            protein_id=protein_id,
            states=states,
            log_likelihood=log_likelihood,
            labels=labels,
        )

    def parameter_count(self) -> int:
        return (
            (self.num_states - 1)
            + self.num_states * (self.num_states - 1)
            + self.num_states * (self.alphabet_size - 1)
        )

    def bic(self, sequences: Iterable[np.ndarray]) -> float:
        sequences = [np.asarray(sequence, dtype=int) for sequence in sequences]
        total_observations = sum(len(sequence) for sequence in sequences)
        if total_observations == 0:
            raise ValueError("BIC requires at least one observed residue.")
        log_likelihood = self.score_many(sequences)
        return -2.0 * log_likelihood + self.parameter_count() * np.log(total_observations)

    def save(self, path: str | Path) -> None:
        params = self._require_params()
        write_json(
            path,
            {
                "num_states": self.num_states,
                "alphabet_size": self.alphabet_size,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "pseudocount": self.pseudocount,
                "random_state": self.random_state,
                "params": params.to_dict(),
                "training_history": self.training_history.to_dict(),
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> "DiscreteHMM":
        payload = read_json(path)
        model = cls(
            num_states=int(payload["num_states"]),
            alphabet_size=int(payload["alphabet_size"]),
            max_iter=int(payload["max_iter"]),
            tol=float(payload["tol"]),
            pseudocount=float(payload["pseudocount"]),
            random_state=payload.get("random_state"),
        )
        model.params = HMMParameters.from_dict(payload["params"])
        history = payload.get("training_history", {})
        model.training_history = TrainingHistory(
            log_likelihoods=list(history.get("log_likelihoods", [])),
            converged=bool(history.get("converged", False)),
            iterations=int(history.get("iterations", 0)),
        )
        return model
