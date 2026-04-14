"""Constrained reference HMM using partially observed DSSP labels."""

from __future__ import annotations

import numpy as np

from protein_hmm.constants import DSSP_COLLAPSE_MAP, DSSP_LABELS
from protein_hmm.models.discrete_hmm import DiscreteHMM
from protein_hmm.types import HMMParameters, TrainingHistory
from protein_hmm.utils.io import read_json, write_json


class SemiSupervisedHMM(DiscreteHMM):
    def __init__(
        self,
        state_labels: tuple[str, ...] = DSSP_LABELS,
        alphabet_size: int = 20,
        max_iter: int = 20,
        tol: float = 1e-3,
        pseudocount: float = 1e-2,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            num_states=len(state_labels),
            alphabet_size=alphabet_size,
            max_iter=max_iter,
            tol=tol,
            pseudocount=pseudocount,
            random_state=random_state,
        )
        self.state_labels = tuple(state_labels)
        self.label_to_state = {label: index for index, label in enumerate(self.state_labels)}

    def _normalize_label(self, label: str) -> str | None:
        normalized = DSSP_COLLAPSE_MAP.get(label.upper())
        if normalized in self.label_to_state:
            return normalized
        return None

    def build_state_masks(self, label_sequences: list[str]) -> list[np.ndarray]:
        masks: list[np.ndarray] = []
        for labels in label_sequences:
            mask = np.ones((len(labels), self.num_states), dtype=bool)
            for step, label in enumerate(labels):
                normalized = self._normalize_label(label)
                if normalized is None:
                    continue
                mask[step] = False
                mask[step, self.label_to_state[normalized]] = True
            masks.append(mask)
        return masks

    def initial_params_from_labels(
        self,
        sequences: list[np.ndarray],
        label_sequences: list[str],
    ) -> HMMParameters:
        start_counts = np.full(self.num_states, self.pseudocount, dtype=float)
        transition_counts = np.full((self.num_states, self.num_states), self.pseudocount, dtype=float)
        emission_counts = np.full((self.num_states, self.alphabet_size), self.pseudocount, dtype=float)

        for sequence, labels in zip(sequences, label_sequences):
            previous_state: int | None = None
            for step, (symbol, label) in enumerate(zip(sequence, labels)):
                normalized = self._normalize_label(label)
                if normalized is None:
                    previous_state = None
                    continue
                state = self.label_to_state[normalized]
                emission_counts[state, int(symbol)] += 1.0
                if step == 0:
                    start_counts[state] += 1.0
                if previous_state is not None:
                    transition_counts[previous_state, state] += 1.0
                previous_state = state

        return HMMParameters(
            start_probs=start_counts / start_counts.sum(),
            transition_probs=transition_counts / transition_counts.sum(axis=1, keepdims=True),
            emission_probs=emission_counts / emission_counts.sum(axis=1, keepdims=True),
        )

    def fit(self, sequences: list[np.ndarray], label_sequences: list[str]) -> "SemiSupervisedHMM":
        encoded_sequences = [np.asarray(sequence, dtype=int) for sequence in sequences]
        if len(encoded_sequences) != len(label_sequences):
            raise ValueError("label_sequences must align with sequences.")
        for sequence, labels in zip(encoded_sequences, label_sequences):
            if len(sequence) != len(labels):
                raise ValueError("Every label sequence must match the corresponding observation length.")

        initial_params = self.initial_params_from_labels(encoded_sequences, label_sequences)
        state_masks = self.build_state_masks(label_sequences)
        super().fit(encoded_sequences, state_masks=state_masks, initial_params=initial_params)
        return self

    def predict_labels(self, sequence: np.ndarray) -> str:
        decoded = self.decode(sequence)
        return "".join(self.state_labels[state] for state in decoded.states)

    def save(self, path: str) -> None:
        params = self._require_params()
        write_json(
            path,
            {
                "state_labels": list(self.state_labels),
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
    def load(cls, path: str) -> "SemiSupervisedHMM":
        payload = read_json(path)
        model = cls(
            state_labels=tuple(payload.get("state_labels", DSSP_LABELS)),
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
