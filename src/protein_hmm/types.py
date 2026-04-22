"""Lightweight shared datatypes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class ProteinRecord:
    protein_id: str
    family: str
    sequence: str
    labels: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sequence = self.sequence.upper().strip()
        self.family = self.family or "unknown"
        if self.labels is not None:
            self.labels = self.labels.upper().strip()

    @property
    def length(self) -> int:
        return len(self.sequence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protein_id": self.protein_id,
            "family": self.family,
            "sequence": self.sequence,
            "labels": self.labels,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProteinRecord":
        if "alignment_score" in payload or "structure_sequence" in payload:
            return AlignedProteinRecord.from_dict(payload)
        return cls(
            protein_id=payload["protein_id"],
            family=payload.get("family", "unknown"),
            sequence=payload["sequence"],
            labels=payload.get("labels"),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class AlignedProteinRecord(ProteinRecord):
    structure_sequence: str | None = None
    alignment_score: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        payload = super(AlignedProteinRecord, self).to_dict()
        payload["structure_sequence"] = self.structure_sequence
        payload["alignment_score"] = self.alignment_score
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AlignedProteinRecord":
        return cls(
            protein_id=payload["protein_id"],
            family=payload.get("family", "unknown"),
            sequence=payload["sequence"],
            labels=payload.get("labels"),
            metadata=dict(payload.get("metadata", {})),
            structure_sequence=payload.get("structure_sequence"),
            alignment_score=float(payload.get("alignment_score", 1.0)),
        )


@dataclass(slots=True)
class DecodedSequence:
    protein_id: str
    states: list[int]
    log_likelihood: float
    labels: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "protein_id": self.protein_id,
            "states": self.states,
            "log_likelihood": self.log_likelihood,
            "labels": self.labels,
        }


@dataclass(slots=True)
class HMMParameters:
    start_probs: np.ndarray
    transition_probs: np.ndarray
    emission_probs: np.ndarray

    def copy(self) -> "HMMParameters":
        return HMMParameters(
            start_probs=self.start_probs.copy(),
            transition_probs=self.transition_probs.copy(),
            emission_probs=self.emission_probs.copy(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_probs": self.start_probs.tolist(),
            "transition_probs": self.transition_probs.tolist(),
            "emission_probs": self.emission_probs.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HMMParameters":
        return cls(
            start_probs=np.asarray(payload["start_probs"], dtype=float),
            transition_probs=np.asarray(payload["transition_probs"], dtype=float),
            emission_probs=np.asarray(payload["emission_probs"], dtype=float),
        )


@dataclass(slots=True)
class TrainingHistory:
    log_likelihoods: list[float] = field(default_factory=list)
    converged: bool = False
    iterations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "log_likelihoods": self.log_likelihoods,
            "converged": self.converged,
            "iterations": self.iterations,
        }
