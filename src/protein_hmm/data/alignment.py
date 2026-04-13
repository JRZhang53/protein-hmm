"""Residue-level alignment helpers for sequence and DSSP labels."""

from __future__ import annotations

from dataclasses import dataclass

from protein_hmm.constants import DSSP_COLLAPSE_MAP
from protein_hmm.types import AlignedProteinRecord, ProteinRecord


class AlignmentError(ValueError):
    """Raised when residue-level sequence alignment fails."""


@dataclass(slots=True)
class ResidueAligner:
    gap_tokens: frozenset[str] = frozenset({"-", ".", "_"})

    def normalize_labels(self, labels: str) -> str:
        return "".join(DSSP_COLLAPSE_MAP.get(label.upper(), "C") for label in labels.strip())

    def align_labels(self, sequence: str, structure_sequence: str, labels: str) -> str:
        sequence = sequence.upper().strip()
        structure_sequence = structure_sequence.upper().strip()
        normalized_labels = self.normalize_labels(labels)
        if len(structure_sequence) != len(normalized_labels):
            raise AlignmentError("Structure sequence and label string must have identical length.")

        aligned_labels: list[str] = []
        sequence_index = 0
        for residue, label in zip(structure_sequence, normalized_labels):
            if residue in self.gap_tokens:
                continue
            if sequence_index >= len(sequence):
                raise AlignmentError("Structure sequence is longer than primary sequence after trimming gaps.")
            if residue != sequence[sequence_index]:
                raise AlignmentError(
                    f"Residue mismatch at position {sequence_index}: "
                    f"{residue!r} != {sequence[sequence_index]!r}."
                )
            aligned_labels.append(label)
            sequence_index += 1

        if sequence_index != len(sequence):
            raise AlignmentError("Structure sequence does not fully cover the primary sequence.")
        return "".join(aligned_labels)

    def align_record(
        self,
        record: ProteinRecord,
        structure_sequence: str,
        labels: str,
        alignment_score: float = 1.0,
    ) -> AlignedProteinRecord:
        aligned_labels = self.align_labels(record.sequence, structure_sequence, labels)
        return AlignedProteinRecord(
            protein_id=record.protein_id,
            family=record.family,
            sequence=record.sequence,
            labels=aligned_labels,
            metadata=dict(record.metadata),
            structure_sequence=structure_sequence,
            alignment_score=alignment_score,
        )
