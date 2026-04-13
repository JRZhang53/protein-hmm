"""Dataset preprocessing and summary utilities."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from protein_hmm.constants import AMINO_ACIDS, VALID_AMINO_ACIDS, DSSP_COLLAPSE_MAP
from protein_hmm.types import ProteinRecord


def collapse_dssp_labels(labels: str | None) -> str | None:
    if labels is None:
        return None
    return "".join(DSSP_COLLAPSE_MAP.get(label.upper(), "C") for label in labels.strip())


@dataclass(slots=True)
class DatasetPreprocessor:
    min_length: int = 1
    max_length: int | None = None
    drop_unknown_residues: bool = True

    def clean(self, records: Iterable[ProteinRecord]) -> list[ProteinRecord]:
        cleaned: list[ProteinRecord] = []
        for record in records:
            if record.length < self.min_length:
                continue
            if self.max_length is not None and record.length > self.max_length:
                continue
            if self.drop_unknown_residues and any(residue not in VALID_AMINO_ACIDS for residue in record.sequence):
                continue

            normalized_labels = collapse_dssp_labels(record.labels)
            if normalized_labels is not None and len(normalized_labels) != record.length:
                continue

            payload = record.to_dict()
            payload["labels"] = normalized_labels
            cleaned.append(ProteinRecord.from_dict(payload))
        return cleaned


def group_by_family(records: Iterable[ProteinRecord]) -> dict[str, list[ProteinRecord]]:
    grouped: dict[str, list[ProteinRecord]] = defaultdict(list)
    for record in records:
        grouped[record.family].append(record)
    return dict(grouped)


def summarize_records(records: Iterable[ProteinRecord]) -> dict[str, object]:
    record_list = list(records)
    lengths = [record.length for record in record_list]
    amino_acid_counts = Counter(residue for record in record_list for residue in record.sequence)
    family_counts = Counter(record.family for record in record_list)
    total_residues = sum(amino_acid_counts.values()) or 1
    amino_acid_frequencies = {
        residue: amino_acid_counts.get(residue, 0) / total_residues for residue in AMINO_ACIDS
    }
    return {
        "num_proteins": len(record_list),
        "length_summary": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": sum(lengths) / len(lengths) if lengths else 0.0,
        },
        "family_counts": dict(sorted(family_counts.items())),
        "amino_acid_frequencies": amino_acid_frequencies,
    }
