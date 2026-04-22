"""Utilities for extracting matched records from Pfam seed alignments."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path


@dataclass(slots=True)
class PfamSeedRecord:
    protein_id: str
    accession: str
    family: str
    family_name: str
    start: int
    end: int
    sequence: str

    @property
    def length(self) -> int:
        return len(self.sequence)


def parse_seed_sequence_id(sequence_id: str) -> tuple[str, int, int]:
    accession_token, residue_span = sequence_id.rsplit("/", 1)
    accession = accession_token.split("_", 1)[0]
    start_text, end_text = residue_span.split("-", 1)
    return accession, int(start_text), int(end_text)


def _ungap_stockholm_sequence(sequence: str) -> str:
    return "".join(character for character in sequence if character.isalpha()).upper()


def load_seed_records(
    path: str | Path,
    families: set[str] | None = None,
    accessions_by_family: dict[str, set[str]] | None = None,
) -> list[PfamSeedRecord]:
    records: list[PfamSeedRecord] = []
    current_family: str | None = None
    current_family_name = "unknown"
    current_sequences: dict[str, list[str]] = {}

    def flush_records() -> None:
        if current_family is None:
            return
        family_accessions = None if accessions_by_family is None else accessions_by_family.get(current_family)
        for sequence_id, fragments in current_sequences.items():
            accession, start, end = parse_seed_sequence_id(sequence_id)
            if family_accessions is not None and accession not in family_accessions:
                continue
            sequence = _ungap_stockholm_sequence("".join(fragments))
            if not sequence:
                continue
            records.append(
                PfamSeedRecord(
                    protein_id=sequence_id,
                    accession=accession,
                    family=current_family,
                    family_name=current_family_name,
                    start=start,
                    end=end,
                    sequence=sequence,
                )
            )

    with gzip.open(Path(path), "rt", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.startswith("#=GF AC"):
                current_family = line.split()[-1].rstrip(";").split(".", 1)[0]
            elif line.startswith("#=GF ID"):
                current_family_name = line.split()[-1]
            elif line == "//":
                flush_records()
                current_family = None
                current_family_name = "unknown"
                current_sequences = {}
            elif line.startswith("#") or not line.strip():
                continue
            else:
                if current_family is None:
                    continue
                if families is not None and current_family not in families:
                    continue
                sequence_id, alignment_fragment = line.split(maxsplit=1)
                current_sequences.setdefault(sequence_id, []).append(alignment_fragment.strip())

    return records
