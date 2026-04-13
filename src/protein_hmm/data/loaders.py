"""Load and save protein datasets."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from protein_hmm.data.alignment import ResidueAligner
from protein_hmm.types import ProteinRecord
from protein_hmm.utils.io import read_jsonl, write_jsonl


def parse_fasta_header(header: str) -> tuple[str, dict[str, str]]:
    tokens = header.strip().split()
    protein_id = tokens[0]
    metadata: dict[str, str] = {"header": header.strip()}
    for token in tokens[1:]:
        if "=" in token:
            key, value = token.split("=", 1)
            metadata[key] = value
    return protein_id, metadata


def load_fasta_records(path: str | Path, default_family: str = "unknown") -> list[ProteinRecord]:
    fasta_path = Path(path)
    if not fasta_path.exists():
        raise FileNotFoundError(fasta_path)

    records: list[ProteinRecord] = []
    current_header: str | None = None
    current_sequence: list[str] = []
    for line in fasta_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        if line.startswith(">"):
            if current_header is not None:
                protein_id, metadata = parse_fasta_header(current_header)
                records.append(
                    ProteinRecord(
                        protein_id=protein_id,
                        family=metadata.get("family", default_family),
                        sequence="".join(current_sequence),
                        metadata=metadata,
                    )
                )
            current_header = line[1:]
            current_sequence = []
        else:
            current_sequence.append(line.strip())

    if current_header is not None:
        protein_id, metadata = parse_fasta_header(current_header)
        records.append(
            ProteinRecord(
                protein_id=protein_id,
                family=metadata.get("family", default_family),
                sequence="".join(current_sequence),
                metadata=metadata,
            )
        )
    return records


def load_annotation_table(path: str | Path) -> dict[str, dict[str, str]]:
    annotation_path = Path(path)
    if not annotation_path.exists():
        raise FileNotFoundError(annotation_path)
    with annotation_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["protein_id"]: row for row in reader if row.get("protein_id")}


def attach_annotations(
    records: Iterable[ProteinRecord],
    annotations: dict[str, dict[str, str]],
    aligner: ResidueAligner | None = None,
) -> list[ProteinRecord]:
    aligner = aligner or ResidueAligner()
    merged: list[ProteinRecord] = []
    for record in records:
        annotation = annotations.get(record.protein_id)
        if annotation is None:
            merged.append(record)
            continue

        labels = annotation.get("labels")
        structure_sequence = annotation.get("structure_sequence")
        family = annotation.get("family") or record.family
        merged_record = ProteinRecord(
            protein_id=record.protein_id,
            family=family,
            sequence=record.sequence,
            labels=record.labels,
            metadata={**record.metadata, **annotation},
        )
        if labels and structure_sequence:
            merged.append(aligner.align_record(merged_record, structure_sequence, labels))
        elif labels and len(labels) == record.length:
            merged_record.labels = labels
            merged.append(merged_record)
        else:
            merged.append(merged_record)
    return merged


def save_records(path: str | Path, records: Iterable[ProteinRecord]) -> None:
    write_jsonl(path, [record.to_dict() for record in records])


def load_records(path: str | Path) -> list[ProteinRecord]:
    return [ProteinRecord.from_dict(row) for row in read_jsonl(path)]


def save_split_records(processed_dir: str | Path, splits: dict[str, list[ProteinRecord]]) -> None:
    root = Path(processed_dir)
    for split_name, records in splits.items():
        save_records(root / f"{split_name}.jsonl", records)


def load_split_records(processed_dir: str | Path) -> dict[str, list[ProteinRecord]]:
    root = Path(processed_dir)
    splits: dict[str, list[ProteinRecord]] = {}
    for split_name in ("train", "val", "test"):
        split_path = root / f"{split_name}.jsonl"
        splits[split_name] = load_records(split_path) if split_path.exists() else []
    return splits
