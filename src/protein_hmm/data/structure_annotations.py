"""Build structure-linked Pfam subsets and DSSP annotation tables."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from protein_hmm.data.alignment import ResidueAligner
from protein_hmm.data.dssp import dssp_legacy_url, load_legacy_dssp
from protein_hmm.data.pfam_seed import PfamSeedRecord, load_seed_records
from protein_hmm.data.sifts import (
    build_family_accession_index,
    load_pfam_chain_mappings,
    load_sifts_residue_mappings,
    load_uniprot_chain_mappings,
    select_best_structure_mapping,
    sifts_xml_url,
)
from protein_hmm.utils.io import ensure_dir, write_json


DEFAULT_SELECTED_FAMILIES: tuple[str, ...] = (
    "PF01590",  # GAF
    "PF00210",  # Ferritin
    "PF02826",  # 2-Hacid_dh_C
    "PF07715",  # Plug
)


@dataclass(slots=True)
class AnnotationBuildResult:
    fasta_path: Path
    annotation_csv_path: Path
    summary_path: Path
    num_selected_records: int
    num_annotated_records: int
    selected_families: tuple[str, ...]
    family_counts: dict[str, int]


def _download_if_missing(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=60) as response:
        path.write_bytes(response.read())


def _build_annotation_strings(
    record: PfamSeedRecord,
    residue_map: dict[int, Any],
    dssp_map: dict[tuple[str, str], Any],
    chain_id: str,
) -> tuple[str, str, int, int]:
    structure_chars: list[str] = []
    label_chars: list[str] = []
    observed_count = 0
    fallback_count = 0

    for offset, sequence_residue in enumerate(record.sequence):
        uniprot_position = record.start + offset
        residue = residue_map.get(uniprot_position)
        if residue is None or not residue.observed or residue.pdb_resnum is None:
            structure_chars.append("-")
            label_chars.append("-")
            continue

        dssp = dssp_map.get((chain_id, residue.pdb_resnum))
        secondary_structure = None if dssp is None else dssp.secondary_structure
        if secondary_structure is None:
            secondary_structure = residue.secondary_structure
            if secondary_structure is None:
                structure_chars.append("-")
                label_chars.append("-")
                continue
            fallback_count += 1

        source_residue = residue.uniprot_residue or sequence_residue
        if source_residue.upper() != sequence_residue:
            structure_chars.append("-")
            label_chars.append("-")
            continue

        structure_chars.append(sequence_residue)
        label_chars.append(secondary_structure)
        observed_count += 1

    return "".join(structure_chars), "".join(label_chars), observed_count, fallback_count


def build_structured_annotation_dataset(
    *,
    seed_path: str | Path,
    pfam_mapping_path: str | Path,
    uniprot_mapping_path: str | Path,
    sequence_fasta_path: str | Path,
    annotation_csv_path: str | Path,
    sifts_cache_dir: str | Path,
    dssp_cache_dir: str | Path,
    summary_path: str | Path,
    selected_families: tuple[str, ...] = DEFAULT_SELECTED_FAMILIES,
    min_observed_fraction: float = 0.6,
    min_observed_residues: int = 40,
) -> AnnotationBuildResult:
    pfam_mappings = load_pfam_chain_mappings(pfam_mapping_path)
    uniprot_mappings = load_uniprot_chain_mappings(uniprot_mapping_path)
    family_accessions = build_family_accession_index(pfam_mappings, set(selected_families))
    seed_records = load_seed_records(
        seed_path,
        families=set(selected_families),
        accessions_by_family=family_accessions,
    )

    aligner = ResidueAligner()
    selected_rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = {
        "no_structure_mapping": 0,
        "missing_sifts_xml": 0,
        "low_observed_fraction": 0,
        "alignment_failure": 0,
        "missing_dssp_file": 0,
    }
    family_counts = {family: 0 for family in selected_families}

    sifts_dir = ensure_dir(sifts_cache_dir)
    dssp_dir = ensure_dir(dssp_cache_dir)

    for record in seed_records:
        structure_mapping = select_best_structure_mapping(record, pfam_mappings, uniprot_mappings)
        if structure_mapping is None:
            skipped["no_structure_mapping"] += 1
            continue

        sifts_path = sifts_dir / f"{structure_mapping.pdb_id}.xml.gz"
        dssp_path = dssp_dir / f"{structure_mapping.pdb_id}.dssp"
        try:
            _download_if_missing(sifts_xml_url(structure_mapping.pdb_id), sifts_path)
            residue_map = load_sifts_residue_mappings(
                sifts_path,
                accession=record.accession,
                chain_id=structure_mapping.chain_id,
            )
        except (HTTPError, URLError, OSError, ValueError):
            skipped["missing_sifts_xml"] += 1
            continue

        try:
            _download_if_missing(dssp_legacy_url(structure_mapping.pdb_id), dssp_path)
            dssp_map = load_legacy_dssp(dssp_path)
        except (HTTPError, URLError, OSError, ValueError):
            dssp_map = {}
            skipped["missing_dssp_file"] += 1
        structure_sequence, labels, observed_count, fallback_count = _build_annotation_strings(
            record=record,
            residue_map=residue_map,
            dssp_map=dssp_map,
            chain_id=structure_mapping.chain_id,
        )

        if observed_count < min_observed_residues or observed_count / max(record.length, 1) < min_observed_fraction:
            skipped["low_observed_fraction"] += 1
            continue

        try:
            aligner.align_labels(record.sequence, structure_sequence, labels)
        except ValueError:
            skipped["alignment_failure"] += 1
            continue

        selected_rows.append(
            {
                "protein_id": record.protein_id,
                "family": record.family,
                "accession": record.accession,
                "family_name": record.family_name,
                "pdb_id": structure_mapping.pdb_id,
                "chain_id": structure_mapping.chain_id,
                "seed_start": record.start,
                "seed_end": record.end,
                "structure_sequence": structure_sequence,
                "labels": labels,
                "observed_fraction": observed_count / record.length,
                "observed_residues": observed_count,
                "fallback_secondary_structure": fallback_count,
            }
        )
        family_counts[record.family] += 1

    fasta_path = Path(sequence_fasta_path)
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    with fasta_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected_rows:
            sequence = next(record.sequence for record in seed_records if record.protein_id == row["protein_id"])
            handle.write(
                f">{row['protein_id']} family={row['family']} family_name={row['family_name']} "
                f"accession={row['accession']} pdb={row['pdb_id']} chain={row['chain_id']} "
                f"start={row['seed_start']} end={row['seed_end']}\n"
            )
            for index in range(0, len(sequence), 80):
                handle.write(sequence[index:index + 80] + "\n")

    annotation_path = Path(annotation_csv_path)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "protein_id",
        "family",
        "accession",
        "family_name",
        "pdb_id",
        "chain_id",
        "seed_start",
        "seed_end",
        "structure_sequence",
        "labels",
        "observed_fraction",
        "observed_residues",
        "fallback_secondary_structure",
    ]
    with annotation_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)

    summary_payload = {
        "selected_families": list(selected_families),
        "num_seed_records": len(seed_records),
        "num_annotated_records": len(selected_rows),
        "family_counts": family_counts,
        "skipped": skipped,
    }
    write_json(summary_path, summary_payload)

    return AnnotationBuildResult(
        fasta_path=fasta_path,
        annotation_csv_path=annotation_path,
        summary_path=Path(summary_path),
        num_selected_records=len(seed_records),
        num_annotated_records=len(selected_rows),
        selected_families=selected_families,
        family_counts=family_counts,
    )
