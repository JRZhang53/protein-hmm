"""Build structure-linked Pfam subsets and DSSP annotation tables."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from protein_hmm.constants import DSSP_COLLAPSE_MAP, VALID_AMINO_ACIDS
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


FAMILY_NAMES: dict[str, str] = {
    "PF00069": "Pkinase",
    "PF00071": "Ras",
    "PF00076": "RRM_1",
    "PF00595": "PDZ",
    "PF13499": "EF-hand_7",
    "PF00400": "WD40",
    "PF00018": "SH3_1",
    "PF07679": "I-set",
    "PF00046": "Homeobox",
    "PF00270": "DEAD",
    "PF00210": "Ferritin",
    "PF02826": "2-Hacid_dh_C",
    "PF07715": "Plug",
    "PF01590": "GAF",
}


DEFAULT_SELECTED_FAMILIES: tuple[str, ...] = (
    "PF00069",  # Pkinase, mixed alpha/beta, very large in PDB
    "PF00071",  # Ras, small GTPase, alpha/beta sandwich
    "PF00076",  # RRM_1, beta-rich RNA recognition motif
    "PF00595",  # PDZ, beta-sheet adaptor
    "PF13499",  # EF-hand_7, helix-rich Ca-binding
    "PF00400",  # WD40, beta-propeller blade
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
    min_observed_fraction: float = 0.4,
    min_observed_residues: int = 25,
    max_per_family: int | None = None,
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

    total_seed = len(seed_records)
    print(f"[annotate] {total_seed} seed candidates across {len(selected_families)} families", flush=True)
    for index, record in enumerate(seed_records):
        if index and index % 25 == 0:
            print(
                f"[annotate] processed {index}/{total_seed} | accepted {len(selected_rows)} | "
                f"per-family {family_counts}",
                flush=True,
            )
        if max_per_family is not None and family_counts.get(record.family, 0) >= max_per_family:
            continue
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


def _build_sequence_and_labels_from_sifts(
    *,
    residue_map: dict[int, Any],
    sp_beg: int,
    sp_end: int,
    chain_id: str,
    dssp_map: dict[tuple[str, str], Any] | None = None,
) -> tuple[str, str, int]:
    """Walk SIFTS residues over [sp_beg, sp_end] and return (sequence, labels, observed_count)."""
    dssp_map = dssp_map or {}
    sequence_chars: list[str] = []
    label_chars: list[str] = []
    for position in range(sp_beg, sp_end + 1):
        residue = residue_map.get(position)
        if residue is None:
            continue
        residue_name = (residue.uniprot_residue or "").upper()
        if not residue_name or residue_name not in VALID_AMINO_ACIDS:
            continue
        secondary_structure: str | None = None
        if residue.observed and residue.pdb_resnum:
            dssp_entry = dssp_map.get((chain_id, residue.pdb_resnum))
            if dssp_entry is not None:
                secondary_structure = dssp_entry.secondary_structure
        if secondary_structure is None:
            secondary_structure = residue.secondary_structure
        if secondary_structure is None:
            continue
        collapsed = DSSP_COLLAPSE_MAP.get(secondary_structure.upper())
        if collapsed is None:
            continue
        sequence_chars.append(residue_name)
        label_chars.append(collapsed)
    sequence = "".join(sequence_chars)
    labels = "".join(label_chars)
    return sequence, labels, len(sequence)


def build_dataset_from_sifts_chains(
    *,
    pfam_mapping_path: str | Path,
    uniprot_mapping_path: str | Path,
    sequence_fasta_path: str | Path,
    annotation_csv_path: str | Path,
    sifts_cache_dir: str | Path,
    dssp_cache_dir: str | Path,
    summary_path: str | Path,
    selected_families: tuple[str, ...] = DEFAULT_SELECTED_FAMILIES,
    min_observed_fraction: float = 0.4,
    min_observed_residues: int = 25,
    max_per_family: int | None = 200,
    use_pdb_redo_dssp_fallback: bool = True,
) -> AnnotationBuildResult:
    """Build the structured-annotation dataset from SIFTS pdb_chain_pfam directly.

    For each Pfam family in ``selected_families``, every distinct UniProt
    accession with a PDB classification under that family contributes one
    record. The sequence is reconstructed from SIFTS UniProt residues over the
    PDB-mapped span; secondary-structure labels come from SIFTS first and fall
    back to a pdb-redo legacy DSSP file when SIFTS leaves them blank.
    """
    pfam_mappings = load_pfam_chain_mappings(pfam_mapping_path)
    uniprot_mappings = load_uniprot_chain_mappings(uniprot_mapping_path)

    uniprot_index: dict[tuple[str, str, str], Any] = {
        (mapping.pdb_id, mapping.chain_id, mapping.accession): mapping
        for mapping in uniprot_mappings
    }

    selected_set = set(selected_families)
    family_to_mappings: dict[str, list[Any]] = defaultdict(list)
    for mapping in pfam_mappings:
        if mapping.family in selected_set:
            family_to_mappings[mapping.family].append(mapping)
    for family in family_to_mappings:
        family_to_mappings[family].sort(key=lambda mapping: -mapping.coverage)

    sifts_dir = ensure_dir(sifts_cache_dir)
    dssp_dir = ensure_dir(dssp_cache_dir)

    selected_rows: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {family: 0 for family in selected_families}
    skipped: Counter[str] = Counter()

    for family in selected_families:
        mappings = family_to_mappings.get(family, [])
        seen_accessions: set[str] = set()
        print(f"[annotate] family {family}: {len(mappings)} chain mappings", flush=True)
        for mapping_index, mapping in enumerate(mappings):
            if max_per_family is not None and family_counts[family] >= max_per_family:
                break
            if mapping.accession in seen_accessions:
                continue

            span = uniprot_index.get((mapping.pdb_id, mapping.chain_id, mapping.accession))
            if span is None:
                skipped["no_uniprot_span"] += 1
                continue

            sifts_path = sifts_dir / f"{mapping.pdb_id}.xml.gz"
            try:
                _download_if_missing(sifts_xml_url(mapping.pdb_id), sifts_path)
                residue_map = load_sifts_residue_mappings(
                    sifts_path,
                    accession=mapping.accession,
                    chain_id=mapping.chain_id,
                )
            except (HTTPError, URLError, OSError, ValueError):
                skipped["missing_sifts_xml"] += 1
                continue

            sequence, labels, observed = _build_sequence_and_labels_from_sifts(
                residue_map=residue_map,
                sp_beg=span.sp_beg,
                sp_end=span.sp_end,
                chain_id=mapping.chain_id,
            )

            domain_length = max(span.sp_end - span.sp_beg + 1, 1)
            if (
                use_pdb_redo_dssp_fallback
                and (observed < min_observed_residues or observed / domain_length < min_observed_fraction)
            ):
                dssp_path = dssp_dir / f"{mapping.pdb_id}.dssp"
                try:
                    _download_if_missing(dssp_legacy_url(mapping.pdb_id), dssp_path)
                    dssp_map = load_legacy_dssp(dssp_path)
                    sequence, labels, observed = _build_sequence_and_labels_from_sifts(
                        residue_map=residue_map,
                        sp_beg=span.sp_beg,
                        sp_end=span.sp_end,
                        chain_id=mapping.chain_id,
                        dssp_map=dssp_map,
                    )
                except (HTTPError, URLError, OSError, ValueError):
                    pass

            if observed < min_observed_residues or observed / domain_length < min_observed_fraction:
                skipped["low_observed_fraction"] += 1
                continue

            seen_accessions.add(mapping.accession)
            family_counts[family] += 1
            protein_id = f"{mapping.accession}/{span.sp_beg}-{span.sp_end}"
            selected_rows.append(
                {
                    "protein_id": protein_id,
                    "family": family,
                    "accession": mapping.accession,
                    "family_name": FAMILY_NAMES.get(family, family),
                    "pdb_id": mapping.pdb_id,
                    "chain_id": mapping.chain_id,
                    "seed_start": span.sp_beg,
                    "seed_end": span.sp_end,
                    "sequence": sequence,
                    "structure_sequence": sequence,
                    "labels": labels,
                    "observed_fraction": observed / domain_length,
                    "observed_residues": observed,
                    "fallback_secondary_structure": 0,
                }
            )
            if family_counts[family] % 25 == 0:
                print(
                    f"[annotate] {family}: accepted {family_counts[family]}",
                    flush=True,
                )

    fasta_path = Path(sequence_fasta_path)
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    with fasta_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected_rows:
            handle.write(
                f">{row['protein_id']} family={row['family']} family_name={row['family_name']} "
                f"accession={row['accession']} pdb={row['pdb_id']} chain={row['chain_id']} "
                f"start={row['seed_start']} end={row['seed_end']}\n"
            )
            sequence = row["sequence"]
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected_rows)

    summary_payload = {
        "source": "sifts_pdb_chain_pfam",
        "selected_families": list(selected_families),
        "num_seed_records": sum(len(family_to_mappings.get(family, [])) for family in selected_families),
        "num_annotated_records": len(selected_rows),
        "family_counts": family_counts,
        "skipped": dict(skipped),
        "min_observed_fraction": min_observed_fraction,
        "min_observed_residues": min_observed_residues,
        "max_per_family": max_per_family,
    }
    write_json(summary_path, summary_payload)

    return AnnotationBuildResult(
        fasta_path=fasta_path,
        annotation_csv_path=annotation_path,
        summary_path=Path(summary_path),
        num_selected_records=summary_payload["num_seed_records"],
        num_annotated_records=len(selected_rows),
        selected_families=selected_families,
        family_counts=family_counts,
    )
