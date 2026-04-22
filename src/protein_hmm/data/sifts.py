"""SIFTS chain mapping and residue-level XML utilities."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path
from xml.etree import ElementTree as ET

from protein_hmm.data.pfam_seed import PfamSeedRecord


SIFTS_NS = {"s": "http://www.ebi.ac.uk/pdbe/docs/sifts/eFamily.xsd"}


@dataclass(frozen=True, slots=True)
class PfamChainMapping:
    pdb_id: str
    chain_id: str
    accession: str
    family: str
    coverage: float


@dataclass(frozen=True, slots=True)
class UniProtChainMapping:
    pdb_id: str
    chain_id: str
    accession: str
    sp_beg: int
    sp_end: int
    pdb_beg: str | None
    pdb_end: str | None


@dataclass(frozen=True, slots=True)
class SelectedStructureMapping:
    pdb_id: str
    chain_id: str
    accession: str
    family: str
    coverage: float
    sp_beg: int
    sp_end: int
    overlap: int
    covers_domain: bool


@dataclass(frozen=True, slots=True)
class SiftsResidueMapping:
    accession: str
    chain_id: str
    uniprot_position: int
    uniprot_residue: str
    pdb_resnum: str | None
    pdb_residue: str | None
    observed: bool
    secondary_structure: str | None = None


def load_pfam_chain_mappings(path: str | Path) -> list[PfamChainMapping]:
    mappings: list[PfamChainMapping] = []
    with gzip.open(Path(path), "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip() or line.startswith("PDB\t"):
                continue
            pdb_id, chain_id, accession, family, coverage = line.rstrip("\n").split("\t")
            mappings.append(
                PfamChainMapping(
                    pdb_id=pdb_id.lower(),
                    chain_id=chain_id,
                    accession=accession,
                    family=family,
                    coverage=float(coverage),
                )
            )
    return mappings


def load_uniprot_chain_mappings(path: str | Path) -> list[UniProtChainMapping]:
    mappings: list[UniProtChainMapping] = []
    with gzip.open(Path(path), "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip() or line.startswith("PDB\t"):
                continue
            pdb_id, chain_id, accession, _, _, pdb_beg, pdb_end, sp_beg, sp_end = line.rstrip("\n").split("\t")
            mappings.append(
                UniProtChainMapping(
                    pdb_id=pdb_id.lower(),
                    chain_id=chain_id,
                    accession=accession,
                    sp_beg=int(sp_beg),
                    sp_end=int(sp_end),
                    pdb_beg=pdb_beg or None,
                    pdb_end=pdb_end or None,
                )
            )
    return mappings


def build_family_accession_index(mappings: list[PfamChainMapping], families: set[str]) -> dict[str, set[str]]:
    accessions: dict[str, set[str]] = {family: set() for family in families}
    for mapping in mappings:
        if mapping.family in families:
            accessions.setdefault(mapping.family, set()).add(mapping.accession)
    return accessions


def select_best_structure_mapping(
    record: PfamSeedRecord,
    pfam_mappings: list[PfamChainMapping],
    uniprot_mappings: list[UniProtChainMapping],
) -> SelectedStructureMapping | None:
    candidates: list[SelectedStructureMapping] = []
    chain_spans = {
        (mapping.pdb_id, mapping.chain_id, mapping.accession): mapping for mapping in uniprot_mappings
    }
    for mapping in pfam_mappings:
        if mapping.family != record.family or mapping.accession != record.accession:
            continue
        span = chain_spans.get((mapping.pdb_id, mapping.chain_id, mapping.accession))
        if span is None:
            continue
        overlap = max(0, min(record.end, span.sp_end) - max(record.start, span.sp_beg) + 1)
        if overlap == 0:
            continue
        candidates.append(
            SelectedStructureMapping(
                pdb_id=mapping.pdb_id,
                chain_id=mapping.chain_id,
                accession=mapping.accession,
                family=mapping.family,
                coverage=mapping.coverage,
                sp_beg=span.sp_beg,
                sp_end=span.sp_end,
                overlap=overlap,
                covers_domain=span.sp_beg <= record.start and span.sp_end >= record.end,
            )
        )

    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            item.covers_domain,
            item.overlap,
            item.coverage,
            item.sp_end - item.sp_beg,
            item.pdb_id,
            item.chain_id,
        ),
        reverse=True,
    )
    return candidates[0]


def load_sifts_residue_mappings(
    path: str | Path,
    accession: str,
    chain_id: str,
) -> dict[int, SiftsResidueMapping]:
    root = ET.parse(gzip.open(Path(path), "rb")).getroot()
    mappings: dict[int, SiftsResidueMapping] = {}
    for residue in root.findall(".//s:residue", SIFTS_NS):
        pdb_crossref: dict[str, str] | None = None
        uniprot_crossref: dict[str, str] | None = None
        annotation_details: dict[str, str] = {}
        for child in residue:
            tag = child.tag.split("}", 1)[-1]
            if tag == "crossRefDb":
                if child.attrib.get("dbSource") == "PDB" and child.attrib.get("dbChainId") == chain_id:
                    pdb_crossref = dict(child.attrib)
                elif (
                    child.attrib.get("dbSource") == "UniProt"
                    and child.attrib.get("dbAccessionId") == accession
                ):
                    uniprot_crossref = dict(child.attrib)
            elif tag == "residueDetail":
                annotation_details[child.attrib.get("property", "")] = (child.text or "").strip()

        if uniprot_crossref is None:
            continue
        uniprot_resnum = uniprot_crossref.get("dbResNum")
        if uniprot_resnum is None or not uniprot_resnum.isdigit():
            continue

        pdb_resnum = None if pdb_crossref is None else pdb_crossref.get("dbResNum")
        observed = (
            annotation_details.get("Annotation") != "Not_Observed"
            and pdb_resnum not in {None, "null"}
        )
        mappings[int(uniprot_resnum)] = SiftsResidueMapping(
            accession=accession,
            chain_id=chain_id,
            uniprot_position=int(uniprot_resnum),
            uniprot_residue=uniprot_crossref.get("dbResName", "").upper(),
            pdb_resnum=None if pdb_resnum in {None, "null"} else pdb_resnum,
            pdb_residue=None if pdb_crossref is None else pdb_crossref.get("dbResName"),
            observed=observed,
            secondary_structure=annotation_details.get("codeSecondaryStructure"),
        )
    return mappings


def sifts_xml_url(pdb_id: str) -> str:
    return f"https://ftp.ebi.ac.uk/pub/databases/msd/sifts/xml/{pdb_id.lower()}.xml.gz"
