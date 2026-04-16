"""Legacy DSSP parsing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DsspResidue:
    chain_id: str
    pdb_resnum: str
    residue: str
    secondary_structure: str


def parse_dssp_line(line: str) -> DsspResidue | None:
    if len(line) < 17:
        return None
    residue_index = line[:5].strip()
    if not residue_index or not residue_index.lstrip("-").isdigit():
        return None

    pdb_resnum = line[5:11].strip()
    chain_id = line[11].strip()
    residue = line[13].strip()
    if not pdb_resnum or not chain_id or residue in {"", "!", "*"}:
        return None

    secondary_structure = line[16].strip() or "C"
    return DsspResidue(
        chain_id=chain_id,
        pdb_resnum=pdb_resnum,
        residue=residue.upper(),
        secondary_structure=secondary_structure,
    )


def load_legacy_dssp(path: str | Path) -> dict[tuple[str, str], DsspResidue]:
    residue_map: dict[tuple[str, str], DsspResidue] = {}
    in_residue_table = False
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("  #  RESIDUE AA STRUCTURE"):
            in_residue_table = True
            continue
        if not in_residue_table:
            continue
        residue = parse_dssp_line(line)
        if residue is None:
            continue
        residue_map[(residue.chain_id, residue.pdb_resnum)] = residue
    return residue_map


def dssp_legacy_url(pdb_id: str) -> str:
    return f"https://pdb-redo.eu/dssp/db/{pdb_id.lower()}/legacy"
