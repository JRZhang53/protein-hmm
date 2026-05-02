"""Per-residue solvent accessibility (RSA) analysis aligned to HMM states.

RSA = ACC / max-ASA[residue type], where ACC comes from the DSSP file's
accessibility column and max-ASA is the empirical maximum from
Tien et al. 2013 (PLOS ONE).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np

from protein_hmm.constants import VALID_AMINO_ACIDS, DSSP_COLLAPSE_MAP
from protein_hmm.data.dssp import dssp_legacy_url, load_legacy_dssp
from protein_hmm.data.sifts import load_sifts_residue_mappings


# Tien et al. 2013 empirical maxima (Å²); used to normalise DSSP ACC -> RSA.
MAX_ASA: dict[str, float] = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0,
    "E": 223.0, "Q": 225.0, "G": 104.0, "H": 224.0, "I": 197.0,
    "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0, "P": 159.0,
    "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
}


def _ensure_dssp(pdb_id: str, dssp_dir: Path) -> Path | None:
    """Download the legacy DSSP file for ``pdb_id`` into ``dssp_dir`` if absent."""
    target = dssp_dir / f"{pdb_id}.dssp"
    if target.exists() and target.stat().st_size > 0:
        return target
    try:
        with urlopen(dssp_legacy_url(pdb_id), timeout=60) as response:
            target.write_bytes(response.read())
        return target
    except (HTTPError, URLError, OSError):
        return None


def _walk_sifts_with_optional_dssp(
    residue_map: dict[int, Any],
    dssp_map: dict[tuple[str, str], Any],
    chain_id: str,
    sp_beg: int,
    sp_end: int,
    use_dssp_for_ss: bool,
) -> tuple[list[str], list[float | None]]:
    """Return (kept_residues, rsa_values) for one walk.

    ``use_dssp_for_ss`` mirrors the dataset builder's two modes: SIFTS-only
    (DSSP labels ignored) vs. DSSP-fallback (DSSP labels override SIFTS).
    The original build defaulted to SIFTS-only and only re-tried with DSSP
    when too few residues were observed.
    """
    kept: list[str] = []
    rsa: list[float | None] = []
    for position in range(sp_beg, sp_end + 1):
        residue = residue_map.get(position)
        if residue is None:
            continue
        residue_name = (residue.uniprot_residue or "").upper()
        if not residue_name or residue_name not in VALID_AMINO_ACIDS:
            continue

        dssp_entry = None
        if residue.observed and residue.pdb_resnum:
            dssp_entry = dssp_map.get((chain_id, residue.pdb_resnum))

        secondary_structure: str | None = None
        if use_dssp_for_ss and dssp_entry is not None:
            secondary_structure = dssp_entry.secondary_structure
        if secondary_structure is None:
            secondary_structure = residue.secondary_structure
        if secondary_structure is None:
            continue
        if DSSP_COLLAPSE_MAP.get(secondary_structure.upper()) is None:
            continue

        rsa_value: float | None = None
        if dssp_entry is not None and dssp_entry.accessibility is not None:
            max_asa = MAX_ASA.get(residue_name)
            if max_asa:
                rsa_value = min(dssp_entry.accessibility / max_asa, 1.0)

        kept.append(residue_name)
        rsa.append(rsa_value)
    return kept, rsa


def per_residue_rsa(
    *,
    sifts_xml_path: str | Path,
    dssp_path: str | Path,
    accession: str,
    chain_id: str,
    sp_beg: int,
    sp_end: int,
    expected_sequence: str,
) -> list[float | None] | None:
    """Reconstruct per-residue RSA aligned to ``expected_sequence``.

    The dataset builder ran one of two walks (SIFTS-only or DSSP-fallback)
    depending on observed-residue counts. We try both here and return the
    walk whose kept-residue string equals ``expected_sequence`` -- this
    guarantees the returned RSA list lines up with the saved record's
    sequence positions. Returns ``None`` if neither walk matches.
    """
    residue_map = load_sifts_residue_mappings(
        sifts_xml_path, accession=accession, chain_id=chain_id
    )
    dssp_map = load_legacy_dssp(dssp_path)

    for use_dssp_for_ss in (False, True):
        kept, rsa = _walk_sifts_with_optional_dssp(
            residue_map=residue_map,
            dssp_map=dssp_map,
            chain_id=chain_id,
            sp_beg=sp_beg,
            sp_end=sp_end,
            use_dssp_for_ss=use_dssp_for_ss,
        )
        if "".join(kept) == expected_sequence:
            return rsa
    return None


def aggregate_state_rsa(
    state_paths: list[list[int]],
    rsa_per_record: list[list[float | None]],
    num_states: int,
) -> dict[str, Any]:
    """Aggregate per-residue RSA samples by latent state."""
    if len(state_paths) != len(rsa_per_record):
        raise ValueError("state_paths and rsa_per_record must align.")
    samples: dict[int, list[float]] = defaultdict(list)
    for path, rsa_values in zip(state_paths, rsa_per_record):
        if len(path) != len(rsa_values):
            continue
        for state, value in zip(path, rsa_values):
            if value is None:
                continue
            samples[int(state)].append(float(value))

    summary = {
        "mean": [float(np.mean(samples[k])) if samples.get(k) else float("nan") for k in range(num_states)],
        "median": [float(np.median(samples[k])) if samples.get(k) else float("nan") for k in range(num_states)],
        "std": [float(np.std(samples[k])) if samples.get(k) else float("nan") for k in range(num_states)],
        "n": [len(samples.get(k, [])) for k in range(num_states)],
    }
    return summary
