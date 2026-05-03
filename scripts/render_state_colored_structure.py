"""Render one canonical test protein's 3D backbone, colored by Viterbi-decoded HMM state.

Picks one test protein per family, fetches its mmCIF from RCSB, parses CA
coordinates, decodes its sequence with the saved K=4 unsupervised HMM, and
plots the backbone as a 3D ribbon-like trace colored by latent state.

Output: reports/figures/structure_colored_*.png (one per family chosen).
"""

from __future__ import annotations

import gzip
import io
from pathlib import Path
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
from Bio.PDB import PDBParser

from _bootstrap import bootstrap

ROOT = bootstrap()

from protein_hmm.config import load_project_config
from protein_hmm.data.encoding import AminoAcidEncoder
from protein_hmm.data.loaders import load_split_records
from protein_hmm.data.sifts import load_sifts_residue_mappings
from protein_hmm.models.discrete_hmm import DiscreteHMM
from protein_hmm.utils.paths import by_K_dir, resolve_project_path
from protein_hmm.visualization.style import apply_style


DSSP_COLORS = {"H": "#1f4e79", "E": "#c1432a", "C": "#7f7f7f"}

# Per-K mapping: which state index represents the buried hydrophobic core
# (highlighted orange) and which represents the acidic surface loop
# (highlighted green). Determined by inspecting state_summaries.json after
# training: the core state is the most-buried, most-hydrophobic; the surface
# state is the most-exposed, most-polar.
CORE_STATE_BY_K = {4: 3, 6: 4}
SURFACE_STATE_BY_K = {4: 2, 6: 2}
STATE_NAMES_BY_K = {
    4: {0: "turn-rich coil", 1: "amphipathic helix",
        2: "acidic surface loop", 3: "buried hydrophobic core"},
    6: {0: "polar surface helix", 1: "proline / Gly turn",
        2: "charged surface helix", 3: "Gly-mixed buried region",
        4: "pure aliphatic core", 5: "polar / mixed strand"},
}


def _fetch_pdb(pdb_id: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    with urlopen(url, timeout=60) as response:
        dest.write_bytes(response.read())
    return dest


def _ca_coords_for_chain(pdb_path: Path, chain_id: str) -> dict[str, np.ndarray]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", str(pdb_path))
    coords: dict[str, np.ndarray] = {}
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                if "CA" not in residue:
                    continue
                resnum = str(residue.id[1]) + (residue.id[2].strip() if residue.id[2].strip() else "")
                coords[resnum] = residue["CA"].coord
        break
    return coords


def _aligned_states_and_coords(
    record,
    model: DiscreteHMM,
    encoder: AminoAcidEncoder,
    sifts_dir: Path,
    pdb_dir: Path,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    meta = record.metadata
    pdb_id = meta.get("pdb")
    chain_id = meta.get("chain")
    accession = meta.get("accession")
    try:
        sp_beg = int(meta.get("start"))
        sp_end = int(meta.get("end"))
    except (TypeError, ValueError):
        if verbose:
            print(f"    no_metadata for {record.protein_id}")
        return None
    if not (pdb_id and chain_id and accession):
        if verbose:
            print(f"    missing fields for {record.protein_id}")
        return None

    sifts_path = sifts_dir / f"{pdb_id}.xml.gz"
    if not sifts_path.exists():
        if verbose:
            print(f"    no SIFTS xml at {sifts_path}")
        return None
    pdb_path = pdb_dir / f"{pdb_id}.pdb"
    try:
        _fetch_pdb(pdb_id, pdb_path)
    except Exception as exc:
        if verbose:
            print(f"    PDB fetch failed for {pdb_id}: {exc}")
        return None

    residue_map = load_sifts_residue_mappings(sifts_path, accession=accession, chain_id=chain_id)
    ca_coords = _ca_coords_for_chain(pdb_path, chain_id)

    states = list(model.decode(encoder.encode(record.sequence), protein_id=record.protein_id).states)

    # Walk SIFTS in the same order as the dataset builder, collecting (state, ca_coord)
    # pairs only where we have both.
    from protein_hmm.constants import VALID_AMINO_ACIDS, DSSP_COLLAPSE_MAP

    aligned_states: list[int] = []
    aligned_coords: list[np.ndarray] = []
    seq_index = 0

    for position in range(sp_beg, sp_end + 1):
        residue = residue_map.get(position)
        if residue is None:
            continue
        residue_name = (residue.uniprot_residue or "").upper()
        if not residue_name or residue_name not in VALID_AMINO_ACIDS:
            continue

        ss = None
        if residue.observed and residue.pdb_resnum:
            ss = residue.secondary_structure
        if ss is None:
            continue
        if DSSP_COLLAPSE_MAP.get(ss.upper()) is None:
            continue

        # This residue is in our sequence at position seq_index
        if seq_index >= len(states):
            break
        state = states[seq_index]
        seq_index += 1

        if not residue.observed or residue.pdb_resnum is None:
            continue
        ca = ca_coords.get(residue.pdb_resnum)
        if ca is None:
            continue
        aligned_states.append(state)
        aligned_coords.append(ca)

    if verbose:
        print(f"    {record.protein_id}: kept_states={len(aligned_states)}, seq_len={len(states)}, ca_residues={len(ca_coords)}, sp={sp_beg}-{sp_end}")
    if len(aligned_states) < 30:
        return None
    return np.asarray(aligned_states), np.asarray(aligned_coords)


def render(record, states: np.ndarray, coords: np.ndarray, fig_dir: Path, K: int) -> None:
    """Render the backbone in pale grey with the buried-core and surface
    states highlighted (per-K mapping in CORE_STATE_BY_K / SURFACE_STATE_BY_K).
    """
    apply_style()
    core_state = CORE_STATE_BY_K[K]
    surface_state = SURFACE_STATE_BY_K[K]
    state_names = STATE_NAMES_BY_K[K]
    core_color = "#c1432a"
    surface_color = "#3f7d20"

    fig = plt.figure(figsize=(15, 11))
    ax = fig.add_subplot(111, projection="3d")

    # Pale-grey backbone tube
    for i in range(len(coords) - 1):
        ax.plot(
            [coords[i, 0], coords[i + 1, 0]],
            [coords[i, 1], coords[i + 1, 1]],
            [coords[i, 2], coords[i + 1, 2]],
            color="#bdbdbd", linewidth=2.0, solid_capstyle="round", zorder=1,
        )

    background_mask = (states != core_state) & (states != surface_state)
    if background_mask.any():
        ax.scatter(
            coords[background_mask, 0], coords[background_mask, 1], coords[background_mask, 2],
            c="#bdbdbd", s=10, alpha=0.55, depthshade=False, zorder=2,
            label="other states (helix / coil)",
        )

    core_mask = states == core_state
    if core_mask.any():
        ax.scatter(
            coords[core_mask, 0], coords[core_mask, 1], coords[core_mask, 2],
            c=core_color, s=110, edgecolors="white", linewidths=1.3,
            depthshade=False, alpha=0.96, zorder=4,
            label=f"S{core_state}: {state_names[core_state]} ({int(core_mask.sum())})",
        )

    surface_mask = states == surface_state
    if surface_mask.any():
        ax.scatter(
            coords[surface_mask, 0], coords[surface_mask, 1], coords[surface_mask, 2],
            c=surface_color, s=110, edgecolors="white", linewidths=1.3,
            depthshade=False, alpha=0.92, zorder=3,
            label=f"S{surface_state}: {state_names[surface_state]} ({int(surface_mask.sum())})",
        )

    centroid = coords.mean(axis=0)
    dist = np.linalg.norm(coords - centroid, axis=1)
    core_dist = dist[core_mask].mean() if core_mask.any() else float("nan")
    surface_dist = dist[surface_mask].mean() if surface_mask.any() else float("nan")
    avg = dist.mean()

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_box_aspect((1, 1, 1))
    ax.legend(loc="upper left", fontsize=33, framealpha=0.95, markerscale=2.5,
              handletextpad=0.5, borderpad=0.6)
    family = record.family
    family_name = {"PF00069": "Pkinase", "PF00071": "Ras", "PF00076": "RRM_1",
                   "PF00595": "PDZ", "PF13499": "EF-hand_7", "PF00400": "WD40"}.get(family, family)
    title = f"{family_name} ({record.protein_id})"
    subtitle = (
        f"⟨dist from centroid⟩  S{core_state} (core) {core_dist:.1f} Å   "
        f"<   S{surface_state} (surface) {surface_dist:.1f} Å   (mean {avg:.1f} Å)"
    )
    ax.set_title(title, fontsize=39, fontweight="bold", pad=18)
    ax.text2D(0.5, -0.02, subtitle, transform=ax.transAxes,
              ha="center", va="top", fontsize=27, fontweight="bold", color="#333")
    fig.tight_layout()
    out_path = fig_dir / f"structure_colored_{family}_{record.protein_id.replace('/', '_')}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  rendered {out_path.name}: {len(states)} CA, S{core_state} d={core_dist:.1f}, S{surface_state} d={surface_dist:.1f}, mean {avg:.1f}")


def main() -> None:
    config = load_project_config(ROOT)
    splits = load_split_records(resolve_project_path(config.data["processed_dir"], ROOT))
    test_records = splits["test"]
    if not test_records:
        raise RuntimeError("test split empty")

    K = int(config.models["unsupervised"]["num_states"])
    model = DiscreteHMM.load(by_K_dir(K, "models", ROOT) / "unsupervised_hmm.json")
    encoder = AminoAcidEncoder()

    sifts_dir = resolve_project_path("data/raw/pdb/sifts_xml", ROOT)
    pdb_dir = resolve_project_path("data/raw/pdb/coords", ROOT)
    fig_dir = by_K_dir(K, "figures", ROOT)

    # For each family, try test records in order of length (longest first) until one renders.
    by_family: dict[str, list] = {}
    for record in test_records:
        by_family.setdefault(record.family, []).append(record)

    print(f"Rendering one test protein per family (try up to 6 candidates each):")
    for family in sorted(by_family):
        candidates = sorted(by_family[family], key=lambda r: -r.length)[:6]
        for record in candidates:
            result = _aligned_states_and_coords(record, model, encoder, sifts_dir, pdb_dir, verbose=True)
            if result is None:
                continue
            states, coords = result
            render(record, states, coords, fig_dir, K=K)
            break
        else:
            print(f"  no renderable candidate for {family}")


if __name__ == "__main__":
    main()
