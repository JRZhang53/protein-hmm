from __future__ import annotations

from _bootstrap import bootstrap

ROOT = bootstrap()

from protein_hmm.config import load_project_config
from protein_hmm.data.structure_annotations import build_dataset_from_sifts_chains
from protein_hmm.utils.paths import resolve_project_path


def main() -> None:
    config = load_project_config(ROOT)
    family_config = config.experiments.get("families", {})
    selected_families = tuple(family_config.get("selected", ()))
    kwargs: dict = {}
    if selected_families:
        kwargs["selected_families"] = selected_families
    if "min_observed_fraction" in family_config:
        kwargs["min_observed_fraction"] = float(family_config["min_observed_fraction"])
    if "min_observed_residues" in family_config:
        kwargs["min_observed_residues"] = int(family_config["min_observed_residues"])
    if "max_per_family" in family_config:
        kwargs["max_per_family"] = int(family_config["max_per_family"])
    result = build_dataset_from_sifts_chains(
        pfam_mapping_path=resolve_project_path("data/raw/pdb/pdb_chain_pfam.tsv.gz", ROOT),
        uniprot_mapping_path=resolve_project_path("data/raw/pdb/pdb_chain_uniprot.tsv.gz", ROOT),
        sequence_fasta_path=resolve_project_path(config.data["raw"]["sequence_fasta"], ROOT),
        annotation_csv_path=resolve_project_path(config.data["raw"]["annotation_csv"], ROOT),
        sifts_cache_dir=resolve_project_path("data/raw/pdb/sifts_xml", ROOT),
        dssp_cache_dir=resolve_project_path("data/raw/dssp/files", ROOT),
        summary_path=resolve_project_path("results/metrics/structure_annotation_summary.json", ROOT),
        **kwargs,
    )
    print(f"Wrote structured sequence FASTA to {result.fasta_path}")
    print(f"Wrote annotation CSV to {result.annotation_csv_path}")
    print(f"Wrote summary to {result.summary_path}")
    print(f"Annotated {result.num_annotated_records} of {result.num_selected_records} candidate records")
    for family, count in result.family_counts.items():
        print(f"{family}\t{count}")


if __name__ == "__main__":
    main()
