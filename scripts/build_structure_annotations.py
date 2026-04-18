from __future__ import annotations

from _bootstrap import bootstrap

ROOT = bootstrap()

from protein_hmm.config import load_project_config
from protein_hmm.data.structure_annotations import build_structured_annotation_dataset
from protein_hmm.utils.paths import resolve_project_path


def main() -> None:
    config = load_project_config(ROOT)
    selected_families = tuple(config.experiments.get("families", {}).get("selected", ()))
    kwargs = {"selected_families": selected_families} if selected_families else {}
    result = build_structured_annotation_dataset(
        seed_path=resolve_project_path("data/raw/pfam/Pfam-A.seed.gz", ROOT),
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
