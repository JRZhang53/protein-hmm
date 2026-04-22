from __future__ import annotations

from collections import Counter

from _bootstrap import bootstrap

ROOT = bootstrap()

from protein_hmm.config import load_project_config
from protein_hmm.data.loaders import load_fasta_records, load_split_records
from protein_hmm.utils.io import write_json
from protein_hmm.utils.paths import resolve_project_path


def main() -> None:
    config = load_project_config(ROOT)
    sequence_path = resolve_project_path(config.data["raw"]["sequence_fasta"], ROOT)
    if sequence_path.exists():
        records = load_fasta_records(sequence_path)
    else:
        records = [
            record
            for split in load_split_records(resolve_project_path(config.data["processed_dir"], ROOT)).values()
            for record in split
        ]

    family_counts = dict(sorted(Counter(record.family for record in records).items()))
    output_path = resolve_project_path(config.experiments["outputs"]["metrics_dir"], ROOT) / "family_scouting.json"
    write_json(output_path, {"num_proteins": len(records), "family_counts": family_counts})
    print(f"Wrote family scouting summary to {output_path}")


if __name__ == "__main__":
    main()
