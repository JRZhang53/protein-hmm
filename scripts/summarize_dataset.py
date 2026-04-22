from __future__ import annotations

from _bootstrap import bootstrap

ROOT = bootstrap()

from protein_hmm.config import load_project_config
from protein_hmm.data.loaders import load_split_records
from protein_hmm.data.preprocessing import summarize_records
from protein_hmm.utils.io import write_json
from protein_hmm.utils.paths import resolve_project_path


def main() -> None:
    config = load_project_config(ROOT)
    splits = load_split_records(resolve_project_path(config.data["processed_dir"], ROOT))
    summary = {split_name: summarize_records(records) for split_name, records in splits.items()}
    summary["all"] = summarize_records(record for records in splits.values() for record in records)
    output_path = resolve_project_path(config.experiments["outputs"]["metrics_dir"], ROOT) / "dataset_summary.json"
    write_json(output_path, summary)
    print(f"Wrote dataset summary to {output_path}")


if __name__ == "__main__":
    main()
