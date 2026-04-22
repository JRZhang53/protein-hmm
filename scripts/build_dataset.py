from __future__ import annotations

from _bootstrap import bootstrap

ROOT = bootstrap()

from protein_hmm.config import load_project_config
from protein_hmm.data.loaders import attach_annotations, load_annotation_table, load_fasta_records, save_split_records
from protein_hmm.data.preprocessing import DatasetPreprocessor, summarize_records
from protein_hmm.data.splits import ProteinLevelSplitter
from protein_hmm.utils.io import write_json
from protein_hmm.utils.paths import resolve_project_path


def main() -> None:
    config = load_project_config(ROOT)
    sequence_path = resolve_project_path(config.data["raw"]["sequence_fasta"], ROOT)
    annotation_path = resolve_project_path(config.data["raw"]["annotation_csv"], ROOT)
    processed_dir = resolve_project_path(config.data["processed_dir"], ROOT)

    records = load_fasta_records(sequence_path)
    if annotation_path.exists():
        annotations = load_annotation_table(annotation_path)
        records = attach_annotations(records, annotations)

    preprocessor = DatasetPreprocessor(**config.data["filters"])
    cleaned = preprocessor.clean(records)

    split_config = config.data["split"]
    splitter = ProteinLevelSplitter(
        train_fraction=split_config["train_fraction"],
        val_fraction=split_config["val_fraction"],
        test_fraction=split_config["test_fraction"],
        random_state=split_config["seed"],
    )
    splits = splitter.split(cleaned)
    save_split_records(processed_dir, splits)

    summary = {split_name: summarize_records(records) for split_name, records in splits.items()}
    summary["all"] = summarize_records(cleaned)
    summary_path = resolve_project_path(config.experiments["outputs"]["metrics_dir"], ROOT) / "dataset_summary.json"
    write_json(summary_path, summary)
    print(f"Wrote processed splits to {processed_dir}")
    print(f"Wrote dataset summary to {summary_path}")


if __name__ == "__main__":
    main()
