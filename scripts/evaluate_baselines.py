from __future__ import annotations

from _bootstrap import bootstrap

ROOT = bootstrap()

from protein_hmm.analysis.evaluation import annotation_baselines
from protein_hmm.config import load_project_config
from protein_hmm.data.loaders import load_split_records
from protein_hmm.utils.io import write_json
from protein_hmm.utils.paths import resolve_project_path


def main() -> None:
    config = load_project_config(ROOT)
    splits = load_split_records(resolve_project_path(config.data["processed_dir"], ROOT))
    output_path = resolve_project_path(config.experiments["outputs"]["metrics_dir"], ROOT) / "baselines.json"
    write_json(output_path, annotation_baselines(splits["train"], splits["test"]))
    print(f"Wrote baseline metrics to {output_path}")


if __name__ == "__main__":
    main()
