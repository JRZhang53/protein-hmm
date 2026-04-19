from __future__ import annotations

from _bootstrap import bootstrap

ROOT = bootstrap()

from protein_hmm.analysis.evaluation import evaluate_hmm_annotations
from protein_hmm.config import load_project_config
from protein_hmm.data.encoding import AminoAcidEncoder
from protein_hmm.data.loaders import load_split_records
from protein_hmm.models.discrete_hmm import DiscreteHMM
from protein_hmm.utils.io import write_json
from protein_hmm.utils.paths import resolve_project_path


def main() -> None:
    config = load_project_config(ROOT)
    splits = load_split_records(resolve_project_path(config.data["processed_dir"], ROOT))
    train_labeled_records = [record for record in splits["train"] if record.labels]
    labeled_records = [record for record in splits["test"] if record.labels]
    if not train_labeled_records:
        raise RuntimeError("No labeled training records were found for state-label mapping.")
    if not labeled_records:
        raise RuntimeError("No labeled test records were found for annotation evaluation.")

    model = DiscreteHMM.load(resolve_project_path(config.experiments["outputs"]["model_dir"], ROOT) / "unsupervised_hmm.json")
    encoder = AminoAcidEncoder()

    metrics = evaluate_hmm_annotations(
        model=model,
        encoder=encoder,
        mapping_records=train_labeled_records,
        evaluation_records=labeled_records,
    )
    output_path = resolve_project_path(config.experiments["outputs"]["metrics_dir"], ROOT) / "annotation_evaluation.json"
    write_json(output_path, metrics)
    print(f"Wrote annotation evaluation to {output_path}")


if __name__ == "__main__":
    main()
