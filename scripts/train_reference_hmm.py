from __future__ import annotations

import numpy as np

from _bootstrap import bootstrap

ROOT = bootstrap()

from protein_hmm.analysis.evaluation import training_diagnostics
from protein_hmm.analysis.metrics import q3_accuracy, segment_overlap_score
from protein_hmm.config import load_project_config
from protein_hmm.data.encoding import AminoAcidEncoder
from protein_hmm.data.loaders import load_split_records
from protein_hmm.models.semi_supervised_hmm import SemiSupervisedHMM
from protein_hmm.utils.io import write_json
from protein_hmm.utils.paths import resolve_project_path


def main() -> None:
    config = load_project_config(ROOT)
    splits = load_split_records(resolve_project_path(config.data["processed_dir"], ROOT))
    train_records = [record for record in splits["train"] if record.labels]
    test_records = [record for record in splits["test"] if record.labels]
    if not train_records:
        raise RuntimeError("No labeled training records are available for the reference HMM.")

    encoder = AminoAcidEncoder()
    train_sequences = [encoder.encode(record.sequence) for record in train_records]
    train_labels = [record.labels or "" for record in train_records]

    model = SemiSupervisedHMM(**config.models["reference"])
    model.fit(train_sequences, train_labels)

    model_path = resolve_project_path(config.experiments["outputs"]["model_dir"], ROOT) / "reference_hmm.json"
    model.save(model_path)

    predicted = [model.predict_labels(encoder.encode(record.sequence)) for record in test_records]
    truth = [record.labels or "" for record in test_records]
    metrics = {
        "train_log_likelihood": model.score_many(train_sequences),
        "test_log_likelihood": model.score_many([encoder.encode(record.sequence) for record in test_records]) if test_records else 0.0,
        "q3": q3_accuracy("".join(truth), "".join(predicted)) if truth else 0.0,
        "sov": float(np.mean([segment_overlap_score(true, pred) for true, pred in zip(truth, predicted)])) if truth else 0.0,
        "training_history": model.training_history.to_dict(),
        "training_diagnostics": training_diagnostics(
            model.training_history,
            sum(len(sequence) for sequence in train_sequences),
        ),
    }
    output_path = resolve_project_path(config.experiments["outputs"]["metrics_dir"], ROOT) / "reference_metrics.json"
    write_json(output_path, metrics)
    print(f"Wrote reference model to {model_path}")
    print(f"Wrote reference metrics to {output_path}")


if __name__ == "__main__":
    main()
