from __future__ import annotations

from _bootstrap import bootstrap

ROOT = bootstrap()

from protein_hmm.analysis.evaluation import training_diagnostics
from protein_hmm.config import load_project_config
from protein_hmm.data.encoding import AminoAcidEncoder
from protein_hmm.data.loaders import load_split_records
from protein_hmm.models.discrete_hmm import DiscreteHMM
from protein_hmm.utils.io import write_json
from protein_hmm.utils.paths import resolve_project_path


def encode_records(records, encoder):
    return [encoder.encode(record.sequence) for record in records]


def main() -> None:
    config = load_project_config(ROOT)
    splits = load_split_records(resolve_project_path(config.data["processed_dir"], ROOT))
    encoder = AminoAcidEncoder()
    train_sequences = encode_records(splits["train"], encoder)
    val_sequences = encode_records(splits["val"], encoder)
    test_sequences = encode_records(splits["test"], encoder)

    model = DiscreteHMM(**config.models["unsupervised"])
    model.fit(train_sequences)

    model_path = resolve_project_path(config.experiments["outputs"]["model_dir"], ROOT) / "unsupervised_hmm.json"
    model.save(model_path)
    metrics = {
        "train_log_likelihood": model.score_many(train_sequences),
        "val_log_likelihood": model.score_many(val_sequences) if val_sequences else 0.0,
        "test_log_likelihood": model.score_many(test_sequences) if test_sequences else 0.0,
        "train_bic": model.bic(train_sequences),
        "training_history": model.training_history.to_dict(),
        "training_diagnostics": training_diagnostics(
            model.training_history,
            sum(len(sequence) for sequence in train_sequences),
        ),
    }
    metrics_path = resolve_project_path(config.experiments["outputs"]["metrics_dir"], ROOT) / "unsupervised_metrics.json"
    write_json(metrics_path, metrics)
    print(f"Wrote model to {model_path}")
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
