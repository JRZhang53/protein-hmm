from __future__ import annotations

from _bootstrap import bootstrap

ROOT = bootstrap()

from protein_hmm.analysis.metrics import bic_score
from protein_hmm.analysis.evaluation import evaluate_hmm_annotations, training_diagnostics
from protein_hmm.config import load_project_config
from protein_hmm.data.encoding import AminoAcidEncoder
from protein_hmm.data.loaders import load_split_records
from protein_hmm.models.discrete_hmm import DiscreteHMM
from protein_hmm.utils.io import write_json
from protein_hmm.utils.paths import resolve_project_path


def main() -> None:
    config = load_project_config(ROOT)
    splits = load_split_records(resolve_project_path(config.data["processed_dir"], ROOT))
    encoder = AminoAcidEncoder()
    train_records = splits["train"]
    val_records = splits["val"]
    test_records = splits["test"]
    train_sequences = [encoder.encode(record.sequence) for record in train_records]
    val_sequences = [encoder.encode(record.sequence) for record in val_records]
    test_sequences = [encoder.encode(record.sequence) for record in test_records]
    train_labeled = [record for record in train_records if record.labels]
    test_labeled = [record for record in test_records if record.labels]

    results = []
    for num_states in config.models["model_selection"]["candidate_states"]:
        model = DiscreteHMM(num_states=num_states, **{
            key: value
            for key, value in config.models["unsupervised"].items()
            if key != "num_states"
        })
        model.fit(train_sequences)
        train_log_likelihood = model.score_many(train_sequences)
        val_log_likelihood = model.score_many(val_sequences) if val_sequences else 0.0
        test_log_likelihood = model.score_many(test_sequences) if test_sequences else 0.0
        total_train_residues = sum(len(sequence) for sequence in train_sequences) or 1
        row = {
            "num_states": num_states,
            "train_log_likelihood": train_log_likelihood,
            "val_log_likelihood": val_log_likelihood,
            "test_log_likelihood": test_log_likelihood,
            "bic": bic_score(train_log_likelihood, model.parameter_count(), total_train_residues),
            "training_diagnostics": training_diagnostics(model.training_history, total_train_residues),
        }
        if train_labeled and test_labeled:
            row["annotation_evaluation"] = evaluate_hmm_annotations(
                model=model,
                encoder=encoder,
                mapping_records=train_labeled,
                evaluation_records=test_labeled,
            )
        results.append(row)

    output_path = resolve_project_path(config.experiments["outputs"]["metrics_dir"], ROOT) / "model_selection.json"
    write_json(output_path, {"results": results})
    print(f"Wrote model selection results to {output_path}")


if __name__ == "__main__":
    main()
