from __future__ import annotations

from _bootstrap import bootstrap

ROOT = bootstrap()

from protein_hmm.analysis.family_comparison import (
    cross_family_likelihood_matrix,
    stationary_distance_matrix,
    transition_distance_matrix,
)
from protein_hmm.config import load_project_config
from protein_hmm.data.encoding import AminoAcidEncoder
from protein_hmm.data.loaders import load_split_records
from protein_hmm.data.preprocessing import group_by_family
from protein_hmm.models.discrete_hmm import DiscreteHMM
from protein_hmm.utils.io import write_json
from protein_hmm.utils.paths import resolve_project_path


def main() -> None:
    config = load_project_config(ROOT)
    splits = load_split_records(resolve_project_path(config.data["processed_dir"], ROOT))
    encoder = AminoAcidEncoder()
    train_records_by_family = group_by_family(splits["train"])
    test_records_by_family = group_by_family(splits["test"])

    models = {}
    for family, records in train_records_by_family.items():
        sequences = [encoder.encode(record.sequence) for record in records]
        if not sequences:
            continue
        model = DiscreteHMM(**config.models["unsupervised"])
        model.fit(sequences)
        model_path = resolve_project_path(config.experiments["outputs"]["model_dir"], ROOT) / f"family_{family}.json"
        model.save(model_path)
        models[family] = model

    encoded_test = {
        family: [encoder.encode(record.sequence) for record in records]
        for family, records in test_records_by_family.items()
    }
    ordered_families, transition_distances_aligned = transition_distance_matrix(models, align_states=True)
    _, transition_distances_raw = transition_distance_matrix(models, align_states=False)
    _, stationary_distances = stationary_distance_matrix(models)
    model_families, test_families, cross_family = cross_family_likelihood_matrix(models, encoded_test)

    metrics = {
        "families": ordered_families,
        "model_families": model_families,
        "test_families": test_families,
        "transition_distance_matrix_aligned": transition_distances_aligned.tolist(),
        "transition_distance_matrix_raw": transition_distances_raw.tolist(),
        "stationary_distance_matrix": stationary_distances.tolist(),
        "cross_family_log_likelihood_per_residue": cross_family.tolist(),
    }
    output_path = resolve_project_path(config.experiments["outputs"]["metrics_dir"], ROOT) / "family_comparison.json"
    write_json(output_path, metrics)
    print(f"Wrote family comparison metrics to {output_path}")


if __name__ == "__main__":
    main()
