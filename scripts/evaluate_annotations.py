from __future__ import annotations

import numpy as np

from _bootstrap import bootstrap

ROOT = bootstrap()

from protein_hmm.analysis.metrics import adjusted_rand_index, q3_accuracy, segment_overlap_score
from protein_hmm.analysis.state_interpretation import state_label_enrichment
from protein_hmm.config import load_project_config
from protein_hmm.constants import DSSP_LABELS
from protein_hmm.data.encoding import AminoAcidEncoder
from protein_hmm.data.loaders import load_split_records
from protein_hmm.models.discrete_hmm import DiscreteHMM
from protein_hmm.utils.io import write_json
from protein_hmm.utils.paths import resolve_project_path


def infer_state_to_label_map(decoded_states: list[list[int]], label_sequences: list[str], num_states: int) -> dict[int, str]:
    enrichment = state_label_enrichment(decoded_states, label_sequences, num_states=num_states, label_order=DSSP_LABELS)
    return {state: DSSP_LABELS[int(np.argmax(enrichment[state]))] for state in range(num_states)}


def main() -> None:
    config = load_project_config(ROOT)
    splits = load_split_records(resolve_project_path(config.data["processed_dir"], ROOT))
    labeled_records = [record for record in splits["test"] if record.labels]
    if not labeled_records:
        raise RuntimeError("No labeled test records were found for annotation evaluation.")

    model = DiscreteHMM.load(resolve_project_path(config.experiments["outputs"]["model_dir"], ROOT) / "unsupervised_hmm.json")
    encoder = AminoAcidEncoder()

    decoded_paths: list[list[int]] = []
    label_sequences: list[str] = []
    for record in labeled_records:
        decoded = model.decode(encoder.encode(record.sequence), protein_id=record.protein_id, labels=record.labels)
        decoded_paths.append(decoded.states)
        label_sequences.append(record.labels or "")

    state_to_label = infer_state_to_label_map(decoded_paths, label_sequences, model.num_states)
    predicted_labels = ["".join(state_to_label[state] for state in path) for path in decoded_paths]

    flattened_states = [state for path in decoded_paths for state in path]
    flattened_labels = [label for labels in label_sequences for label in labels]
    metrics = {
        "ari": adjusted_rand_index(flattened_labels, flattened_states),
        "q3": q3_accuracy("".join(label_sequences), "".join(predicted_labels)),
        "sov": float(np.mean([segment_overlap_score(true, pred) for true, pred in zip(label_sequences, predicted_labels)])),
        "state_label_map": state_to_label,
        "state_label_enrichment": state_label_enrichment(
            decoded_paths,
            label_sequences,
            num_states=model.num_states,
            label_order=DSSP_LABELS,
        ).tolist(),
    }
    output_path = resolve_project_path(config.experiments["outputs"]["metrics_dir"], ROOT) / "annotation_evaluation.json"
    write_json(output_path, metrics)
    print(f"Wrote annotation evaluation to {output_path}")


if __name__ == "__main__":
    main()
