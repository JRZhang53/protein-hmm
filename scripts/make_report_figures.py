from __future__ import annotations

from _bootstrap import bootstrap

ROOT = bootstrap()

from protein_hmm.config import load_project_config
from protein_hmm.constants import AMINO_ACIDS
from protein_hmm.models.discrete_hmm import DiscreteHMM
from protein_hmm.utils.io import read_json
from protein_hmm.utils.paths import resolve_project_path
from protein_hmm.visualization.heatmaps import plot_matrix
from protein_hmm.visualization.summary_plots import plot_likelihood_curve


def main() -> None:
    config = load_project_config(ROOT)
    figure_dir = resolve_project_path("reports/figures", ROOT)

    model_selection_path = resolve_project_path(config.experiments["outputs"]["metrics_dir"], ROOT) / "model_selection.json"
    if model_selection_path.exists():
        payload = read_json(model_selection_path)
        state_counts = [row["num_states"] for row in payload["results"]]
        bic_scores = [row["bic"] for row in payload["results"]]
        plot_likelihood_curve(
            state_counts=state_counts,
            scores=bic_scores,
            title="Model Selection (BIC)",
            path=figure_dir / "model_selection_bic.png",
        )

    model_path = resolve_project_path(config.experiments["outputs"]["model_dir"], ROOT) / "unsupervised_hmm.json"
    if model_path.exists():
        model = DiscreteHMM.load(model_path)
        params = model.params
        if params is None:
            raise RuntimeError("Saved model did not contain parameters.")
        plot_matrix(
            params.emission_probs,
            row_labels=[f"State {index}" for index in range(model.num_states)],
            col_labels=list(AMINO_ACIDS),
            title="Emission Probabilities",
            path=figure_dir / "emission_heatmap.png",
        )
        plot_matrix(
            params.transition_probs,
            row_labels=[f"State {index}" for index in range(model.num_states)],
            col_labels=[f"State {index}" for index in range(model.num_states)],
            title="Transition Matrix",
            path=figure_dir / "transition_heatmap.png",
        )

    family_path = resolve_project_path(config.experiments["outputs"]["metrics_dir"], ROOT) / "family_comparison.json"
    if family_path.exists():
        payload = read_json(family_path)
        families = payload["families"]
        plot_matrix(
            payload["transition_distance_matrix"],
            row_labels=families,
            col_labels=families,
            title="Family Transition Distances",
            path=figure_dir / "family_transition_distances.png",
        )
    print(f"Saved report figures under {figure_dir}")


if __name__ == "__main__":
    main()
