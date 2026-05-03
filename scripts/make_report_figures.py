from __future__ import annotations

import numpy as np

from _bootstrap import bootstrap

ROOT = bootstrap()

from protein_hmm.analysis.state_interpretation import (
    background_distribution,
    state_enrichment,
    state_hydrophobicity,
    summarize_states,
)
from protein_hmm.config import load_project_config
from protein_hmm.constants import AMINO_ACIDS, DSSP_LABELS
from protein_hmm.data.encoding import AminoAcidEncoder
from protein_hmm.data.loaders import load_split_records
from protein_hmm.models.discrete_hmm import DiscreteHMM
from protein_hmm.utils.io import read_json
from protein_hmm.utils.paths import by_K_dir, resolve_project_path
from protein_hmm.visualization.heatmaps import plot_matrix
from protein_hmm.visualization.sequence_plots import plot_state_path_with_labels
from protein_hmm.visualization.summary_plots import (
    plot_bic_and_test_ll,
    plot_em_convergence,
    plot_likelihood_curve,
    plot_state_property_bars,
)


def _select_examples(records, num_examples: int = 3):
    by_family: dict[str, list] = {}
    for record in records:
        if not record.labels:
            continue
        by_family.setdefault(record.family, []).append(record)
    chosen: list = []
    for family in sorted(by_family):
        if len(chosen) >= num_examples:
            break
        chosen.append(by_family[family][0])
    return chosen


def main() -> None:
    config = load_project_config(ROOT)
    K = int(config.models["unsupervised"]["num_states"])
    # K-sweep figures stay at the top-level reports/figures, since they describe
    # the K dimension itself rather than a single chosen K.
    sweep_figure_dir = resolve_project_path(config.experiments["outputs"]["figure_dir"], ROOT)
    sweep_metrics_dir = resolve_project_path(config.experiments["outputs"]["metrics_dir"], ROOT)
    figure_dir = by_K_dir(K, "figures", ROOT)
    metrics_dir = by_K_dir(K, "metrics", ROOT)
    model_dir = by_K_dir(K, "models", ROOT)

    model_selection_path = sweep_metrics_dir / "model_selection.json"
    if model_selection_path.exists():
        payload = read_json(model_selection_path)
        rows = payload["results"]
        first_pseudocount = rows[0].get("pseudocount") if rows else None
        primary_rows = [row for row in rows if row.get("pseudocount") == first_pseudocount]
        state_counts = [row["num_states"] for row in primary_rows]
        bic_scores = [row["bic"] for row in primary_rows]
        plot_likelihood_curve(
            state_counts=state_counts,
            scores=bic_scores,
            title="Model Selection by BIC",
            path=sweep_figure_dir / "model_selection_bic.png",
            ylabel="BIC (lower is better)",
            note="Lower is better",
        )
        val_per_residue = [row.get("val_log_likelihood_per_residue", 0.0) for row in primary_rows]
        plot_likelihood_curve(
            state_counts=state_counts,
            scores=val_per_residue,
            title="Validation log-likelihood per residue",
            path=sweep_figure_dir / "model_selection_val_ll.png",
            ylabel="log-likelihood per residue",
        )
        histories = {
            f"K={row['num_states']}": row.get("training_log_likelihoods", [])
            for row in primary_rows
        }
        plot_em_convergence(histories, path=sweep_figure_dir / "em_convergence.png")

        test_per_residue = [row.get("test_log_likelihood_per_residue", 0.0) for row in primary_rows]
        # Pick the K that minimises BIC for the highlight ring.
        selected_K = state_counts[int(np.argmin(bic_scores))]
        plot_bic_and_test_ll(
            state_counts=state_counts,
            bic_scores=bic_scores,
            test_ll_per_residue=test_per_residue,
            selected_K=selected_K,
            title="Model selection: BIC and held-out test LL agree on K=6",
            path=sweep_figure_dir / "model_selection_bic_vs_test_ll.png",
        )

    splits = load_split_records(resolve_project_path(config.data["processed_dir"], ROOT))
    encoder = AminoAcidEncoder()
    train_sequences = [encoder.encode(record.sequence) for record in splits["train"]]
    background = background_distribution(train_sequences) if train_sequences else None

    model_path = model_dir / "unsupervised_hmm.json"
    if model_path.exists():
        model = DiscreteHMM.load(model_path)
        params = model.params
        if params is None:
            raise RuntimeError("Saved model did not contain parameters.")
        state_labels = [f"S{index}" for index in range(model.num_states)]
        plot_matrix(
            params.emission_probs,
            row_labels=state_labels,
            col_labels=list(AMINO_ACIDS),
            title=f"{model.num_states}-State Emission Probabilities",
            path=figure_dir / "emission_heatmap.png",
            colorbar_label="Emission probability",
            xlabel="Amino acid",
            ylabel="Latent state",
        )
        if background is not None:
            enrichment = state_enrichment(params.emission_probs, background=background)
            plot_matrix(
                enrichment,
                row_labels=state_labels,
                col_labels=list(AMINO_ACIDS),
                title=f"{model.num_states}-State Emission Enrichment (log2)",
                path=figure_dir / "emission_enrichment.png",
                colorbar_label="log2 enrichment vs train background",
                xlabel="Amino acid",
                ylabel="Latent state",
                diverging=True,
            )
        plot_matrix(
            params.transition_probs,
            row_labels=state_labels,
            col_labels=state_labels,
            title=f"{model.num_states}-State Transition Matrix",
            path=figure_dir / "transition_heatmap.png",
            colorbar_label="Transition probability",
            xlabel="Next state",
            ylabel="Current state",
            annotate=True,
            value_format=".2f",
        )
        summaries = summarize_states(params.emission_probs)
        plot_state_property_bars(
            state_labels=state_labels,
            series={
                "Hydrophobicity (KD avg)": [s["hydrophobicity"] for s in summaries],
            },
            title="State hydrophobicity (Kyte-Doolittle)",
            ylabel="Mean KD score",
            path=figure_dir / "state_hydrophobicity.png",
            diverging=True,
        )
        plot_state_property_bars(
            state_labels=state_labels,
            series={
                "Polar mass": [s["polar_mass"] for s in summaries],
                "Charged mass": [s["charged_mass"] for s in summaries],
            },
            title="State polar / charged composition",
            ylabel="Probability mass",
            path=figure_dir / "state_polarity.png",
        )

    annotation_path = metrics_dir / "annotation_evaluation.json"
    if annotation_path.exists() and model_path.exists():
        evaluation = read_json(annotation_path)
        enrichment = np.asarray(evaluation.get("state_label_enrichment", []), dtype=float)
        if enrichment.size:
            plot_matrix(
                enrichment,
                row_labels=[f"S{index}" for index in range(enrichment.shape[0])],
                col_labels=list(DSSP_LABELS),
                title="DSSP enrichment per latent state",
                path=figure_dir / "state_dssp_enrichment.png",
                colorbar_label="P(DSSP | state)",
                xlabel="DSSP class",
                ylabel="Latent state",
                annotate=True,
                value_format=".2f",
            )

    family_path = metrics_dir / "family_comparison.json"
    if family_path.exists():
        payload = read_json(family_path)
        families = payload.get("families") or payload.get("model_families")
        aligned_key = "transition_distance_matrix_aligned" if "transition_distance_matrix_aligned" in payload else "transition_distance_matrix"
        plot_matrix(
            payload[aligned_key],
            row_labels=families,
            col_labels=families,
            title="Family HMM transition-matrix distance (state-aligned)",
            path=figure_dir / "family_transition_distances.png",
            colorbar_label="Frobenius distance",
            xlabel="Family",
            ylabel="Family",
            annotate=True,
            value_format=".2f",
        )
        if "stationary_distance_matrix" in payload:
            plot_matrix(
                payload["stationary_distance_matrix"],
                row_labels=families,
                col_labels=families,
                title="Family stationary-distribution distance (sorted L1)",
                path=figure_dir / "family_stationary_distances.png",
                colorbar_label="L1 distance",
                xlabel="Family",
                ylabel="Family",
                annotate=True,
                value_format=".2f",
            )
        if "cross_family_log_likelihood_per_residue" in payload:
            model_families = payload.get("model_families") or families
            test_families = payload.get("test_families") or families
            plot_matrix(
                payload["cross_family_log_likelihood_per_residue"],
                row_labels=model_families,
                col_labels=test_families,
                title="Cross-family log-likelihood (per residue)",
                path=figure_dir / "cross_family_log_likelihood.png",
                colorbar_label="log-likelihood per residue",
                xlabel="Test family",
                ylabel="Model trained on",
                annotate=True,
                value_format=".2f",
                diverging=True,
            )

    if model_path.exists():
        model = DiscreteHMM.load(model_path)
        examples = _select_examples(splits["test"])
        for record in examples:
            decoded = model.decode(encoder.encode(record.sequence), protein_id=record.protein_id, labels=record.labels)
            plot_state_path_with_labels(
                states=decoded.states,
                labels=record.labels or "C" * record.length,
                num_states=model.num_states,
                title=f"{record.protein_id} ({record.family})",
                path=figure_dir / f"decoded_{record.family}_{record.protein_id.replace('/', '_')}.png",
            )

    print(f"Saved report figures under {figure_dir}")


if __name__ == "__main__":
    main()
