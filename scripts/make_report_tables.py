from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from _bootstrap import bootstrap

ROOT = bootstrap()

from protein_hmm.config import load_project_config
from protein_hmm.utils.io import read_json
from protein_hmm.utils.paths import resolve_project_path


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _family_counts_text(counts: dict[str, int]) -> str:
    return "; ".join(f"{family}:{count}" for family, count in counts.items())


def _state_map_text(mapping: dict[str, str] | dict[int, str] | None) -> str:
    if not mapping:
        return ""
    return "; ".join(f"{state}->{label}" for state, label in mapping.items())


def _model_evaluation_rows(model_selection: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in model_selection.get("results", []):
        evaluation = row.get("annotation_evaluation", {})
        diagnostics = row.get("training_diagnostics", {})
        rows.append(
            {
                "num_states": row["num_states"],
                "train_log_likelihood": _fmt(row.get("train_log_likelihood"), 2),
                "val_log_likelihood": _fmt(row.get("val_log_likelihood"), 2),
                "test_log_likelihood": _fmt(row.get("test_log_likelihood"), 2),
                "bic": _fmt(row.get("bic"), 2),
                "q3": _fmt(evaluation.get("q3")),
                "sov": _fmt(evaluation.get("sov")),
                "ari": _fmt(evaluation.get("ari")),
                "state_label_map": _state_map_text(evaluation.get("state_label_map")),
                "iterations": diagnostics.get("iterations", ""),
                "converged": diagnostics.get("converged", ""),
                "last_delta_per_observation": _fmt(diagnostics.get("last_delta_per_observation"), 8),
                "near_converged_per_observation": diagnostics.get("near_converged_per_observation", ""),
            }
        )
    return rows


def _family_self_rank_rows(family_comparison: dict[str, Any]) -> list[dict[str, Any]]:
    families = family_comparison["families"]
    matrix = family_comparison["cross_family_log_likelihood_per_residue"]
    rows = []
    for test_index, family in enumerate(families):
        ranked = sorted(
            ((matrix[model_index][test_index], families[model_index]) for model_index in range(len(families))),
            reverse=True,
        )
        rows.append(
            {
                "test_family": family,
                "best_model_family": ranked[0][1],
                "best_log_likelihood_per_residue": _fmt(ranked[0][0]),
                "self_log_likelihood_per_residue": _fmt(matrix[test_index][test_index]),
                "self_rank": [candidate[1] for candidate in ranked].index(family) + 1,
            }
        )
    return rows


def _caption_text() -> str:
    return """# Figure Captions

## Model Selection by BIC
Candidate unsupervised HMMs were fit with 2, 3, 4, 6, and 8 latent states. Lower BIC indicates stronger support after penalizing additional transition, emission, and start-probability parameters.

## 2-State Emission Probabilities
Rows show latent states and columns show amino-acid emission probabilities. The selected 2-state model separates amino-acid composition regimes rather than DSSP secondary-structure classes.

## 2-State Transition Matrix
Entries show transition probabilities between latent states. State 1 is comparatively persistent, while State 0 more often transitions into State 1.

## Family HMM Transition-Matrix Distances
Pairwise Frobenius distances between family-specific transition matrices. Smaller values indicate families whose learned transition dynamics are more similar.
"""


def main() -> None:
    config = load_project_config(ROOT)
    metrics_dir = resolve_project_path(config.experiments["outputs"]["metrics_dir"], ROOT)
    table_dir = resolve_project_path("reports/tables", ROOT)
    notes_dir = resolve_project_path("reports/notes", ROOT)

    dataset_summary = read_json(metrics_dir / "dataset_summary.json")
    model_selection = read_json(metrics_dir / "model_selection.json")
    baselines = read_json(metrics_dir / "baselines.json")
    family_comparison = read_json(metrics_dir / "family_comparison.json")
    structure_summary = read_json(metrics_dir / "structure_annotation_summary.json")

    dataset_rows = []
    for split_name in ("train", "val", "test", "all"):
        split = dataset_summary[split_name]
        dataset_rows.append(
            {
                "split": split_name,
                "num_proteins": split["num_proteins"],
                "min_length": split["length_summary"]["min"],
                "max_length": split["length_summary"]["max"],
                "mean_length": _fmt(split["length_summary"]["mean"], 2),
                "family_counts": _family_counts_text(split["family_counts"]),
            }
        )
    _write_csv(
        table_dir / "dataset_summary.csv",
        dataset_rows,
        ["split", "num_proteins", "min_length", "max_length", "mean_length", "family_counts"],
    )

    _write_csv(
        table_dir / "model_evaluation.csv",
        _model_evaluation_rows(model_selection),
        [
            "num_states",
            "train_log_likelihood",
            "val_log_likelihood",
            "test_log_likelihood",
            "bic",
            "q3",
            "sov",
            "ari",
            "state_label_map",
            "iterations",
            "converged",
            "last_delta_per_observation",
            "near_converged_per_observation",
        ],
    )

    baseline_rows = [
        {
            "baseline": row["name"],
            "q3": _fmt(row["q3"]),
            "sov": _fmt(row["sov"]),
            "mapping": row["mapping"] if isinstance(row["mapping"], str) else _state_map_text(row["mapping"]),
        }
        for row in baselines["baselines"]
    ]
    _write_csv(table_dir / "baseline_metrics.csv", baseline_rows, ["baseline", "q3", "sov", "mapping"])

    _write_csv(
        table_dir / "family_model_self_ranks.csv",
        _family_self_rank_rows(family_comparison),
        [
            "test_family",
            "best_model_family",
            "best_log_likelihood_per_residue",
            "self_log_likelihood_per_residue",
            "self_rank",
        ],
    )

    notes_dir.mkdir(parents=True, exist_ok=True)
    best_bic = min(model_selection["results"], key=lambda row: row["bic"])
    best_val = max(model_selection["results"], key=lambda row: row["val_log_likelihood"])
    baseline_best = max(baselines["baselines"], key=lambda row: row["q3"])
    key_results = f"""# Poster Key Results

- Structure-linked dataset: {structure_summary["num_annotated_records"]} annotated proteins from {structure_summary["num_seed_records"]} Pfam seed candidates.
- Final split sizes: {dataset_summary["train"]["num_proteins"]} train, {dataset_summary["val"]["num_proteins"]} validation, {dataset_summary["test"]["num_proteins"]} test.
- BIC-selected unsupervised HMM: {best_bic["num_states"]} latent states with BIC {_fmt(best_bic["bic"], 2)}.
- Validation-likelihood-selected HMM: {best_val["num_states"]} latent states with validation log-likelihood {_fmt(best_val["val_log_likelihood"], 2)}.
- Best annotation baseline by Q3: {baseline_best["name"]} with Q3 {_fmt(baseline_best["q3"])} and SOV {_fmt(baseline_best["sov"])}.
- Main interpretation: unsupervised HMMs capture amino-acid composition and family-level dynamics more clearly than DSSP secondary-structure classes.
"""
    (notes_dir / "poster_key_results.md").write_text(key_results, encoding="utf-8")
    (notes_dir / "figure_captions.md").write_text(_caption_text(), encoding="utf-8")
    print(f"Wrote report tables under {table_dir}")
    print(f"Wrote report notes under {notes_dir}")


if __name__ == "__main__":
    main()
