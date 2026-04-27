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
                "pseudocount": _fmt(row.get("pseudocount"), 4),
                "train_log_likelihood_per_residue": _fmt(row.get("train_log_likelihood_per_residue"), 4),
                "val_log_likelihood_per_residue": _fmt(row.get("val_log_likelihood_per_residue"), 4),
                "test_log_likelihood_per_residue": _fmt(row.get("test_log_likelihood_per_residue"), 4),
                "bic": _fmt(row.get("bic"), 2),
                "q3": _fmt(evaluation.get("q3")),
                "sov": _fmt(evaluation.get("sov")),
                "ari": _fmt(evaluation.get("ari")),
                "state_label_map": _state_map_text(evaluation.get("state_label_map")),
                "iterations": diagnostics.get("iterations", ""),
                "converged": diagnostics.get("converged", ""),
                "near_converged_per_observation": diagnostics.get("near_converged_per_observation", ""),
                "restarts_best_minus_worst": _fmt(
                    (max(row.get("restart_log_likelihoods", [0]) or [0])
                     - min(row.get("restart_log_likelihoods", [0]) or [0])),
                    2,
                ),
            }
        )
    return rows


def _family_self_rank_rows(family_comparison: dict[str, Any]) -> list[dict[str, Any]]:
    model_families = family_comparison.get("model_families") or family_comparison["families"]
    test_families = family_comparison.get("test_families") or family_comparison["families"]
    matrix = family_comparison["cross_family_log_likelihood_per_residue"]
    rows = []
    for test_index, family in enumerate(test_families):
        ranked = sorted(
            ((matrix[model_index][test_index], model_families[model_index]) for model_index in range(len(model_families))),
            reverse=True,
        )
        try:
            self_index = model_families.index(family)
            self_score = matrix[self_index][test_index]
            self_rank = [candidate[1] for candidate in ranked].index(family) + 1
        except ValueError:
            self_score = float("nan")
            self_rank = -1
        rows.append(
            {
                "test_family": family,
                "best_model_family": ranked[0][1],
                "best_log_likelihood_per_residue": _fmt(ranked[0][0]),
                "self_log_likelihood_per_residue": _fmt(self_score),
                "self_rank": self_rank,
            }
        )
    return rows


def _unified_rows(unified: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in unified.get("rows", []):
        rows.append(
            {
                "model": row["model"],
                "kind": row.get("kind", ""),
                "num_states": row.get("num_states", ""),
                "test_log_likelihood_per_residue": _fmt(row.get("test_log_likelihood_per_residue")),
                "val_log_likelihood_per_residue": _fmt(row.get("val_log_likelihood_per_residue")),
                "q3": _fmt(row.get("q3")),
                "sov": _fmt(row.get("sov")),
                "ari": _fmt(row.get("ari")),
                "bic": _fmt(row.get("bic"), 2),
            }
        )
    return rows


def _caption_text() -> str:
    return """# Figure Captions

## model_selection_bic.png
Train-set BIC for unsupervised HMMs at K=2,3,4,6,8 (lower is better). The BIC penalty grows roughly as O(K^2) through the transition matrix; the curve trades likelihood gain against this penalty.

## model_selection_val_ll.png
Validation log-likelihood per residue versus K. Useful as a held-out cross-check on BIC.

## em_convergence.png
Total training log-likelihood per Baum-Welch iteration for each candidate K. Used to confirm that EM converged (or to flag the cases where it did not).

## emission_heatmap.png / emission_enrichment.png
Per-state emission probabilities and the log2 enrichment of each amino acid relative to the training-set background. Enrichment is the more interpretable view since it removes the dominant frequency baseline.

## transition_heatmap.png
Latent transition matrix for the main unsupervised HMM. Diagonal mass quantifies state persistence (expected dwell length is 1/(1 - p_ii)).

## state_dssp_enrichment.png
P(DSSP class | latent state) on the test set after Hungarian-mapping latent states to H/E/C using the training set. Quantifies how much DSSP signal each unsupervised state captures.

## state_hydrophobicity.png / state_polarity.png
Per-state biochemical summaries: Kyte-Doolittle hydrophobicity (mass-weighted) and the probability mass on polar / charged residues.

## decoded_*.png
Example test proteins with the Viterbi state path overlaid on the DSSP label band, one per family.

## family_transition_distances.png
Pairwise Frobenius distance between family-specific transition matrices after Hungarian alignment of states by emission Jensen-Shannon distance. Without alignment, these distances are dominated by arbitrary state labelling.

## family_stationary_distances.png
Pairwise L1 distance between sorted stationary distributions across family models. Sorting makes the comparison permutation-invariant without requiring alignment.

## cross_family_log_likelihood.png
Per-residue log-likelihood of each family-trained HMM evaluated on every family's test sequences. Diagonal dominance is the indication that families have distinct sequential organisation.
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
    structure_summary_path = metrics_dir / "structure_annotation_summary.json"
    structure_summary = read_json(structure_summary_path) if structure_summary_path.exists() else None
    unified_path = metrics_dir / "unified_evaluation.json"
    unified = read_json(unified_path) if unified_path.exists() else None

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
            "pseudocount",
            "train_log_likelihood_per_residue",
            "val_log_likelihood_per_residue",
            "test_log_likelihood_per_residue",
            "bic",
            "q3",
            "sov",
            "ari",
            "state_label_map",
            "iterations",
            "converged",
            "near_converged_per_observation",
            "restarts_best_minus_worst",
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

    if unified is not None:
        _write_csv(
            table_dir / "unified_evaluation.csv",
            _unified_rows(unified),
            [
                "model",
                "kind",
                "num_states",
                "test_log_likelihood_per_residue",
                "val_log_likelihood_per_residue",
                "q3",
                "sov",
                "ari",
                "bic",
            ],
        )

    notes_dir.mkdir(parents=True, exist_ok=True)
    best_bic = min(model_selection["results"], key=lambda row: row["bic"])
    best_val = max(
        model_selection["results"],
        key=lambda row: row.get("val_log_likelihood_per_residue", row.get("val_log_likelihood", float("-inf"))),
    )
    baseline_best = max(baselines["baselines"], key=lambda row: row["q3"])

    structure_line = (
        f"- Structure-linked dataset: {structure_summary['num_annotated_records']} annotated proteins "
        f"from {structure_summary['num_seed_records']} Pfam seed candidates."
        if structure_summary is not None else ""
    )
    interpretation_K = config.models.get("model_selection", {}).get("interpretation_num_states")
    interpretation_row = next(
        (row for row in model_selection["results"] if row["num_states"] == interpretation_K),
        None,
    )

    unified_q3_lines = ""
    if unified is not None:
        unified_q3_lines = "\n".join(
            f"  - {row['model']}: Q3={_fmt(row.get('q3'))}, SOV={_fmt(row.get('sov'))}, "
            f"test LL/res={_fmt(row.get('test_log_likelihood_per_residue'))}"
            for row in unified["rows"]
            if row.get("q3") is not None or row.get("test_log_likelihood_per_residue") is not None
        )

    interpretation_text = ""
    if interpretation_row is not None:
        evaluation = interpretation_row.get("annotation_evaluation", {})
        interpretation_text = (
            f"- Interpretation HMM (K={interpretation_K}): val LL/res "
            f"{_fmt(interpretation_row.get('val_log_likelihood_per_residue'))}, "
            f"Q3 {_fmt(evaluation.get('q3'))}, SOV {_fmt(evaluation.get('sov'))}, "
            f"ARI {_fmt(evaluation.get('ari'))}."
        )

    key_results = f"""# Poster Key Results

{structure_line}
- Final split sizes: {dataset_summary['train']['num_proteins']} train, {dataset_summary['val']['num_proteins']} validation, {dataset_summary['test']['num_proteins']} test.
- BIC-selected unsupervised HMM: K={best_bic['num_states']} (BIC {_fmt(best_bic['bic'], 2)}). With small N, BIC's penalty pushes toward composition-only solutions; treat as a lower bound on useful K.
- Validation-LL-best HMM: K={best_val['num_states']} (val LL/res {_fmt(best_val.get('val_log_likelihood_per_residue'))}).
{interpretation_text}
- Best annotation baseline by Q3: {baseline_best['name']} (Q3 {_fmt(baseline_best['q3'])}, SOV {_fmt(baseline_best['sov'])}). The supervised reference HMM should improve on this; the unsupervised HMM is not expected to.
- Unified evaluation (test set):
{unified_q3_lines}
- Main interpretation: unsupervised HMMs primarily separate amino-acid composition regimes (hydrophobic vs polar), recovering some helix/coil signal but not strand boundaries; the constrained reference HMM is the upper bound on Q3/SOV achievable from these emissions alone.
"""
    (notes_dir / "poster_key_results.md").write_text(key_results, encoding="utf-8")
    (notes_dir / "figure_captions.md").write_text(_caption_text(), encoding="utf-8")
    print(f"Wrote report tables under {table_dir}")
    print(f"Wrote report notes under {notes_dir}")


if __name__ == "__main__":
    main()
