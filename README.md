# protein-hmm

Research codebase for discovering latent structure in protein sequences with
Hidden Markov Models.

The repository is organized for a class project rather than deployment:

- `scripts/` contains the main runnable entrypoints for each project stage.
- `src/protein_hmm/` contains reusable data, modeling, inference, analysis,
  and visualization code.
- `data/`, `results/`, and `reports/` separate raw inputs from intermediate
  artifacts and final writeup assets.

## Quickstart

1. Create a Python environment and install the package:

```bash
pip install -e .
```

2. Adjust the JSON-compatible YAML files in `configs/` for your data paths and
   experiment settings.

3. Run the workflow scripts from the repository root:

```bash
python scripts/build_dataset.py
python scripts/train_unsupervised.py
python scripts/run_model_selection.py
python scripts/evaluate_baselines.py
python scripts/make_report_tables.py
```

4. Run the lightweight test suite:

```bash
python -m unittest discover -s tests -v
```

## Configs

The files in `configs/` use `.yaml` extensions, but the default templates are
written in JSON syntax so they remain valid YAML and can be parsed without an
extra dependency. If `PyYAML` is installed, standard YAML syntax is also
supported.

## Main Workflow

- `scripts/scout_families.py`: summarize available protein families.
- `scripts/build_dataset.py`: load raw data, align labels, filter records, and
  create train/validation/test splits.
- `scripts/summarize_dataset.py`: write dataset summaries for QC.
- `scripts/train_unsupervised.py`: fit the main unsupervised discrete HMM.
- `scripts/train_family_models.py`: fit one HMM per family.
- `scripts/evaluate_annotations.py`: compare decoded states against DSSP labels.
- `scripts/train_reference_hmm.py`: fit a constrained reference HMM using
  label supervision.
- `scripts/run_model_selection.py`: evaluate candidate numbers of latent states
  with BIC, held-out likelihood, annotation metrics, and convergence diagnostics.
- `scripts/evaluate_baselines.py`: compute global, family-specific, and
  residue-specific DSSP label baselines.
- `scripts/make_report_figures.py`: generate report-ready plots.
- `scripts/make_report_tables.py`: export poster-ready tables and figure
  captions from generated metrics.

## Current Scope

This first implementation gives you:

- a clean scripts-first repository layout
- reusable object-oriented HMM code
- protein sequence encoding, filtering, and splitting utilities
- log-space forward-backward, Viterbi, and Baum-Welch routines
- simple baseline models
- a small smoke-tested pipeline on toy data

It is intentionally concise and modular so you can extend the biological data
pipeline without fighting the codebase structure.
