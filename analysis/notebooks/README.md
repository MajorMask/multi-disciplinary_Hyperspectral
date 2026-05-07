# Notebooks

This folder is reserved for lightweight Jupyter notebooks that consume the reusable modules in `src/`.

Recommended workflow:

1. Use `analysis/scripts/run_experiment.py` with a YAML config to run baseline experiments.
2. Load `outputs/` artifacts in a notebook for figure generation and model comparison.
3. Keep business logic in `src/` so notebooks remain thin and reproducible.
