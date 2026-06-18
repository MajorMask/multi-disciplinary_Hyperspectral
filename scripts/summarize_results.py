#!/usr/bin/env python3
"""
Results Summary Script — generates a formatted report from experiment outputs.

Reads the JSON metrics files and per-fold CSVs produced by runner.py,
prints a clean summary suitable for pasting into the DOCX report.

Usage:
    python scripts/summarize_results.py --results-dir outputs/baseline_hyytialä_casi
    python scripts/summarize_results.py --results-dir outputs/baseline_hyytialä_casi --format markdown
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_experiment_summary(results_dir: Path) -> dict:
    """Load experiment_summary.json."""
    summary_path = results_dir / "experiment_summary.json"
    if not summary_path.exists():
        print(f"ERROR: No experiment_summary.json in {results_dir}")
        print("Run the experiment first: python scripts/run_baseline.py")
        sys.exit(1)

    with open(summary_path) as f:
        return json.load(f)


def format_table_row(cells: list, widths: list) -> str:
    return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " |"


def format_separator(widths: list) -> str:
    return "|" + "|".join("-" * (w + 2) for w in widths) + "|"


def print_summary(summary: dict, fmt: str = "text"):
    """Print formatted experiment summary."""
    print()
    print("=" * 70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)

    # Dataset info
    print(f"\nDataset: {summary.get('n_tiles', '?')} tiles loaded, "
          f"{summary.get('n_matched', '?')} matched to metadata")
    print(f"Bands: {summary.get('n_bands_original', '?')} original → "
          f"{summary.get('n_bands_filtered', '?')} after water vapor exclusion")
    print(f"CV strategy: {summary.get('cv_strategy', '?')} ({summary.get('n_folds', '?')} folds)")

    dist = summary.get("class_distribution", {})
    if dist:
        print(f"Class distribution: {dist}")

    # Model comparison table
    models = summary.get("models", {})
    if not models:
        print("\nNo model results found.")
        return

    print(f"\n{'='*70}")
    print("MODEL COMPARISON")
    print(f"{'='*70}")

    headers = ["Model", "OA", "BA", "Kappa", "F1 (macro)", "Warnings"]
    widths = [20, 12, 12, 12, 12, 10]

    if fmt == "markdown":
        print(format_table_row(headers, widths))
        print(format_separator(widths))
    else:
        print(f"{'Model':<20s} {'OA':>12s} {'BA':>12s} {'Kappa':>12s} {'F1':>12s} {'Warn':>6s}")
        print("-" * 70)

    for model_name, mdata in sorted(models.items()):
        oa = mdata.get("mean_overall_accuracy", float("nan"))
        oa_std = mdata.get("std_overall_accuracy", 0)
        ba = mdata.get("mean_balanced_accuracy", float("nan"))
        ba_std = mdata.get("std_balanced_accuracy", 0)
        kappa = mdata.get("mean_cohens_kappa", float("nan"))
        kappa_std = mdata.get("std_cohens_kappa", 0)
        f1 = mdata.get("mean_macro_f1", float("nan"))
        f1_std = mdata.get("std_macro_f1", 0)
        n_warn = len(mdata.get("warnings", []))

        if fmt == "markdown":
            row = [
                model_name,
                f"{oa:.3f}±{oa_std:.3f}",
                f"{ba:.3f}±{ba_std:.3f}",
                f"{kappa:.3f}±{kappa_std:.3f}",
                f"{f1:.3f}±{f1_std:.3f}",
                str(n_warn) if n_warn else "—",
            ]
            print(format_table_row(row, widths))
        else:
            print(f"{model_name:<20s} {oa:.3f}±{oa_std:.3f}  {ba:.3f}±{ba_std:.3f}  "
                  f"{kappa:.3f}±{kappa_std:.3f}  {f1:.3f}±{f1_std:.3f}  {n_warn:>4d}")

    # Best model
    best_model = max(models.items(), key=lambda x: x[1].get("mean_balanced_accuracy", 0))
    print(f"\nBest model (by balanced accuracy): {best_model[0]} "
          f"(BA = {best_model[1].get('mean_balanced_accuracy', 0):.3f})")

    # Warnings
    all_warnings = []
    for model_name, mdata in models.items():
        for w in mdata.get("warnings", []):
            all_warnings.append(f"  [{model_name}] {w}")

    if all_warnings:
        print(f"\n{'='*70}")
        print("DIAGNOSTICS / WARNINGS")
        print(f"{'='*70}")
        for w in all_warnings:
            print(w)

    # Confusion matrix for best model
    best_cm = best_model[1].get("total_confusion_matrix")
    if best_cm:
        print(f"\n{'='*70}")
        print(f"CONFUSION MATRIX — {best_model[0]} (aggregated across folds)")
        print(f"{'='*70}")
        cm = np.array(best_cm)
        # Try to get class names
        print("(rows = true, columns = predicted)")
        for i, row in enumerate(cm):
            print(f"  Class {i}: {row.tolist()}")

    # Figures produced
    figures_dir = Path(results_dir) / "figures"
    if figures_dir.exists():
        figs = sorted(figures_dir.glob("*.png"))
        if figs:
            print(f"\n{'='*70}")
            print(f"FIGURES GENERATED ({len(figs)} files)")
            print(f"{'='*70}")
            for fig in figs:
                print(f"  {fig.name}")

    # Timing
    timings = summary.get("timings", {})
    if timings:
        print(f"\nRuntime: {timings.get('total', 0):.1f}s total "
              f"(load: {timings.get('load_tiles', 0):.1f}s, "
              f"classify: {timings.get('classification', 0):.1f}s)")

    # Report-ready text block
    print(f"\n{'='*70}")
    print("REPORT-READY TEXT (copy into Section 3 of the DOCX)")
    print(f"{'='*70}")
    print()
    print(f"The baseline experiment evaluated five classifiers on stand-level mean "
          f"CASI spectra ({summary.get('n_bands_filtered', '?')} bands after water vapor "
          f"exclusion) using leave-one-stand-out cross-validation "
          f"({summary.get('n_folds', '?')} folds, {summary.get('n_matched', '?')} stands). "
          f"Class distribution: {dist}.")
    print()
    print(f"The best-performing model was {best_model[0]} with a balanced accuracy of "
          f"{best_model[1].get('mean_balanced_accuracy', 0):.3f} "
          f"(± {best_model[1].get('std_balanced_accuracy', 0):.3f}) "
          f"and Cohen's kappa of {best_model[1].get('mean_cohens_kappa', 0):.3f} "
          f"(± {best_model[1].get('std_cohens_kappa', 0):.3f}).")
    print()

    if all_warnings:
        print("Diagnostic checks flagged the following concerns: "
              + "; ".join(w.strip() for w in all_warnings[:3]) + ".")
    else:
        print("No diagnostic warnings were triggered (permutation test, shortcut detection).")


def main():
    parser = argparse.ArgumentParser(description="Summarize experiment results")
    parser.add_argument("--results-dir", required=True, help="Path to experiment output directory")
    parser.add_argument("--format", choices=["text", "markdown"], default="text")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        # Try under outputs/
        results_dir = Path("outputs") / args.results_dir
        if not results_dir.exists():
            print(f"ERROR: Results directory not found: {args.results_dir}")
            sys.exit(1)

    summary = load_experiment_summary(results_dir)
    print_summary(summary, args.format)


if __name__ == "__main__":
    main()
