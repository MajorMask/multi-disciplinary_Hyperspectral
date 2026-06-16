"""
Per-fold detailed reporting and diagnostics.

Generates a comprehensive per-fold report CSV and checks for
data leakage or spatial autocorrelation shortcut patterns
(Ploton et al. 2020, Roberts et al. 2017).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_fold_report(
    fold_results: List[Dict[str, Any]],
    stand_ids: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    experiment_name: str = "experiment",
) -> pd.DataFrame:
    """
    Build a detailed per-fold report DataFrame.

    Parameters
    ----------
    fold_results : list of dicts from compute_metrics, one per fold
    stand_ids : if available, the stand_id held out per fold
    output_dir : save CSV and JSON here

    Returns
    -------
    DataFrame with one row per fold and metric columns.
    """
    rows = []
    for i, fr in enumerate(fold_results):
        row = {
            "fold": i,
            "n_samples": fr.get("n_samples", 0),
            "overall_accuracy": fr.get("overall_accuracy", np.nan),
            "balanced_accuracy": fr.get("balanced_accuracy", np.nan),
            "cohens_kappa": fr.get("cohens_kappa", np.nan),
            "macro_f1": fr.get("macro_f1", np.nan),
        }
        if stand_ids and i < len(stand_ids):
            row["held_out_stand"] = stand_ids[i]
        rows.append(row)

    df = pd.DataFrame(rows)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / f"{experiment_name}_per_fold.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved per-fold report: {csv_path}")

        # Also save full results as JSON
        json_path = output_dir / f"{experiment_name}_full_results.json"
        serializable = []
        for fr in fold_results:
            s = {}
            for k, v in fr.items():
                if isinstance(v, np.ndarray):
                    s[k] = v.tolist()
                elif isinstance(v, (np.integer, np.floating)):
                    s[k] = float(v)
                else:
                    s[k] = v
            serializable.append(s)
        with open(json_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"Saved full results: {json_path}")

    return df


def check_shortcut_patterns(
    fold_results: List[Dict[str, Any]],
    threshold_perfect: float = 0.98,
    threshold_zero: float = 0.05,
) -> List[str]:
    """
    Check for suspicious patterns that indicate data leakage or shortcuts.

    Returns a list of warning strings (empty if no issues found).

    Based on Ploton et al. (2020) and Meyer & Pebesma (2022) guidance.
    """
    warnings = []

    accuracies = [r["overall_accuracy"] for r in fold_results]
    kappas = [r["cohens_kappa"] for r in fold_results]

    # Check 1: suspiciously perfect accuracy
    if np.mean(accuracies) > threshold_perfect:
        warnings.append(
            f"SUSPICIOUS: Mean accuracy {np.mean(accuracies):.3f} is near-perfect. "
            "Check for data leakage (duplicate samples, spatial autocorrelation, "
            "or label encoding in features)."
        )

    # Check 2: high variance across folds
    if len(accuracies) > 2 and np.std(accuracies) > 0.3:
        warnings.append(
            f"HIGH VARIANCE: Accuracy std = {np.std(accuracies):.3f}. "
            "This suggests unstable classification, possibly due to "
            "small sample size or high class imbalance."
        )

    # Check 3: some folds near zero
    zero_folds = sum(1 for a in accuracies if a < threshold_zero)
    if zero_folds > 0:
        warnings.append(
            f"ZERO FOLDS: {zero_folds} fold(s) have accuracy < {threshold_zero}. "
            "The model may fail on certain stand types. Check class balance."
        )

    # Check 4: negative kappa
    neg_kappas = sum(1 for k in kappas if k < 0)
    if neg_kappas > 0:
        warnings.append(
            f"NEGATIVE KAPPA: {neg_kappas} fold(s) have κ < 0 "
            "(worse than random). Model may not be learning meaningful patterns."
        )

    if warnings:
        for w in warnings:
            logger.warning(w)
    else:
        logger.info("No shortcut patterns detected.")

    return warnings
