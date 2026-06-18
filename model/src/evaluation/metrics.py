"""
Evaluation metrics and reporting for classification experiments.

Computes: overall accuracy, balanced accuracy, Cohen's kappa,
macro F1, per-class precision/recall, confusion matrix.

Generates: confusion matrix heatmaps, per-fold bar charts,
mean spectra per class plots, band importance plots.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute all classification metrics.

    Returns
    -------
    dict with scalar metrics and per-class arrays.
    """
    results = {
        "overall_accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "cohens_kappa": cohen_kappa_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "per_class_precision": precision_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "per_class_recall": recall_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "per_class_f1": f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=class_names).tolist() if class_names else confusion_matrix(y_true, y_pred).tolist(),
        "n_samples": len(y_true),
        "n_correct": int((y_true == y_pred).sum()),
    }

    if class_names:
        results["class_names"] = class_names
        results["classification_report"] = classification_report(
            y_true, y_pred, labels=class_names, target_names=class_names, zero_division=0
        )

    return results


def aggregate_fold_metrics(
    fold_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate metrics across CV folds (mean ± std).

    Parameters
    ----------
    fold_results : list of dicts from compute_metrics

    Returns
    -------
    dict with "mean_*" and "std_*" entries for each scalar metric.
    """
    scalar_keys = [
        "overall_accuracy", "balanced_accuracy", "cohens_kappa",
        "macro_f1", "macro_precision", "macro_recall",
    ]

    agg = {}
    for key in scalar_keys:
        values = [r[key] for r in fold_results if key in r]
        if values:
            agg[f"mean_{key}"] = float(np.mean(values))
            agg[f"std_{key}"] = float(np.std(values))

    # Aggregate confusion matrix
    cms = [np.array(r["confusion_matrix"]) for r in fold_results]
    agg["total_confusion_matrix"] = np.sum(cms, axis=0).tolist()

    agg["n_folds"] = len(fold_results)
    agg["total_samples"] = sum(r.get("n_samples", 0) for r in fold_results)

    return agg


def print_summary(agg: Dict[str, Any], model_name: str = "") -> str:
    """Format aggregated metrics as a readable summary string."""
    lines = []
    if model_name:
        lines.append(f"=== {model_name} ===")

    lines.append(f"Folds: {agg.get('n_folds', '?')}, "
                 f"Total samples: {agg.get('total_samples', '?')}")

    for metric in ["overall_accuracy", "balanced_accuracy", "cohens_kappa", "macro_f1"]:
        mean_key = f"mean_{metric}"
        std_key = f"std_{metric}"
        if mean_key in agg:
            lines.append(
                f"  {metric}: {agg[mean_key]:.3f} ± {agg[std_key]:.3f}"
            )

    summary = "\n".join(lines)
    logger.info(summary)
    return summary


# ---------- Visualization ----------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    output_path: Optional[Path] = None,
    normalize: bool = True,
):
    """
    Plot confusion matrix as a heatmap.

    Saves to output_path if provided, otherwise displays.
    """
    import matplotlib.pyplot as plt

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_plot = cm.astype(float) / row_sums
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_plot, cmap="Blues", vmin=0)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # Annotate cells
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm_plot[i, j]
            color = "white" if val > cm_plot.max() / 2 else "black"
            ax.text(j, i, format(val, fmt), ha="center", va="center", color=color)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved confusion matrix: {output_path}")
    else:
        plt.show()


def plot_per_fold_accuracy(
    fold_results: List[Dict[str, Any]],
    metric: str = "balanced_accuracy",
    title: str = "Per-Fold Performance",
    output_path: Optional[Path] = None,
):
    """Bar chart of metric across folds."""
    import matplotlib.pyplot as plt

    values = [r[metric] for r in fold_results]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(values)), values, color="steelblue", alpha=0.8)
    ax.axhline(np.mean(values), color="red", linestyle="--",
               label=f"Mean = {np.mean(values):.3f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_mean_spectra_per_class(
    X: np.ndarray,
    y: np.ndarray,
    wavelengths: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Mean Spectra by Forest Type",
    output_path: Optional[Path] = None,
):
    """Plot mean ± std spectra for each class."""
    import matplotlib.pyplot as plt

    classes = np.unique(y)
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = plt.cm.Set2(np.linspace(0, 1, len(classes)))
    for i, cls in enumerate(classes):
        mask = y == cls
        mean_spec = X[mask].mean(axis=0)
        std_spec = X[mask].std(axis=0)
        label = class_names[i] if class_names else str(cls)
        ax.plot(wavelengths, mean_spec, color=colors[i], label=label, linewidth=1.5)
        ax.fill_between(
            wavelengths,
            mean_spec - std_spec,
            mean_spec + std_spec,
            color=colors[i], alpha=0.15,
        )

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_band_importance(
    importances: np.ndarray,
    wavelengths: np.ndarray,
    top_n: int = 20,
    title: str = "Band Importance",
    output_path: Optional[Path] = None,
):
    """Plot feature importance across spectral bands."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(wavelengths, importances, width=12, color="steelblue", alpha=0.7)

    # Highlight top N
    top_idx = np.argsort(importances)[-top_n:]
    ax.bar(wavelengths[top_idx], importances[top_idx],
           width=12, color="tomato", alpha=0.9, label=f"Top {top_n}")

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Importance")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
