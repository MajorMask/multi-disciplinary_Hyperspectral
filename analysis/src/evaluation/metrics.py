from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[Iterable[str]] = None,
    include_kappa: bool = True,
) -> Dict[str, Any]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    if include_kappa:
        metrics["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))
    if labels is not None:
        label_indices = list(range(len(labels)))
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=label_indices
        )
        index = list(labels)
    else:
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
        index = None
    metrics["per_class"] = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        },
        index=index,
    )
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    return metrics


def save_classification_report(metrics: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = metrics["per_class"].copy()
    report["accuracy"] = metrics["accuracy"]
    report["balanced_accuracy"] = metrics["balanced_accuracy"]
    if "cohen_kappa" in metrics:
        report["cohen_kappa"] = metrics["cohen_kappa"]
    report.to_csv(output_path, index=True)


def save_confusion_matrix(cm: np.ndarray, labels: List[str], output_path: Path) -> None:
    df = pd.DataFrame(cm, index=labels, columns=labels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
