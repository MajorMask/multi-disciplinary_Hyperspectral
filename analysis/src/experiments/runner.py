from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, cross_val_predict

from src.datasets.dataset import PlotTileDataset
from model.src.evaluation.metrics import classification_metrics, save_classification_report, save_confusion_matrix
from src.models.baselines import build_baseline_pipelines, build_pca_classifier
from src.utils.config import load_config, save_json


@dataclass
class ExperimentConfig:
    experiment_name: str
    metadata_path: Path
    tile_dir: Path
    tile_id_column: str
    plot_id_column: str
    site_id_column: str
    label_column: str
    product_type: Optional[str] = None
    feature_method: str = "mean"
    normalize: bool = False
    normalization_method: str = "standard"
    cv_strategy: str = "leave-one-site-out"
    model_names: List[str] = None
    output_dir: Path = Path("outputs")

    def __post_init__(self) -> None:
        if self.model_names is None:
            self.model_names = ["LogisticRegression", "RandomForest", "GradientBoosting"]


def make_splitter(strategy: str):
    if strategy == "leave-one-site-out":
        return LeaveOneGroupOut()
    if strategy == "group-k-fold":
        return GroupKFold(n_splits=5)
    raise ValueError(f"Unsupported cross-validation strategy: {strategy}")


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = PlotTileDataset(
            tile_dir=self.config.tile_dir,
            metadata_path=self.config.metadata_path,
            tile_id_column=self.config.tile_id_column,
            plot_id_column=self.config.plot_id_column,
            site_id_column=self.config.site_id_column,
            label_column=self.config.label_column,
            product_type=self.config.product_type,
        )

    def run(self) -> Dict[str, Any]:
        summary_df = self.dataset.summarize_tiles(
            summary_method=self.config.feature_method,
            normalize=self.config.normalize,
            normalize_method=self.config.normalization_method,
        )
        X, y_str, groups, labels = self.dataset.build_feature_matrix(
            summary_df=summary_df,
            feature_mode=self.config.feature_method,
            normalize=self.config.normalize,
            normalize_method=self.config.normalization_method,
        )
        target_encoder = {label: idx for idx, label in enumerate(labels)}
        y = np.array([target_encoder[label] for label in y_str], dtype=int)
        splitter = make_splitter(self.config.cv_strategy)
        model_pipelines = build_baseline_pipelines()
        if "PCA" in self.config.model_names:
            model_pipelines["PCA"] = build_pca_classifier()
        results: Dict[str, Any] = {}
        predictions: Dict[str, List[int]] = {}

        for model_name in self.config.model_names:
            if model_name not in model_pipelines:
                raise ValueError(f"Unknown baseline model: {model_name}")
            pipeline = model_pipelines[model_name]
            y_pred = cross_val_predict(pipeline, X, y, groups=groups, cv=splitter, n_jobs=-1)
            metrics = classification_metrics(y, y_pred, labels=labels)
            results[model_name] = metrics
            predictions[model_name] = y_pred.tolist()
            self._save_model_outputs(model_name, metrics, y, y_pred, labels)

        summary = {
            "experiment_name": self.config.experiment_name,
            "config": self._config_to_dict(),
            "results": {
                name: {
                    "accuracy": float(metrics["accuracy"]),
                    "balanced_accuracy": float(metrics["balanced_accuracy"]),
                    "macro_f1": float(metrics["macro_f1"]),
                    "cohen_kappa": float(metrics.get("cohen_kappa", np.nan)),
                }
                for name, metrics in results.items()
            },
        }
        save_json(summary, self.config.output_dir / "experiment_summary.json")
        return summary

    def _save_model_outputs(self, model_name: str, metrics: Dict[str, Any], y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> None:
        prefix = self.config.output_dir / model_name.lower().replace(" ", "_")
        prefix.mkdir(parents=True, exist_ok=True)
        save_classification_report(metrics, prefix / "classification_report.csv")
        save_confusion_matrix(metrics["confusion_matrix"], labels, prefix / "confusion_matrix.csv")
        save_json({"y_true": y_true.tolist(), "y_pred": y_pred.tolist()}, prefix / "predictions.json")

    def _config_to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_name": self.config.experiment_name,
            "metadata_path": str(self.config.metadata_path),
            "tile_dir": str(self.config.tile_dir),
            "tile_id_column": self.config.tile_id_column,
            "plot_id_column": self.config.plot_id_column,
            "site_id_column": self.config.site_id_column,
            "label_column": self.config.label_column,
            "product_type": self.config.product_type,
            "feature_method": self.config.feature_method,
            "normalize": self.config.normalize,
            "normalization_method": self.config.normalization_method,
            "cv_strategy": self.config.cv_strategy,
            "model_names": self.config.model_names,
        }


def run_experiment_from_config(config_path: Path) -> Dict[str, Any]:
    config_data = load_config(config_path)
    config = ExperimentConfig(
        experiment_name=config_data.get("experiment_name", "baseline_experiment"),
        metadata_path=Path(config_data["metadata_path"]),
        tile_dir=Path(config_data["tile_dir"]),
        tile_id_column=config_data.get("tile_id_column", "tile_name"),
        plot_id_column=config_data.get("plot_id_column", "plot_id"),
        site_id_column=config_data.get("site_id_column", "site_id"),
        label_column=config_data.get("label_column", "forest_type"),
        product_type=config_data.get("product_type"),
        feature_method=config_data.get("feature_method", "mean"),
        normalize=config_data.get("normalize", False),
        normalization_method=config_data.get("normalization_method", "standard"),
        cv_strategy=config_data.get("cv_strategy", "leave-one-site-out"),
        model_names=config_data.get(
            "model_names", ["LogisticRegression", "RandomForest", "GradientBoosting"]
        ),
        output_dir=Path(config_data.get("output_dir", "outputs")),
    )
    runner = ExperimentRunner(config)
    return runner.run()
