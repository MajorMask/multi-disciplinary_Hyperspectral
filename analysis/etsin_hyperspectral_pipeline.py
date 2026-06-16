 """
Etsin airborne hyperspectral pipeline
=====================================

This module supports reusable loading, preprocessing, feature extraction,
validation, and baseline modeling for airborne ENVI flightlines and
analysis-ready TIFF plot tiles.

Scientific goals:
- Recover forest location labels (site / plot identity) from spectra.
- Recover forest type labels from spectra.
- Use grouped validation to avoid spatial leakage.
- Provide a modular foundation for later spectral-spatial and domain-generalization work.

Expected input:
- ENVI flightline files: .hdr + .bsq / .dat
- Analysis-ready plot tile rasters: .tif
- Metadata tables linking plot IDs, site IDs, forest-type labels, and geometry

Notes:
- This script is intentionally dataset-agnostic; adapt the metadata mapping
  code to the file names and columns in the Etsin repository.
- If the dataset is not present in this workspace, the script still provides the
  reusable pipeline structure needed for future work.
"""

import argparse
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Optional dependencies for geospatial data
try:
    import geopandas as gpd
except ImportError:  # pragma: no cover
    gpd = None

try:
    import rasterio
    from rasterio.windows import Window
except ImportError:  # pragma: no cover
    rasterio = None

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_fscore_support,
                             r2_score, mean_squared_error)
from sklearn.model_selection import (GroupKFold, LeaveOneGroupOut,
                                     cross_val_predict, cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.cross_decomposition import PLSRegression

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# Dataset-specific wavelength handling
# -----------------------------------------------------------------------------

NOISY_WAVELENGTH_BOUNDS = [
    (1330, 1550),  # water absorption in short-wave infrared
    (1761, 2025),
    (2310, 2501),
]

REGION_BOUNDS = {
    "VIS": (400, 700),
    "Red-edge": (700, 800),
    "NIR": (800, 1350),
    "SWIR-1": (1550, 1760),
    "SWIR-2": (2025, 2310),
}

DEFAULT_FEATURE_METHOD = "mean"

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def ensure_rasterio_installed() -> None:
    if rasterio is None:
        raise ImportError(
            "rasterio is required for ENVI and GeoTIFF loading. "
            "Install with `pip install rasterio`."
        )


def ensure_geopandas_installed() -> None:
    if gpd is None:
        raise ImportError(
            "geopandas is required for geospatial metadata loading. "
            "Install with `pip install geopandas`."
        )


def parse_envi_list(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    text = str(value).strip()
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1]
    parts = [item.strip().strip('"\'') for item in text.split(",") if item.strip()]
    return parts


def parse_wavelengths(tags: Dict[str, str]) -> np.ndarray:
    wavelength_value = tags.get("wavelength") or tags.get("Wavelength")
    if wavelength_value is None:
        wavelength_value = tags.get("band names") or tags.get("band_names")
    if wavelength_value is None:
        raise ValueError("ENVI metadata does not contain wavelength information.")
    values = parse_envi_list(wavelength_value)
    try:
        return np.array([float(w) for w in values], dtype=float)
    except ValueError:
        raise ValueError("Unable to parse wavelength values from ENVI metadata.")


def mask_noisy_bands(wavelengths: np.ndarray,
                     noisy_bounds: List[Tuple[int, int]] = NOISY_WAVELENGTH_BOUNDS) -> np.ndarray:
    mask = np.ones_like(wavelengths, dtype=bool)
    for start, end in noisy_bounds:
        mask &= ~((wavelengths >= start) & (wavelengths < end))
    return mask


def select_wavelength_ranges(wavelengths: np.ndarray,
                             bounds: Dict[str, Tuple[int, int]] = REGION_BOUNDS) -> Dict[str, np.ndarray]:
    ranges = {}
    for name, (start, end) in bounds.items():
        ranges[name] = np.where((wavelengths >= start) & (wavelengths <= end))[0]
    return ranges


def ensure_path_exists(path: Union[str, Path]) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path


def load_metadata_table(path: Union[str, Path], geometry_col: Optional[str] = None) -> pd.DataFrame:
    path = ensure_path_exists(path)
    if path.suffix.lower() in {".geojson", ".json", ".shp", ".gpkg"}:
        ensure_geopandas_installed()
        df = gpd.read_file(path)
        if geometry_col is not None and geometry_col in df.columns:
            df = df.set_geometry(geometry_col)
        return pd.DataFrame(df)
    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "," if path.suffix.lower() == ".csv" else "\t"
        return pd.read_csv(path, sep=sep)
    raise ValueError(f"Unsupported metadata format: {path.suffix}")


def load_envi_flightline(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, Dict[str, str]]:
    ensure_rasterio_installed()
    path = ensure_path_exists(path)
    if path.suffix.lower() == ".bsq":
        hdr_path = path.with_suffix(".hdr")
    elif path.suffix.lower() == ".hdr":
        hdr_path = path
    else:
        raise ValueError("ENVI flightline loader expects a .hdr or .bsq file path.")

    with rasterio.open(hdr_path) as src:
        image = src.read().astype(np.float32)
        tags = {**src.tags(), **src.tags(ns="ENVI")}
        wavelengths = parse_wavelengths(tags)
        meta = {
            "name": src.name,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "crs": str(src.crs) if src.crs else "",
            "transform": src.transform.to_gdal() if src.transform is not None else "",
        }
        meta.update(tags)

    return image, wavelengths, meta


def load_geotiff_image(path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, str]]:
    ensure_rasterio_installed()
    path = ensure_path_exists(path)
    with rasterio.open(path) as src:
        image = src.read().astype(np.float32)
        tags = {**src.tags(), **src.tags(ns="TIFF")}
        meta = {
            "name": src.name,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "crs": str(src.crs) if src.crs else "",
            "transform": src.transform.to_gdal() if src.transform is not None else "",
        }
        meta.update(tags)
    return image, meta


def list_plot_tiles(path: Union[str, Path], extensions: Optional[List[str]] = None) -> List[Path]:
    path = ensure_path_exists(path)
    extensions = extensions or [".tif", ".tiff"]
    if path.is_file():
        return [path]
    tiles = [p for p in sorted(path.glob("**/*")) if p.suffix.lower() in extensions]
    if not tiles:
        raise FileNotFoundError(f"No TIFF tiles found in {path}")
    return tiles


def extract_plot_summary(image: np.ndarray,
                         summary: str = DEFAULT_FEATURE_METHOD,
                         nodata_value: Optional[float] = None) -> np.ndarray:
    if image.ndim == 2:
        image = image[np.newaxis, ...]
    bands, rows, cols = image.shape
    pixels = image.reshape(bands, -1)
    if nodata_value is not None:
        valid = ~(pixels == nodata_value)
        mask = np.all(valid, axis=0)
        pixels = pixels[:, mask]
    pixels = np.where(np.isfinite(pixels), pixels, np.nan)
    if summary == "mean":
        return np.nanmean(pixels, axis=1)
    if summary == "median":
        return np.nanmedian(pixels, axis=1)
    if summary == "std":
        return np.nanstd(pixels, axis=1)
    raise ValueError(f"Unsupported summary method: {summary}")


def extract_patch_spectra(image: np.ndarray,
                          patch_size: int = 9,
                          stride: int = 9,
                          summary: str = "mean") -> np.ndarray:
    if image.ndim != 3:
        raise ValueError("Patch extraction requires a 3D image array with shape (bands, rows, cols).")
    bands, rows, cols = image.shape
    patches = []
    for top in range(0, rows - patch_size + 1, stride):
        for left in range(0, cols - patch_size + 1, stride):
            patch = image[:, top:top + patch_size, left:left + patch_size]
            summary_vector = extract_plot_summary(patch, summary)
            patches.append(summary_vector)
    return np.stack(patches, axis=0) if patches else np.empty((0, bands), dtype=float)


def summarize_plot_tiles(tile_paths: List[Path],
                         summary: str = DEFAULT_FEATURE_METHOD,
                         nodata_value: Optional[float] = None) -> pd.DataFrame:
    records = []
    for path in tile_paths:
        image, meta = load_geotiff_image(path)
        summary_vector = extract_plot_summary(image, summary, nodata_value=nodata_value)
        record = {f"band_{i}": float(v) for i, v in enumerate(summary_vector)}
        record["tile_path"] = str(path)
        record["tile_name"] = path.stem
        records.append(record)
    return pd.DataFrame(records)


def normalize_spectra(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(X)


def build_feature_matrix(summary_df: pd.DataFrame,
                         metadata_df: pd.DataFrame,
                         tile_id_col: str,
                         plot_id_col: str,
                         site_col: str,
                         label_col: str,
                         drop_na_labels: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    if tile_id_col not in summary_df.columns:
        raise ValueError(f"Tile id column '{tile_id_col}' not found in summary table.")
    if plot_id_col not in metadata_df.columns:
        raise ValueError(f"Plot id column '{plot_id_col}' not found in metadata.")
    joint = summary_df.merge(metadata_df, left_on=tile_id_col, right_on=plot_id_col, how="left")
    if drop_na_labels:
        joint = joint.dropna(subset=[label_col])
    if joint.empty:
        raise ValueError("No matching records between summary table and metadata.")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(joint[label_col].astype(str).values)
    groups = joint[site_col].astype(str).values
    X = joint[[c for c in joint.columns if c.startswith("band_")]].astype(float).values
    labels = pd.Index(label_encoder.classes_)
    return X, y, groups, labels


def get_group_splits(groups: np.ndarray,
                     mode: str = "leave-one-site-out") -> Union[GroupKFold, LeaveOneGroupOut]:
    groups_unique = np.unique(groups)
    if mode == "leave-one-site-out":
        return LeaveOneGroupOut()
    if mode == "group-k-fold":
        return GroupKFold(n_splits=min(5, max(2, len(groups_unique))))
    raise ValueError(f"Unsupported CV mode: {mode}")


def evaluate_classification(clf: BaseEstimator,
                            X: np.ndarray,
                            y: np.ndarray,
                            groups: np.ndarray,
                            labels: pd.Index,
                            cv_mode: str = "leave-one-site-out") -> Dict[str, Union[float, np.ndarray, pd.DataFrame]]:
    cv = get_group_splits(groups, cv_mode)
    y_pred = cross_val_predict(clf, X, y, groups=groups, cv=cv, n_jobs=-1)
    acc = accuracy_score(y, y_pred)
    bal_acc = balanced_accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average="macro")
    precision, recall, f1, support = precision_recall_fscore_support(y, y_pred, labels=np.arange(len(labels)))
    cm = confusion_matrix(y, y_pred, labels=np.arange(len(labels)))
    report = pd.DataFrame({
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }, index=labels)
    return {
        "y_pred": y_pred,
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm,
        "classification_report": report,
    }


def evaluate_regression(model: BaseEstimator,
                        X: np.ndarray,
                        y: np.ndarray,
                        groups: np.ndarray,
                        cv_mode: str = "leave-one-site-out") -> Dict[str, float]:
    cv = get_group_splits(groups, cv_mode)
    y_pred = cross_val_predict(model, X, y, groups=groups, cv=cv, n_jobs=-1)
    valid = ~np.isnan(y) & ~np.isnan(y_pred)
    return {
        "r2": float(r2_score(y[valid], y_pred[valid])),
        "rmse": float(np.sqrt(mean_squared_error(y[valid], y_pred[valid]))),
        "y_pred": y_pred,
    }


def build_baseline_classifiers(random_state: int = 42) -> Dict[str, BaseEstimator]:
    return {
        "LogisticRegression": LogisticRegression(
            penalty="l2", solver="liblinear", class_weight="balanced", random_state=random_state,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=random_state, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, random_state=random_state,
        ),
    }


def fit_classification_baselines(X: np.ndarray,
                                 y: np.ndarray,
                                 groups: np.ndarray,
                                 labels: pd.Index,
                                 cv_mode: str = "leave-one-site-out") -> pd.DataFrame:
    results = []
    for name, clf in build_baseline_classifiers().items():
        metrics = evaluate_classification(clf, X, y, groups, labels, cv_mode=cv_mode)
        results.append({
            "method": name,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
        })
    return pd.DataFrame(results)


def build_plsda_classifier(n_components: int = 10, n_iter: int = 500) -> BaseEstimator:
    class PLSDAClassifier(BaseEstimator, ClassifierMixin):
        def __init__(self, n_components: int = n_components, max_iter: int = n_iter):
            self.n_components = n_components
            self.max_iter = max_iter
            self.pls_ = PLSRegression(n_components=n_components, max_iter=max_iter)
            self.classes_ = None
            self.encoder_ = OneHotEncoder(sparse=False)

        def fit(self, X: np.ndarray, y: np.ndarray):
            self.classes_ = np.unique(y)
            Y_oh = self.encoder_.fit_transform(y.reshape(-1, 1))
            self.pls_.fit(X, Y_oh)
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            probs = self.pls_.predict(X)
            return np.argmax(probs, axis=1)

    return PLSDAClassifier()


def run_baseline_experiments(summary_df: pd.DataFrame,
                             metadata_df: pd.DataFrame,
                             tile_id_col: str,
                             plot_id_col: str,
                             site_col: str,
                             label_col: str,
                             summary_method: str = DEFAULT_FEATURE_METHOD,
                             cv_mode: str = "leave-one-site-out") -> Dict[str, pd.DataFrame]:
    X, y, groups, labels = build_feature_matrix(
        summary_df, metadata_df, tile_id_col, plot_id_col, site_col, label_col
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    baseline_results = fit_classification_baselines(X_scaled, y, groups, labels, cv_mode=cv_mode)
    plsda = build_plsda_classifier()
    pls_metrics = evaluate_classification(plsda, X_scaled, y, groups, labels, cv_mode=cv_mode)
    baseline_results = baseline_results.append({
        "method": "PLS-DA",
        "accuracy": pls_metrics["accuracy"],
        "balanced_accuracy": pls_metrics["balanced_accuracy"],
        "macro_f1": pls_metrics["macro_f1"],
    }, ignore_index=True)

    return {
        "feature_matrix": X_scaled,
        "labels": labels,
        "classification_summary": baseline_results,
        "plsda_details": pls_metrics,
    }


def save_metrics(metrics: pd.DataFrame, output_dir: Union[str, Path], filename: str = "metrics.csv") -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    metrics.to_csv(Path(output_dir) / filename, index=False)


def save_confusion_matrix(cm: np.ndarray, labels: pd.Index, output_path: Union[str, Path]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.5), max(4, len(labels) * 0.5)))
    im = ax.imshow(cm.astype(float) / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


def report_grouped_cv_risks(metadata_df: pd.DataFrame,
                            site_col: str,
                            label_col: str) -> pd.DataFrame:
    if site_col not in metadata_df.columns or label_col not in metadata_df.columns:
        raise ValueError("Specified site or label column not found in metadata.")
    tab = metadata_df.groupby([site_col, label_col]).size().unstack(fill_value=0)
    tab["n_samples"] = tab.sum(axis=1)
    tab["n_labels"] = (tab.iloc[:, :-1] > 0).sum(axis=1)
    return tab


def build_script_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reusable Etsin hyperspectral modeling pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metadata", type=str, required=True,
                        help="Path to metadata table with plot/site labels.")
    parser.add_argument("--tile-dir", type=str, required=True,
                        help="Directory containing analysis-ready GeoTIFF plot tiles.")
    parser.add_argument("--tile-id-col", type=str, default="tile_name",
                        help="Column in tile summary / metadata that matches tile IDs.")
    parser.add_argument("--plot-id-col", type=str, default="tile_name",
                        help="Column in metadata that identifies each plot.")
    parser.add_argument("--site-col", type=str, default="site_id",
                        help="Column in metadata that identifies the site group.")
    parser.add_argument("--label-col", type=str, default="forest_type",
                        help="Column in metadata with the target forest-type label.")
    parser.add_argument("--summary-method", type=str, default=DEFAULT_FEATURE_METHOD,
                        choices=["mean", "median", "std"],
                        help="Summary statistic for plot tile spectral aggregation.")
    parser.add_argument("--cv-mode", type=str, default="leave-one-site-out",
                        choices=["leave-one-site-out", "group-k-fold"],
                        help="Grouped cross-validation mode.")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory for saving metrics and artifacts.")
    return parser


def run_main(args: argparse.Namespace) -> None:
    metadata_df = load_metadata_table(args.metadata)
    tile_paths = list_plot_tiles(args.tile_dir)
    summary_df = summarize_plot_tiles(tile_paths, summary=args.summary_method)
    experiment = run_baseline_experiments(
        summary_df=summary_df,
        metadata_df=metadata_df,
        tile_id_col=args.tile_id_col,
        plot_id_col=args.plot_id_col,
        site_col=args.site_col,
        label_col=args.label_col,
        summary_method=args.summary_method,
        cv_mode=args.cv_mode,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(experiment["classification_summary"], output_dir, filename="classification_summary.csv")
    experiment["plsda_details"]["classification_report"].to_csv(output_dir / "plsda_classification_report.csv")
    print("Saved classification summary and PLS-DA report.")


def main() -> None:
    parser = build_script_parser()
    args = parser.parse_args()
    run_main(args)


if __name__ == "__main__":
    main()
