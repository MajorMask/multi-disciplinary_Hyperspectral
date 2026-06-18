"""
Experiment runner — orchestrates the full pipeline from config to results.

Usage:
    python -m src.experiments.runner --config config/default.yaml

Pipeline:
    1. Load config
    2. Discover and load raster tiles
    3. Load metadata, match to tiles
    4. Select bands, normalize
    5. Extract features (stand-level or patch-level)
    6. Run cross-validated classification
    7. Aggregate metrics, save results and figures
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from model.src.dataio.raster_loader import load_all_tiles
from model.src.dataio.metadata_loader import load_metadata, merge_tiles_with_metadata
from model.src.dataio.splits import get_cv_splitter
from model.src.preprocessing.band_selection import get_valid_band_mask, apply_band_selection
from model.src.preprocessing.normalization import apply_normalization
from model.src.features.stand_summary import build_stand_feature_matrix
from model.src.features.pca_features import fit_pca, transform_pca, select_n_components_by_variance
from model.src.models.classical import build_classifier, train_and_predict
from model.src.evaluation.metrics import (
    compute_metrics,
    aggregate_fold_metrics,
    print_summary,
    plot_confusion_matrix,
    plot_per_fold_accuracy,
    plot_mean_spectra_per_class,
    plot_band_importance,
)
from model.src.evaluation.per_fold_report import build_fold_report, check_shortcut_patterns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return cfg


def resolve_paths(cfg: Dict) -> Dict:
    """Resolve relative paths against data root directory."""
    root = Path(cfg["data"]["root_dir"])
    for key in ["metadata_file", "casi_tile_dir", "sasi_tile_dir", "als_dem_dir"]:
        if key in cfg["data"]:
            p = Path(cfg["data"][key])
            if not p.is_absolute():
                cfg["data"][key] = str(root / p)
    return cfg


def run_experiment(cfg: Dict) -> Dict[str, Any]:
    """
    Execute the full classification experiment.

    Returns dict with all results, metrics, and diagnostics.
    """
    t0 = time.time()
    results = {"config": cfg, "timings": {}}

    # ---- 1. Resolve paths ----
    cfg = resolve_paths(cfg)

    # ---- 2. Load raster tiles ----
    sensor = cfg["preprocessing"]["sensor"]
    tile_dir_key = f"{sensor}_tile_dir"
    tile_dir = cfg["data"].get(tile_dir_key, cfg["data"]["casi_tile_dir"])
    pattern = cfg["data"].get("tile_filename_pattern", "*.tif")

    logger.info(f"Loading {sensor.upper()} tiles from {tile_dir}")
    tiles = load_all_tiles(
        tile_dir,
        pattern=pattern,
        nodata_value=cfg["preprocessing"]["nodata_value"],
        min_valid_fraction=cfg["preprocessing"]["min_valid_pixel_fraction"],
    )
    results["n_tiles_loaded"] = len(tiles)
    results["timings"]["load_tiles"] = time.time() - t0

    if not tiles:
        raise RuntimeError(f"No tiles loaded from {tile_dir}")

    # Inject known wavelengths if tiles lack them (band indices 0..N-1)
    known_wl_key = f"{sensor}_wavelengths_nm"
    known_wavelengths = cfg["preprocessing"].get(known_wl_key)
    if known_wavelengths and np.array_equal(tiles[0].wavelengths, np.arange(tiles[0].n_bands)):
        wl_array = np.array(known_wavelengths, dtype=np.float64)
        for tile in tiles:
            if len(wl_array) == tile.n_bands:
                tile.wavelengths = wl_array
        logger.info(f"Injected {len(wl_array)} known {sensor.upper()} wavelengths")

    # Apply reflectance scale factor (raw values are integer reflectance * scale)
    scale = cfg["preprocessing"].get("reflectance_scale_factor", 1.0)
    if scale != 1.0:
        for tile in tiles:
            tile.image = tile.image / scale
        logger.info(f"Applied reflectance scale factor 1/{scale}")

    # ---- 3. Load metadata ----
    t1 = time.time()
    meta_cfg = cfg.get("metadata", {})
    metadata = load_metadata(
        cfg["data"]["metadata_file"],
        stand_id_col=meta_cfg.get("stand_id_col", "stand_id"),
        forest_type_col=meta_cfg.get("forest_type_col", "forest_type"),
    )

    # Strip sensor suffix from tile stems (e.g. "HY_BIRCH1_CASI" → "HY_BIRCH1")
    sensor_upper = sensor.upper()
    for tile in tiles:
        if tile.stand_id.endswith(f"_{sensor_upper}"):
            tile.stand_id = tile.stand_id[: -(len(sensor_upper) + 1)]

    tile_ids = [t.stand_id for t in tiles]
    matched = merge_tiles_with_metadata(tile_ids, metadata)
    results["n_matched_stands"] = len(matched)
    results["class_distribution"] = matched["forest_type"].value_counts().to_dict()
    results["timings"]["load_metadata"] = time.time() - t1

    # Filter tiles to only matched ones
    matched_ids = set(matched["stand_id"])
    tiles = [t for t in tiles if t.stand_id in matched_ids]

    # ---- 4. Band selection ----
    t2 = time.time()
    wavelengths = tiles[0].wavelengths
    exclude_ranges = cfg["preprocessing"].get("exclude_wavelength_ranges_nm", [])
    exclude_tuples = [tuple(r) for r in exclude_ranges]
    band_mask = get_valid_band_mask(wavelengths, exclude_tuples)
    filtered_wavelengths = wavelengths[band_mask]
    results["n_bands_original"] = len(wavelengths)
    results["n_bands_after_filter"] = int(band_mask.sum())
    results["timings"]["band_selection"] = time.time() - t2

    # ---- 5. Feature extraction ----
    t3 = time.time()
    feature_level = cfg["features"]["level"]

    if feature_level == "stand":
        summary_method = cfg["features"].get("stand_summary", "mean")
        X, stand_ids = build_stand_feature_matrix(
            tiles, method=summary_method, band_mask=band_mask
        )
    else:
        raise NotImplementedError(
            f"Feature level '{feature_level}' not yet implemented in runner. "
            "Use stand-level for baseline experiments."
        )

    # Build label vector aligned with X
    id_to_label = dict(zip(matched["stand_id"], matched["forest_type"]))
    y = np.array([id_to_label[sid] for sid in stand_ids])
    results["timings"]["feature_extraction"] = time.time() - t3

    logger.info(f"Feature matrix: {X.shape}, labels: {np.unique(y, return_counts=True)}")

    # ---- 6. PCA (optional) ----
    pca_n = cfg["features"].get("pca_components", 0)
    pca_model = None
    if pca_n > 0:
        pca_n = select_n_components_by_variance(X, target_variance=0.99)
        # PCA will be applied per fold to avoid leakage

    # ---- 7. Run classifiers with CV ----
    t4 = time.time()

    # Build metadata DataFrame for splits
    split_meta = matched.set_index("stand_id").loc[stand_ids].reset_index()
    cv_strategy = cfg["validation"]["primary_cv"]
    splits = get_cv_splitter(cv_strategy, split_meta)

    random_state = cfg.get("random_state", 42)
    model_configs = cfg["models"]["classifiers"]
    all_model_results = {}

    for model_cfg in model_configs:
        model_name = model_cfg["name"]
        model_params = model_cfg.get("params", {})
        logger.info(f"\n--- Running {model_name} ---")

        fold_results = []
        for fold_i, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Normalize (fit on train only)
            norm_method = cfg["preprocessing"]["normalization"]
            X_train_n, X_test_n, _ = apply_normalization(
                X_train, X_test, method=norm_method,
                wavelengths=filtered_wavelengths,
            )

            # Optional PCA
            if pca_n > 0:
                X_train_n, pca_fold = fit_pca(X_train_n, n_components=pca_n)
                X_test_n = transform_pca(X_test_n, pca_fold)

            # Build and run model
            model = build_classifier(model_name, model_params, random_state)
            y_pred, y_proba = train_and_predict(model, X_train_n, y_train, X_test_n)

            # Compute metrics
            class_names = sorted(np.unique(y).tolist())
            metrics = compute_metrics(y_test, y_pred, class_names)
            metrics["fold"] = fold_i
            metrics["test_stand"] = stand_ids[test_idx[0]] if len(test_idx) == 1 else "multi"
            fold_results.append(metrics)

        # Aggregate
        agg = aggregate_fold_metrics(fold_results)
        summary = print_summary(agg, model_name)

        # Check for shortcut patterns
        warnings = check_shortcut_patterns(fold_results)

        all_model_results[model_name] = {
            "fold_results": fold_results,
            "aggregated": agg,
            "warnings": warnings,
        }

    results["model_results"] = all_model_results
    results["timings"]["classification"] = time.time() - t4
    results["timings"]["total"] = time.time() - t0

    # ---- 8. Save outputs ----
    output_dir = Path(cfg["output"]["dir"]) / cfg["output"]["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-model reports
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    for model_name, mresults in all_model_results.items():
        # Per-fold CSV
        build_fold_report(
            mresults["fold_results"],
            output_dir=metrics_dir,
            experiment_name=f"{model_name}_baseline",
        )

        # Confusion matrix
        cm = np.array(mresults["aggregated"]["total_confusion_matrix"])
        class_names = sorted(np.unique(y).tolist())
        plot_confusion_matrix(
            cm, class_names,
            title=f"{model_name} — Confusion Matrix (aggregated)",
            output_path=figures_dir / f"{model_name}_confusion_matrix.png",
        )

        # Per-fold accuracy
        plot_per_fold_accuracy(
            mresults["fold_results"],
            title=f"{model_name} — Per-Fold Balanced Accuracy",
            output_path=figures_dir / f"{model_name}_per_fold_accuracy.png",
        )

    # Mean spectra per class
    plot_mean_spectra_per_class(
        X, y, filtered_wavelengths,
        class_names=sorted(np.unique(y).tolist()),
        output_path=figures_dir / "mean_spectra_per_class.png",
    )

    # Band importance (from Random Forest if available)
    if "RandomForest" in all_model_results:
        rf_model = build_classifier("RandomForest", {"n_estimators": 500}, random_state)
        norm_method = cfg["preprocessing"]["normalization"]
        X_norm, _, _ = apply_normalization(X, None, method=norm_method, wavelengths=filtered_wavelengths)
        rf_model.fit(X_norm, y)
        plot_band_importance(
            rf_model.feature_importances_,
            filtered_wavelengths,
            output_path=figures_dir / "rf_band_importance.png",
        )

    # Save summary JSON
    summary_path = output_dir / "experiment_summary.json"
    summary_data = {
        "n_tiles": results["n_tiles_loaded"],
        "n_matched": results["n_matched_stands"],
        "n_bands_original": results["n_bands_original"],
        "n_bands_filtered": results["n_bands_after_filter"],
        "class_distribution": results["class_distribution"],
        "cv_strategy": cv_strategy,
        "n_folds": len(splits),
        "timings": results["timings"],
        "models": {},
    }
    for mname, mres in all_model_results.items():
        summary_data["models"][mname] = mres["aggregated"]
        summary_data["models"][mname]["warnings"] = mres["warnings"]

    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2, default=str)
    logger.info(f"Saved experiment summary: {summary_path}")

    logger.info(f"\nTotal runtime: {results['timings']['total']:.1f}s")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run hyperspectral forest classification experiment"
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yaml",
        help="Path to YAML config file"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = run_experiment(cfg)

    # Print final summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    for model_name, mres in results["model_results"].items():
        print_summary(mres["aggregated"], model_name)
        if mres["warnings"]:
            for w in mres["warnings"]:
                print(f"  ⚠ {w}")
    print("=" * 60)


if __name__ == "__main__":
    main()
