"""
ALS-derived structural features for fusion with spectral data.

Computes stand-level terrain and canopy structure metrics from
1m ALS DEMs provided in the FREEDLES dataset.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def compute_als_stand_features(
    dem_path: Union[str, Path],
    resolution: float = 1.0,
) -> Dict[str, float]:
    """
    Compute terrain features from a stand's DEM tile.

    Wraps als_loader.extract_stand_terrain_stats.
    """
    from model.src.dataio.als_loader import extract_stand_terrain_stats
    return extract_stand_terrain_stats(dem_path, resolution)


def build_als_feature_matrix(
    dem_dir: Union[str, Path],
    stand_ids: List[str],
    pattern: str = "*.tif",
) -> np.ndarray:
    """
    Build ALS feature matrix aligned with stand_ids.

    Parameters
    ----------
    dem_dir : directory containing DEM tiles
    stand_ids : list of stand IDs (from spectral data)
    pattern : glob pattern for DEM files

    Returns
    -------
    X_als : (n_stands, n_als_features) — NaN for missing DEMs
    """
    dem_dir = Path(dem_dir)
    dem_files = {f.stem: f for f in dem_dir.glob(pattern)}

    feature_names = [
        "elev_mean", "elev_std", "elev_range",
        "slope_mean", "slope_std", "roughness_mean",
    ]

    X_als = np.full((len(stand_ids), len(feature_names)), np.nan)

    for i, sid in enumerate(stand_ids):
        if sid not in dem_files:
            logger.warning(f"No DEM for stand {sid}")
            continue

        try:
            stats = compute_als_stand_features(dem_files[sid])
            for j, fname in enumerate(feature_names):
                X_als[i, j] = stats.get(fname, np.nan)
        except Exception as e:
            logger.warning(f"Failed to compute ALS features for {sid}: {e}")

    n_valid = np.sum(~np.isnan(X_als[:, 0]))
    logger.info(
        f"ALS features: {n_valid}/{len(stand_ids)} stands have valid DEMs"
    )
    return X_als
