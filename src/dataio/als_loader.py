"""
ALS (Airborne Laser Scanning) DEM loader.

Loads 1m digital elevation models from the FREEDLES ALS products.
RIEGL LMS-Q780, ~48 pulses/m² at Hyytiälä.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
except ImportError:
    rasterio = None


def load_dem(
    filepath: Union[str, Path],
    nodata_value: float = -9999,
) -> dict:
    """
    Load a DEM raster and return dict with array + metadata.

    Returns
    -------
    dict with keys: 'elevation' (2D array), 'transform', 'crs', 'nodata_mask'
    """
    if rasterio is None:
        raise ImportError("rasterio required for DEM loading")

    filepath = Path(filepath)
    with rasterio.open(filepath) as src:
        elev = src.read(1).astype(np.float32)
        transform = src.transform
        crs = str(src.crs) if src.crs else None

    nodata_mask = np.isfinite(elev) & (elev != nodata_value)

    logger.info(
        f"Loaded DEM {filepath.name}: {elev.shape}, "
        f"{nodata_mask.sum()}/{nodata_mask.size} valid pixels"
    )

    return {
        "elevation": elev,
        "nodata_mask": nodata_mask,
        "transform": transform,
        "crs": crs,
        "filepath": filepath,
    }


def compute_terrain_features(elevation: np.ndarray, resolution: float = 1.0) -> dict:
    """
    Derive terrain features from a DEM.

    Parameters
    ----------
    elevation : 2D array of elevation values
    resolution : pixel size in metres

    Returns
    -------
    dict with 'slope', 'aspect', 'roughness' arrays (same shape as input)
    """
    # Gradient in x and y directions
    dy, dx = np.gradient(elevation, resolution)

    # Slope in degrees
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    # Aspect in degrees (0 = north, clockwise)
    aspect = np.degrees(np.arctan2(-dx, dy))
    aspect[aspect < 0] += 360

    # Roughness: std of elevation in a 3x3 window
    from scipy.ndimage import uniform_filter
    mean_elev = uniform_filter(elevation, size=3)
    mean_sq = uniform_filter(elevation**2, size=3)
    roughness = np.sqrt(np.maximum(mean_sq - mean_elev**2, 0))

    return {
        "slope": slope,
        "aspect": aspect,
        "roughness": roughness,
    }


def extract_stand_terrain_stats(
    dem_path: Union[str, Path],
    resolution: float = 1.0,
) -> dict:
    """
    Load DEM and compute stand-level terrain summary statistics.

    Returns dict of scalar features suitable for ML.
    """
    dem_data = load_dem(dem_path)
    elev = dem_data["elevation"]
    mask = dem_data["nodata_mask"]

    terrain = compute_terrain_features(elev, resolution)

    valid_elev = elev[mask]
    valid_slope = terrain["slope"][mask]
    valid_roughness = terrain["roughness"][mask]

    stats = {
        "elev_mean": float(np.mean(valid_elev)),
        "elev_std": float(np.std(valid_elev)),
        "elev_range": float(np.ptp(valid_elev)),
        "slope_mean": float(np.mean(valid_slope)),
        "slope_std": float(np.std(valid_slope)),
        "roughness_mean": float(np.mean(valid_roughness)),
    }

    return stats
