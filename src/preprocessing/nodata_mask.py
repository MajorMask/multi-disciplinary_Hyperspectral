"""
Nodata and quality masking utilities.

Handles:
- Zero-value pixel masking (common ENVI nodata)
- NaN/Inf detection
- Shadow pixel detection (low brightness across all bands)
- Edge pixel filtering
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def build_combined_mask(
    image: np.ndarray,
    nodata_value: float = 0,
    brightness_threshold: Optional[float] = 0.01,
) -> np.ndarray:
    """
    Build a combined validity mask for an image cube.

    Parameters
    ----------
    image : (bands, rows, cols)
    nodata_value : pixel value marking missing data
    brightness_threshold : if set, mask pixels with mean reflectance
        below this value (shadow removal)

    Returns
    -------
    mask : (rows, cols), True = valid pixel
    """
    n_bands, n_rows, n_cols = image.shape

    # Start with all valid
    mask = np.ones((n_rows, n_cols), dtype=bool)

    # Remove nodata
    if nodata_value is not None:
        mask &= ~np.any(image == nodata_value, axis=0)

    # Remove non-finite
    mask &= np.all(np.isfinite(image), axis=0)

    # Remove shadow pixels
    if brightness_threshold is not None:
        mean_brightness = np.mean(image, axis=0)
        mask &= mean_brightness > brightness_threshold

    n_invalid = (~mask).sum()
    total = n_rows * n_cols
    logger.debug(
        f"Mask: {mask.sum()}/{total} valid "
        f"({n_invalid} removed: nodata/nan/shadow)"
    )
    return mask


def erode_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Erode mask to remove edge pixels that may be mixed.

    Uses binary erosion to shrink the valid region by `iterations` pixels.
    """
    from scipy.ndimage import binary_erosion

    eroded = binary_erosion(mask, iterations=iterations)
    n_removed = mask.sum() - eroded.sum()
    logger.debug(f"Erosion removed {n_removed} edge pixels")
    return eroded.astype(bool)
