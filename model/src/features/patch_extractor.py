"""
Spatial patch extraction for CNN-based classification.

Extracts fixed-size spatial patches from stand tiles for:
- 1D CNN: center-pixel spectra with spatial context averaging
- 3D CNN: full spatial-spectral cubes (patch_size × patch_size × n_bands)

Patches are extracted only from valid-pixel regions.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def extract_patches(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 15,
    stride: int = 15,
    min_valid_fraction: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract spatial patches from a single image tile.

    Parameters
    ----------
    image : (bands, rows, cols)
    mask : (rows, cols), True = valid pixel
    patch_size : height/width of square patches
    stride : step between patches
    min_valid_fraction : minimum fraction of valid pixels in a patch

    Returns
    -------
    patches : (n_patches, bands, patch_size, patch_size)
    centers : (n_patches, 2) — row, col of each patch center
    """
    n_bands, n_rows, n_cols = image.shape
    half = patch_size // 2

    patches = []
    centers = []

    for r in range(half, n_rows - half, stride):
        for c in range(half, n_cols - half, stride):
            r0, r1 = r - half, r + half + 1
            c0, c1 = c - half, c + half + 1

            # Check if enough valid pixels in this patch
            patch_mask = mask[r0:r1, c0:c1]
            valid_frac = patch_mask.mean()
            if valid_frac < min_valid_fraction:
                continue

            patch = image[:, r0:r1, c0:c1].copy()
            # Zero-out invalid pixels within patch
            patch[:, ~patch_mask] = 0

            patches.append(patch)
            centers.append([r, c])

    if not patches:
        return np.empty((0, n_bands, patch_size, patch_size)), np.empty((0, 2))

    patches = np.array(patches)
    centers = np.array(centers)

    logger.debug(
        f"Extracted {len(patches)} patches ({patch_size}×{patch_size}) "
        f"with stride {stride}"
    )
    return patches, centers


def extract_center_spectra(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 15,
    stride: int = 15,
    context_agg: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract center-pixel spectra with optional spatial context.

    For 1D CNN: use the patch mean as a spatially-smoothed spectrum.

    Parameters
    ----------
    context_agg : "center" (pixel only), "mean" (patch mean), "median"

    Returns
    -------
    spectra : (n_patches, n_bands)
    centers : (n_patches, 2)
    """
    patches, centers = extract_patches(image, mask, patch_size, stride)

    if patches.shape[0] == 0:
        return np.empty((0, image.shape[0])), np.empty((0, 2))

    if context_agg == "center":
        half = patch_size // 2
        spectra = patches[:, :, half, half]
    elif context_agg == "mean":
        # Mean across spatial dims
        spectra = patches.mean(axis=(2, 3))
    elif context_agg == "median":
        spectra = np.median(patches, axis=(2, 3))
    else:
        raise ValueError(f"Unknown context_agg: '{context_agg}'")

    return spectra, centers


def patches_from_tiles(
    tiles: list,
    patch_size: int = 15,
    stride: int = 15,
    min_valid_fraction: float = 0.8,
    band_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Extract patches from multiple RasterTile objects.

    Returns
    -------
    all_patches : (N, bands, patch_size, patch_size)
    stand_ids : list of stand_id for each patch (for GroupKFold)
    all_centers : (N, 2)
    """
    all_patches = []
    all_stand_ids = []
    all_centers = []

    for tile in tiles:
        img = tile.image
        if band_mask is not None:
            img = img[band_mask]

        patches, centers = extract_patches(
            img, tile.nodata_mask, patch_size, stride, min_valid_fraction
        )
        all_patches.append(patches)
        all_stand_ids.extend([tile.stand_id] * len(patches))
        all_centers.append(centers)

    if not all_patches:
        return np.empty((0,)), [], np.empty((0, 2))

    all_patches = np.concatenate(all_patches, axis=0)
    all_centers = np.concatenate(all_centers, axis=0)

    logger.info(
        f"Total patches: {all_patches.shape[0]} from {len(tiles)} tiles"
    )
    return all_patches, all_stand_ids, all_centers
