"""
Spectral band selection and filtering.

Removes water vapor absorption bands (interpolated by ATCOR-4, not measured)
and optionally applies feature-importance-based band reduction.

Water vapor regions to exclude (from Hovi et al. 2024 / ATCOR-4 docs):
    895–1003 nm, 1092–1168 nm, 1302–1528 nm, 1737–2038 nm
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Default water vapor absorption ranges (nm) interpolated by ATCOR-4
DEFAULT_EXCLUDE_RANGES = [
    (895, 1003),
    (1092, 1168),
    (1302, 1528),
    (1737, 2038),
]


def get_valid_band_mask(
    wavelengths: np.ndarray,
    exclude_ranges: Optional[List[Tuple[float, float]]] = None,
    wl_min: Optional[float] = None,
    wl_max: Optional[float] = None,
) -> np.ndarray:
    """
    Create a boolean mask for bands to retain.

    Parameters
    ----------
    wavelengths : 1D array of band center wavelengths (nm)
    exclude_ranges : list of (min_nm, max_nm) ranges to drop
    wl_min, wl_max : optional wavelength bounds

    Returns
    -------
    Boolean mask, True = keep band.
    """
    if exclude_ranges is None:
        exclude_ranges = DEFAULT_EXCLUDE_RANGES

    mask = np.ones(len(wavelengths), dtype=bool)

    # Exclude water vapor ranges
    for lo, hi in exclude_ranges:
        mask &= ~((wavelengths >= lo) & (wavelengths <= hi))

    # Apply wavelength bounds
    if wl_min is not None:
        mask &= wavelengths >= wl_min
    if wl_max is not None:
        mask &= wavelengths <= wl_max

    n_removed = (~mask).sum()
    logger.info(
        f"Band selection: keeping {mask.sum()}/{len(wavelengths)} bands "
        f"(removed {n_removed} in absorption/out-of-range regions)"
    )
    return mask


def apply_band_selection(
    image: np.ndarray,
    wavelengths: np.ndarray,
    band_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a band mask to an image cube and wavelength array.

    Parameters
    ----------
    image : shape (bands, rows, cols) or (n_samples, bands)
    wavelengths : shape (bands,)
    band_mask : boolean mask, shape (bands,)

    Returns
    -------
    (filtered_image, filtered_wavelengths)
    """
    assert len(wavelengths) == len(band_mask), (
        f"Wavelength/mask length mismatch: {len(wavelengths)} vs {len(band_mask)}"
    )

    if image.ndim == 3:
        # (bands, rows, cols) → index along axis 0
        assert image.shape[0] == len(band_mask)
        return image[band_mask], wavelengths[band_mask]
    elif image.ndim == 2:
        # (samples, bands) → index along axis 1
        assert image.shape[1] == len(band_mask)
        return image[:, band_mask], wavelengths[band_mask]
    else:
        raise ValueError(f"Expected 2D or 3D image, got {image.ndim}D")


def select_bands_by_importance(
    wavelengths: np.ndarray,
    importances: np.ndarray,
    n_bands: int = 20,
) -> np.ndarray:
    """
    Select top-N bands by feature importance (e.g., from Random Forest).

    Parameters
    ----------
    wavelengths : 1D array of wavelengths
    importances : 1D array of importance scores (same length)
    n_bands : number of bands to keep

    Returns
    -------
    Boolean mask, True for selected bands.
    """
    assert len(wavelengths) == len(importances)
    n_bands = min(n_bands, len(wavelengths))

    top_indices = np.argsort(importances)[-n_bands:]
    mask = np.zeros(len(wavelengths), dtype=bool)
    mask[top_indices] = True

    selected_wl = wavelengths[mask]
    logger.info(
        f"Selected top {n_bands} bands by importance: "
        f"{selected_wl.min():.0f}–{selected_wl.max():.0f} nm"
    )
    return mask
