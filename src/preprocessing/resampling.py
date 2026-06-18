"""
Spectral resampling utilities.

Used when merging CASI (0.5m) and SASI (1.25m) data to a common
spatial and spectral resolution.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


def resample_spectra(
    spectra: np.ndarray,
    source_wavelengths: np.ndarray,
    target_wavelengths: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """
    Resample spectra from source wavelength grid to target grid.

    Parameters
    ----------
    spectra : (n_samples, n_source_bands)
    source_wavelengths : (n_source_bands,)
    target_wavelengths : (n_target_bands,)
    method : interpolation method ("linear", "cubic")

    Returns
    -------
    resampled : (n_samples, n_target_bands)
    """
    # Clip target to source range to avoid extrapolation
    valid = (target_wavelengths >= source_wavelengths.min()) & \
            (target_wavelengths <= source_wavelengths.max())

    if not valid.all():
        n_clipped = (~valid).sum()
        logger.warning(
            f"Clipping {n_clipped} target bands outside source range "
            f"[{source_wavelengths.min():.0f}, {source_wavelengths.max():.0f}] nm"
        )

    f = interp1d(
        source_wavelengths, spectra, axis=1,
        kind=method, bounds_error=False, fill_value=np.nan,
    )
    resampled = f(target_wavelengths)

    logger.info(
        f"Resampled {spectra.shape[1]} → {resampled.shape[1]} bands "
        f"({method} interpolation)"
    )
    return resampled


def merge_casi_sasi(
    casi_spectra: np.ndarray,
    casi_wavelengths: np.ndarray,
    sasi_spectra: np.ndarray,
    sasi_wavelengths: np.ndarray,
    overlap_range: Tuple[float, float] = (958, 1052),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge CASI and SASI spectra, handling overlap region.

    In the overlap region (958–1052 nm), takes the average of both sensors.
    CASI: 382–1052 nm, SASI: 958–2443 nm.

    Parameters
    ----------
    casi_spectra : (n_samples, n_casi_bands)
    casi_wavelengths : (n_casi_bands,)
    sasi_spectra : (n_samples, n_sasi_bands)
    sasi_wavelengths : (n_sasi_bands,)
    overlap_range : (min_nm, max_nm) of overlap

    Returns
    -------
    (merged_spectra, merged_wavelengths)
    """
    ol_lo, ol_hi = overlap_range

    # CASI-only bands (below overlap)
    casi_only_mask = casi_wavelengths < ol_lo
    # SASI-only bands (above overlap)
    sasi_only_mask = sasi_wavelengths > ol_hi

    # Overlap bands: use CASI wavelength grid, average both sensors
    casi_overlap_mask = (casi_wavelengths >= ol_lo) & (casi_wavelengths <= ol_hi)
    overlap_wl = casi_wavelengths[casi_overlap_mask]

    # Resample SASI to CASI grid in overlap region
    sasi_resampled = resample_spectra(
        sasi_spectra, sasi_wavelengths, overlap_wl
    )
    casi_overlap = casi_spectra[:, casi_overlap_mask]

    # Average where both are valid
    overlap_merged = np.nanmean(
        np.stack([casi_overlap, sasi_resampled], axis=-1), axis=-1
    )

    # Concatenate: CASI-only | overlap-averaged | SASI-only
    merged_spectra = np.concatenate([
        casi_spectra[:, casi_only_mask],
        overlap_merged,
        sasi_spectra[:, sasi_only_mask],
    ], axis=1)

    merged_wavelengths = np.concatenate([
        casi_wavelengths[casi_only_mask],
        overlap_wl,
        sasi_wavelengths[sasi_only_mask],
    ])

    logger.info(
        f"Merged CASI ({casi_wavelengths.shape[0]} bands) + "
        f"SASI ({sasi_wavelengths.shape[0]} bands) → "
        f"{merged_wavelengths.shape[0]} bands "
        f"({merged_wavelengths.min():.0f}–{merged_wavelengths.max():.0f} nm)"
    )
    return merged_spectra, merged_wavelengths
