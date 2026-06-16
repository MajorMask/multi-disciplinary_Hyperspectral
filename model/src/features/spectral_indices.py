"""
Spectral vegetation indices computed from hyperspectral data.

These indices serve as interpretable features and can augment
the full-spectrum feature vector. All functions operate on
stand-summary (mean) spectra or pixel-level arrays.
"""

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _find_nearest_band(wavelengths: np.ndarray, target_nm: float) -> int:
    """Return index of wavelength nearest to target."""
    return int(np.argmin(np.abs(wavelengths - target_nm)))


def ndvi(spectra: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Vegetation Index.
    NDVI = (NIR - Red) / (NIR + Red)
    NIR ~ 800 nm, Red ~ 670 nm
    """
    i_red = _find_nearest_band(wavelengths, 670)
    i_nir = _find_nearest_band(wavelengths, 800)
    red = spectra[..., i_red]
    nir = spectra[..., i_nir]
    denom = nir + red
    denom[denom == 0] = np.nan
    return (nir - red) / denom


def ndre(spectra: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Red Edge Index.
    NDRE = (NIR - RedEdge) / (NIR + RedEdge)
    NIR ~ 800 nm, RedEdge ~ 720 nm
    """
    i_re = _find_nearest_band(wavelengths, 720)
    i_nir = _find_nearest_band(wavelengths, 800)
    re = spectra[..., i_re]
    nir = spectra[..., i_nir]
    denom = nir + re
    denom[denom == 0] = np.nan
    return (nir - re) / denom


def pri(spectra: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """
    Photochemical Reflectance Index.
    PRI = (R531 - R570) / (R531 + R570)
    Sensitive to xanthophyll cycle, light use efficiency.
    """
    i_531 = _find_nearest_band(wavelengths, 531)
    i_570 = _find_nearest_band(wavelengths, 570)
    r531 = spectra[..., i_531]
    r570 = spectra[..., i_570]
    denom = r531 + r570
    denom[denom == 0] = np.nan
    return (r531 - r570) / denom


def evi(spectra: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """
    Enhanced Vegetation Index.
    EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
    """
    i_blue = _find_nearest_band(wavelengths, 470)
    i_red = _find_nearest_band(wavelengths, 670)
    i_nir = _find_nearest_band(wavelengths, 800)
    blue = spectra[..., i_blue]
    red = spectra[..., i_red]
    nir = spectra[..., i_nir]
    denom = nir + 6 * red - 7.5 * blue + 1
    denom[denom == 0] = np.nan
    return 2.5 * (nir - red) / denom


def red_edge_position(spectra: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """
    Red Edge Position: wavelength of maximum first derivative
    in the 680–750 nm region.
    """
    mask = (wavelengths >= 680) & (wavelengths <= 750)
    re_wl = wavelengths[mask]
    re_spectra = spectra[..., mask]

    # First derivative
    deriv = np.gradient(re_spectra, re_wl, axis=-1)
    max_idx = np.argmax(deriv, axis=-1)

    # Map indices to wavelengths
    if spectra.ndim == 1:
        return re_wl[max_idx]
    else:
        return re_wl[max_idx]


def compute_all_indices(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute all vegetation indices.

    Parameters
    ----------
    spectra : (..., n_bands) — any shape with bands as last dim
    wavelengths : (n_bands,)

    Returns
    -------
    dict of index_name → array (same shape as spectra minus last dim)
    """
    indices = {}
    for name, func in [
        ("ndvi", ndvi),
        ("ndre", ndre),
        ("pri", pri),
        ("evi", evi),
        ("rep", red_edge_position),
    ]:
        try:
            indices[name] = func(spectra, wavelengths)
        except Exception as e:
            logger.warning(f"Failed to compute {name}: {e}")
            indices[name] = np.full(spectra.shape[:-1], np.nan)

    return indices


def indices_to_feature_vector(indices: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Stack all index values into a feature vector.

    For stand-level (scalar indices): returns (n_indices,)
    For pixel-level: returns (n_pixels, n_indices)
    """
    arrays = [v for v in indices.values()]
    if arrays[0].ndim == 0:
        return np.array([float(a) for a in arrays])
    else:
        return np.column_stack(arrays)
