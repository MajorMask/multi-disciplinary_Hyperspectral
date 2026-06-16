"""
Spectral normalization methods.

Supports:
- StandardScaler (per-band zero mean, unit variance)
- SNV (Standard Normal Variate — per-spectrum)
- MinMax (per-band to [0,1])
- Continuum removal
"""

import logging
from typing import Optional, Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


def standard_normalize(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], StandardScaler]:
    """
    Per-band standard normalization. Fit on train, apply to both.

    Parameters
    ----------
    X_train : (n_train, n_bands)
    X_test : (n_test, n_bands) or None

    Returns
    -------
    (X_train_norm, X_test_norm, scaler)
    """
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)

    X_test_norm = None
    if X_test is not None:
        X_test_norm = scaler.transform(X_test)

    logger.debug(
        f"Standard normalization: {X_train.shape[1]} features, "
        f"train mean range [{X_train_norm.mean(axis=0).min():.3f}, "
        f"{X_train_norm.mean(axis=0).max():.3f}]"
    )
    return X_train_norm, X_test_norm, scaler


def snv_normalize(X: np.ndarray) -> np.ndarray:
    """
    Standard Normal Variate: per-spectrum centering and scaling.

    Each row (spectrum) is centered by its mean and divided by its std.
    Corrects for multiplicative scatter effects.

    Parameters
    ----------
    X : (n_samples, n_bands)

    Returns
    -------
    X_snv : same shape
    """
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    stds[stds == 0] = 1  # avoid division by zero for constant spectra
    return (X - means) / stds


def minmax_normalize(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], MinMaxScaler]:
    """
    Per-band MinMax normalization to [0, 1]. Fit on train.
    """
    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test) if X_test is not None else None
    return X_train_norm, X_test_norm, scaler


def continuum_removal(X: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """
    Continuum removal normalization.

    Fits a convex hull upper envelope to each spectrum and divides by it.
    Highlights absorption features. Useful for mineral/vegetation indices.

    Parameters
    ----------
    X : (n_samples, n_bands)
    wavelengths : (n_bands,)

    Returns
    -------
    X_cr : same shape, values in [0, 1]
    """
    from scipy.spatial import ConvexHull

    X_cr = np.ones_like(X)
    for i in range(X.shape[0]):
        spectrum = X[i]
        points = np.column_stack([wavelengths, spectrum])
        try:
            hull = ConvexHull(points)
            # Find upper hull vertices (sorted by wavelength)
            hull_verts = sorted(
                set(hull.vertices),
                key=lambda v: wavelengths[v]
            )
            # Interpolate continuum at all wavelengths
            hull_wl = wavelengths[hull_verts]
            hull_vals = spectrum[hull_verts]
            continuum = np.interp(wavelengths, hull_wl, hull_vals)
            continuum[continuum == 0] = 1  # avoid div by zero
            X_cr[i] = spectrum / continuum
        except Exception:
            # If hull fails (e.g., too few points), return raw
            X_cr[i] = spectrum

    return X_cr


def apply_normalization(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray],
    method: str = "standard",
    wavelengths: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], object]:
    """
    Factory function to apply normalization by method name.

    Parameters
    ----------
    method : "standard", "snv", "minmax", "continuum", or "none"

    Returns
    -------
    (X_train_norm, X_test_norm, scaler_or_None)
    """
    if method == "standard":
        return standard_normalize(X_train, X_test)
    elif method == "snv":
        X_tr = snv_normalize(X_train)
        X_te = snv_normalize(X_test) if X_test is not None else None
        return X_tr, X_te, None
    elif method == "minmax":
        return minmax_normalize(X_train, X_test)
    elif method == "continuum":
        if wavelengths is None:
            raise ValueError("continuum removal requires wavelengths")
        X_tr = continuum_removal(X_train, wavelengths)
        X_te = continuum_removal(X_test, wavelengths) if X_test is not None else None
        return X_tr, X_te, None
    elif method == "none":
        return X_train, X_test, None
    else:
        raise ValueError(f"Unknown normalization method: '{method}'")
