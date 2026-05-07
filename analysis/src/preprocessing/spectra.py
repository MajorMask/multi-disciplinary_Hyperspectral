from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DEFAULT_NOISY_REGIONS = [
    (1330, 1550),
    (1761, 2025),
    (2310, 2501),
]


@dataclass
class PreprocessingConfig:
    remove_noisy_bands: bool = True
    noisy_regions: List[Tuple[int, int]] = DEFAULT_NOISY_REGIONS
    normalization: str = "standard"
    valid_pixel_threshold: float = 0.5
    patch_size: int = 9
    patch_stride: int = 9
    summary_method: str = "mean"


def filter_invalid_pixels(image: np.ndarray, nodata_value: float = 10000.0) -> np.ndarray:
    if image.ndim == 2:
        image = image[np.newaxis, ...]
    mask = image == nodata_value
    if np.any(mask):
        image = image.astype(np.float32)
        image[mask] = np.nan
    return image


def band_mask_for_wavelengths(wavelengths: np.ndarray,
                               noisy_regions: List[Tuple[int, int]] = DEFAULT_NOISY_REGIONS) -> np.ndarray:
    if wavelengths.size == 0:
        return np.ones(0, dtype=bool)
    mask = np.ones_like(wavelengths, dtype=bool)
    for start, end in noisy_regions:
        mask &= ~((wavelengths >= start) & (wavelengths < end))
    return mask


def remove_noisy_bands(image: np.ndarray,
                       wavelengths: np.ndarray,
                       noisy_regions: List[Tuple[int, int]] = DEFAULT_NOISY_REGIONS) -> Tuple[np.ndarray, np.ndarray]:
    if wavelengths.size == 0:
        return image, wavelengths
    mask = band_mask_for_wavelengths(wavelengths, noisy_regions)
    if image.ndim == 3:
        return image[mask], wavelengths[mask]
    return image[mask, ...], wavelengths[mask]


def normalize_spectra(X: np.ndarray, method: str = "standard") -> np.ndarray:
    if method == "standard":
        return StandardScaler().fit_transform(X)
    if method == "minmax":
        X_min = np.nanmin(X, axis=0)
        X_max = np.nanmax(X, axis=0)
        denom = np.where(X_max - X_min == 0, 1.0, X_max - X_min)
        return (X - X_min) / denom
    if method == "none":
        return X.astype(np.float32)
    raise ValueError(f"Unsupported normalization method: {method}")


def aggregate_plot_spectra(image: np.ndarray,
                           summary: str = "mean",
                           nodata_value: Optional[float] = None) -> np.ndarray:
    if image.ndim == 2:
        image = image[np.newaxis, ...]
    if nodata_value is not None:
        image = filter_invalid_pixels(image, nodata_value=nodata_value)
    flat = image.reshape(image.shape[0], -1)
    if summary == "mean":
        return np.nanmean(flat, axis=1)
    if summary == "median":
        return np.nanmedian(flat, axis=1)
    if summary == "std":
        return np.nanstd(flat, axis=1)
    raise ValueError(f"Unsupported aggregation method: {summary}")


def compute_patch_tensors(image: np.ndarray,
                          patch_size: int = 9,
                          stride: int = 9,
                          flatten: bool = False) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError("Patch extraction requires image array with shape (bands, rows, cols).")
    bands, rows, cols = image.shape
    patches: list[np.ndarray] = []
    for row in range(0, rows - patch_size + 1, stride):
        for col in range(0, cols - patch_size + 1, stride):
            patch = image[:, row:row + patch_size, col:col + patch_size]
            patches.append(patch)
    if not patches:
        return np.empty((0, bands, patch_size, patch_size), dtype=np.float32)
    patches_array = np.stack(patches, axis=0)
    return patches_array if not flatten else patches_array.reshape(patches_array.shape[0], -1)


def first_derivative(X: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.diff(X, n=1, axis=axis)


def pca_transform(X: np.ndarray, n_components: int = 20) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError("PCA transform expects a 2D array of shape (n_samples, n_features).")
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(np.nan_to_num(X, nan=0.0))


def tile_texture_features(image: np.ndarray) -> Dict[str, np.ndarray]:
    if image.ndim != 3:
        raise ValueError("Texture extraction expects image array with shape (bands, rows, cols).")
    flat = image.reshape(image.shape[0], -1)
    valid = np.isfinite(flat)
    means = np.nanmean(flat, axis=1)
    stds = np.nanstd(flat, axis=1)
    ranges = np.nanmax(flat, axis=1) - np.nanmin(flat, axis=1)
    return {"mean": means, "std": stds, "range": ranges}
