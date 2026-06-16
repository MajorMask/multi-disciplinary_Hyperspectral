"""
Stand-level spectral summary features.

Aggregates pixel-level spectra within each stand tile to a single
feature vector (mean, median, std, percentiles) suitable for
classical ML classifiers.

This is the primary feature extraction for Experiment 1 (baseline).
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_stand_summary(
    spectra: np.ndarray,
    method: str = "mean",
) -> np.ndarray:
    """
    Compute stand-level summary from pixel spectra.

    Parameters
    ----------
    spectra : (n_pixels, n_bands) — valid pixels only
    method : "mean", "median", "both", or "full"
        - "mean": mean spectrum (n_bands,)
        - "median": median spectrum (n_bands,)
        - "both": concatenated mean + median (2*n_bands,)
        - "full": mean + median + std + p10 + p90 (5*n_bands,)

    Returns
    -------
    1D feature vector.
    """
    if spectra.shape[0] == 0:
        logger.warning("Empty spectra array, returning zeros")
        n_bands = spectra.shape[1] if spectra.ndim == 2 else 0
        multiplier = {"mean": 1, "median": 1, "both": 2, "full": 5}.get(method, 1)
        return np.zeros(n_bands * multiplier)

    if method == "mean":
        return np.mean(spectra, axis=0)
    elif method == "median":
        return np.median(spectra, axis=0)
    elif method == "both":
        return np.concatenate([
            np.mean(spectra, axis=0),
            np.median(spectra, axis=0),
        ])
    elif method == "full":
        return np.concatenate([
            np.mean(spectra, axis=0),
            np.median(spectra, axis=0),
            np.std(spectra, axis=0),
            np.percentile(spectra, 10, axis=0),
            np.percentile(spectra, 90, axis=0),
        ])
    else:
        raise ValueError(f"Unknown summary method: '{method}'")


def build_stand_feature_matrix(
    tiles: list,
    method: str = "mean",
    band_mask: Optional[np.ndarray] = None,
) -> tuple:
    """
    Build a feature matrix from a list of RasterTile objects.

    Parameters
    ----------
    tiles : list of RasterTile instances
    method : summary statistic method
    band_mask : optional boolean mask for band selection (applied first)

    Returns
    -------
    (X, stand_ids)
        X : (n_stands, n_features) array
        stand_ids : list of stand ID strings
    """
    features = []
    stand_ids = []

    for tile in tiles:
        spectra = tile.get_valid_spectra()  # (n_valid, n_bands)

        if band_mask is not None:
            spectra = spectra[:, band_mask]

        if spectra.shape[0] < 10:
            logger.warning(
                f"Stand {tile.stand_id}: only {spectra.shape[0]} valid pixels, skipping"
            )
            continue

        feat = compute_stand_summary(spectra, method=method)
        features.append(feat)
        stand_ids.append(tile.stand_id)

    X = np.array(features)
    logger.info(
        f"Built feature matrix: {X.shape[0]} stands × {X.shape[1]} features "
        f"(method: {method})"
    )
    return X, stand_ids
