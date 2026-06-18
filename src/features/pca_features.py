"""
PCA-based dimensionality reduction for hyperspectral data.

Addresses the Hughes phenomenon (curse of dimensionality) common in
hyperspectral classification with small sample sizes.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def fit_pca(
    X_train: np.ndarray,
    n_components: int = 10,
    random_state: int = 42,
) -> Tuple[np.ndarray, PCA]:
    """
    Fit PCA on training data and transform.

    Parameters
    ----------
    X_train : (n_samples, n_features)
    n_components : number of components to keep

    Returns
    -------
    (X_train_pca, pca_model)
    """
    n_components = min(n_components, X_train.shape[0], X_train.shape[1])

    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_train)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    logger.info(
        f"PCA: {X_train.shape[1]} → {n_components} components, "
        f"explained variance: {cumvar[-1]:.3f} "
        f"(first 3: {cumvar[:3]})"
    )
    return X_pca, pca


def transform_pca(
    X: np.ndarray,
    pca: PCA,
) -> np.ndarray:
    """Apply a fitted PCA model to new data."""
    return pca.transform(X)


def select_n_components_by_variance(
    X_train: np.ndarray,
    target_variance: float = 0.95,
    random_state: int = 42,
) -> int:
    """
    Determine number of PCA components needed to explain target variance.

    Returns
    -------
    n_components : int
    """
    max_comp = min(X_train.shape[0], X_train.shape[1])
    pca = PCA(n_components=max_comp, random_state=random_state)
    pca.fit(X_train)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n = int(np.searchsorted(cumvar, target_variance) + 1)
    n = min(n, max_comp)

    logger.info(
        f"Components for {target_variance:.0%} variance: {n} "
        f"(actual: {cumvar[n-1]:.3f})"
    )
    return n
