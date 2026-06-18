"""
Cross-validation split generators.

Implements:
- Leave-One-Stand-Out (LOSO) for within-site validation
- Stratified GroupKFold as alternative
- Leave-One-Site-Out (multi-site, for future use)

Follows Roberts et al. (2017, Ecography) spatial blocking recommendations
and Ploton et al. (2020, Nature Comms) cautions on autocorrelation.
"""

import logging
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    LeaveOneGroupOut,
    StratifiedKFold,
)

logger = logging.getLogger(__name__)


def leave_one_stand_out(
    metadata: pd.DataFrame,
    stand_id_col: str = "stand_id",
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Leave-One-Stand-Out cross-validation.

    Each fold holds out one stand for testing. This is the strictest
    within-site validation and prevents spatial data leakage between
    stands (Roberts et al. 2017).

    Yields (train_indices, test_indices) pairs.
    """
    logo = LeaveOneGroupOut()
    groups = metadata[stand_id_col].values if stand_id_col in metadata.columns \
        else metadata.index.values
    y_dummy = np.zeros(len(metadata))  # not used by LOGO

    n_splits = logo.get_n_splits(groups=groups)
    logger.info(f"Leave-One-Stand-Out: {n_splits} folds")

    for train_idx, test_idx in logo.split(y_dummy, groups=groups):
        yield train_idx, test_idx


def leave_one_site_out(
    metadata: pd.DataFrame,
    site_col: str = "site",
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Leave-One-Site-Out cross-validation for multi-site experiments.

    NOTE: Requires data from multiple sites. With only Hyytiälä,
    this degenerates to a single train/test split with an empty
    test set. Use leave_one_stand_out for single-site work.
    """
    logo = LeaveOneGroupOut()
    groups = metadata[site_col].values
    y_dummy = np.zeros(len(metadata))

    unique_sites = np.unique(groups)
    if len(unique_sites) < 2:
        raise ValueError(
            f"Leave-one-site-out requires ≥2 sites, found: {unique_sites}. "
            "Use leave_one_stand_out for single-site validation."
        )

    logger.info(f"Leave-One-Site-Out: {len(unique_sites)} folds")
    for train_idx, test_idx in logo.split(y_dummy, groups=groups):
        yield train_idx, test_idx


def stratified_group_kfold(
    metadata: pd.DataFrame,
    label_col: str = "forest_type",
    group_col: str = "stand_id",
    n_splits: int = 5,
    random_state: int = 42,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Stratified GroupKFold — groups (stands) are not split across folds,
    and class proportions are approximately maintained.

    Falls back to GroupKFold if sklearn version lacks StratifiedGroupKFold.
    """
    groups = metadata[group_col].values if group_col in metadata.columns \
        else metadata.index.values
    y = metadata[label_col].values

    try:
        from sklearn.model_selection import StratifiedGroupKFold as SGK
        cv = SGK(n_splits=n_splits, shuffle=True, random_state=random_state)
        logger.info(f"StratifiedGroupKFold: {n_splits} folds")
        for train_idx, test_idx in cv.split(np.zeros(len(y)), y, groups):
            yield train_idx, test_idx
    except ImportError:
        logger.warning(
            "StratifiedGroupKFold not available. "
            "Falling back to GroupKFold (no stratification)."
        )
        gkf = GroupKFold(n_splits=n_splits)
        for train_idx, test_idx in gkf.split(np.zeros(len(y)), y, groups):
            yield train_idx, test_idx


def get_cv_splitter(
    strategy: str,
    metadata: pd.DataFrame,
    **kwargs,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Factory function to get CV splits based on strategy name.

    Parameters
    ----------
    strategy : one of "logo_stand", "loso_site", "stratified_kfold"
    metadata : DataFrame with stand-level info
    **kwargs : passed to the underlying splitter

    Returns
    -------
    List of (train_idx, test_idx) tuples.
    """
    if strategy == "logo_stand":
        splits = list(leave_one_stand_out(metadata, **kwargs))
    elif strategy == "loso_site":
        splits = list(leave_one_site_out(metadata, **kwargs))
    elif strategy == "stratified_kfold":
        splits = list(stratified_group_kfold(metadata, **kwargs))
    else:
        raise ValueError(f"Unknown CV strategy: '{strategy}'")

    # Log class distribution per fold
    label_col = kwargs.get("label_col", "forest_type")
    if label_col in metadata.columns:
        labels = metadata[label_col].values
        for i, (tr, te) in enumerate(splits):
            train_dist = pd.Series(labels[tr]).value_counts().to_dict()
            test_dist = pd.Series(labels[te]).value_counts().to_dict()
            logger.debug(
                f"Fold {i}: train={train_dist}, test={test_dist}"
            )

    logger.info(f"CV strategy '{strategy}': {len(splits)} folds")
    return splits
