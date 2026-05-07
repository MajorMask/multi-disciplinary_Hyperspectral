from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class FeatureConfig:
    feature_mode: str = "mean"
    pca_components: Optional[int] = None
    include_texture: bool = False
    texture_methods: List[str] = None

    def __post_init__(self) -> None:
        if self.texture_methods is None:
            self.texture_methods = ["mean", "std", "range"]


def build_feature_matrix(
    summary_dataframe: Any,
    summary_method: str = "mean",
    pca_components: Optional[int] = None,
) -> np.ndarray:
    band_columns = [c for c in summary_dataframe.columns if c.startswith("band_")]
    X = summary_dataframe[band_columns].astype(float).values
    if pca_components is not None and pca_components > 0:
        return PCA(n_components=pca_components, random_state=42).fit_transform(X)
    return X


def add_texture_features(
    summary_dataframe: Any,
    methods: List[str] = ["mean", "std", "range"],
) -> Any:
    texture_features = {}
    for method in methods:
        if method not in {"mean", "std", "range"}:
            raise ValueError(f"Unsupported texture method: {method}")
        band_columns = [c for c in summary_dataframe.columns if c.startswith("band_")]
        values = summary_dataframe[band_columns].astype(float).values
        if method == "mean":
            texture_features["texture_mean"] = np.nanmean(values, axis=1)
        if method == "std":
            texture_features["texture_std"] = np.nanstd(values, axis=1)
        if method == "range":
            texture_features["texture_range"] = np.nanmax(values, axis=1) - np.nanmin(values, axis=1)
    for key, values in texture_features.items():
        summary_dataframe[key] = values
    return summary_dataframe
