from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover
    gpd = None


def load_metadata(path: Path, geometry_col: Optional[str] = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".geojson", ".json", ".gpkg", ".shp"}:
        if gpd is None:
            raise ImportError("geopandas is required to read vector metadata formats. Install with `pip install geopandas`.")
        gdf = gpd.read_file(path)
        if geometry_col is not None and geometry_col in gdf.columns:
            gdf = gdf.set_geometry(geometry_col)
        return pd.DataFrame(gdf)
    raise ValueError(f"Unsupported metadata format: {suffix}")


def validate_metadata_columns(metadata: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [col for col in required_columns if col not in metadata.columns]
    if missing:
        raise ValueError(f"Metadata is missing required columns: {missing}")


def normalize_column_names(metadata: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    return metadata.rename(columns=mapping)
