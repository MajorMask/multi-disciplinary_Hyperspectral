"""
Metadata loader — links stand IDs to forest type, dominant species, site, and coordinates.

Expected metadata CSV columns (configurable via config):
    stand_id, site, forest_type, dominant_species, latitude, longitude

If no metadata CSV exists on disk, the user can create one manually from
the FREEDLES companion data or the ESSD Table 3 (Hovi et al. 2024).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


REQUIRED_COLS = ["stand_id", "forest_type"]


def load_metadata(
    filepath: Union[str, Path],
    stand_id_col: str = "stand_id",
    forest_type_col: str = "forest_type",
    rename_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Load stand-level metadata from CSV or Excel.

    Parameters
    ----------
    filepath : path to .csv or .xlsx
    stand_id_col : column name for stand identifier
    forest_type_col : column name for target label
    rename_map : optional mapping {original_col: canonical_col}

    Returns
    -------
    DataFrame indexed by stand_id with at least a 'forest_type' column.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Metadata file not found: {filepath}")

    if filepath.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    else:
        # Try common delimiters
        for sep in [",", ";", "\t"]:
            df = pd.read_csv(filepath, sep=sep)
            if len(df.columns) > 1:
                break

    # Apply rename map if provided
    if rename_map:
        df = df.rename(columns=rename_map)
    else:
        # Build rename from config-style column names
        if stand_id_col != "stand_id" and stand_id_col in df.columns:
            df = df.rename(columns={stand_id_col: "stand_id"})
        if forest_type_col != "forest_type" and forest_type_col in df.columns:
            df = df.rename(columns={forest_type_col: "forest_type"})

    # Validate required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Metadata missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    # Clean stand_id: strip whitespace, convert to string
    df["stand_id"] = df["stand_id"].astype(str).str.strip()

    # Clean forest_type: lowercase, strip
    df["forest_type"] = df["forest_type"].astype(str).str.strip().str.lower()

    # Validate forest types
    valid_types = {"coniferous", "broadleaved", "mixed"}
    unique_types = set(df["forest_type"].unique())
    unknown = unique_types - valid_types
    if unknown:
        logger.warning(
            f"Unknown forest types: {unknown}. "
            f"Expected: {valid_types}"
        )

    # Drop duplicates on stand_id
    n_before = len(df)
    df = df.drop_duplicates(subset="stand_id", keep="first")
    if len(df) < n_before:
        logger.warning(
            f"Dropped {n_before - len(df)} duplicate stand_id rows"
        )

    df = df.set_index("stand_id")
    logger.info(
        f"Loaded metadata: {len(df)} stands, "
        f"classes: {df['forest_type'].value_counts().to_dict()}"
    )
    return df


def merge_tiles_with_metadata(
    tile_stand_ids: List[str],
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Match discovered tile stand IDs against metadata.

    Returns a DataFrame with columns [stand_id, forest_type, ...] for
    all tiles that have matching metadata entries.
    """
    matched = metadata.loc[metadata.index.isin(tile_stand_ids)]
    unmatched = set(tile_stand_ids) - set(matched.index)

    if unmatched:
        logger.warning(
            f"{len(unmatched)} tiles have no metadata: "
            f"{sorted(list(unmatched))[:10]}..."
        )

    orphan = set(metadata.index) - set(tile_stand_ids)
    if orphan:
        logger.info(
            f"{len(orphan)} metadata entries have no matching tile"
        )

    logger.info(
        f"Matched {len(matched)} / {len(tile_stand_ids)} tiles to metadata"
    )
    return matched.reset_index()


def create_template_metadata(
    stand_ids: List[str],
    output_path: Union[str, Path],
) -> Path:
    """
    Create a template metadata CSV for manual annotation.

    Writes a CSV with stand_id pre-filled and empty columns for the user
    to fill in from the FREEDLES companion data.
    """
    output_path = Path(output_path)
    df = pd.DataFrame({
        "stand_id": stand_ids,
        "site": "Hyytiala",
        "forest_type": "",
        "dominant_species": "",
        "latitude": np.nan,
        "longitude": np.nan,
    })
    df.to_csv(output_path, index=False)
    logger.info(
        f"Created metadata template with {len(stand_ids)} stands: {output_path}"
    )
    return output_path
