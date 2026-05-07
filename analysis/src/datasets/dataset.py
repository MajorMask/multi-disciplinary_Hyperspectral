from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.dataio.metadata import load_metadata, validate_metadata_columns
from src.dataio.raster import RasterData, read_geotiff
from src.preprocessing.spectra import (
    aggregate_plot_spectra,
    filter_invalid_pixels,
    normalize_spectra,
    remove_noisy_bands,
)


@dataclass
class PlotSummaryRecord:
    plot_id: str
    tile_path: str
    summary_vector: np.ndarray
    product_type: Optional[str]
    metadata: Dict[str, Any]


class PlotTileDataset:
    def __init__(
        self,
        tile_dir: Path,
        metadata_path: Path,
        tile_id_column: str,
        plot_id_column: str,
        site_id_column: str,
        label_column: str,
        product_type: Optional[str] = None,
        use_wavelengths: bool = True,
        noisy_regions: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        self.tile_dir = Path(tile_dir)
        self.metadata_path = Path(metadata_path)
        self.tile_id_column = tile_id_column
        self.plot_id_column = plot_id_column
        self.site_id_column = site_id_column
        self.label_column = label_column
        self.product_type = product_type
        self.use_wavelengths = use_wavelengths
        self.noisy_regions = noisy_regions
        self.metadata = self._load_metadata()
        self.tile_paths = self._find_tile_paths()
        self.summary_df: Optional[pd.DataFrame] = None

    def _load_metadata(self) -> pd.DataFrame:
        metadata = load_metadata(self.metadata_path)
        validate_metadata_columns(
            metadata,
            [self.plot_id_column, self.site_id_column, self.label_column],
        )
        return metadata

    def _find_tile_paths(self) -> List[Path]:
        if not self.tile_dir.exists():
            raise FileNotFoundError(f"Tile directory does not exist: {self.tile_dir}")
        tiles = [p for p in sorted(self.tile_dir.glob("**/*.tif")) if p.is_file()]
        if not tiles:
            raise FileNotFoundError(f"No GeoTIFF tiles found in: {self.tile_dir}")
        return tiles

    def load_tile(self, tile_path: Path) -> RasterData:
        raster = read_geotiff(tile_path)
        if self.noisy_regions is not None and raster.wavelengths.size > 0:
            raster.image, raster.wavelengths = remove_noisy_bands(
                raster.image, raster.wavelengths, self.noisy_regions
            )
        if raster.nodata_value is not None:
            raster.image = filter_invalid_pixels(raster.image, nodata_value=raster.nodata_value)
        return raster

    def summarize_tiles(
        self,
        summary_method: str = "mean",
        normalize: bool = False,
        normalize_method: str = "standard",
    ) -> pd.DataFrame:
        records: List[Dict[str, Any]] = []
        for tile_path in self.tile_paths:
            raster = self.load_tile(tile_path)
            summary_vector = aggregate_plot_spectra(
                raster.image,
                summary=summary_method,
                nodata_value=raster.nodata_value,
            )
            if normalize:
                summary_vector = normalize_spectra(summary_vector.reshape(1, -1), normalize_method).ravel()
            tile_name = tile_path.stem
            row = {
                "tile_path": str(tile_path),
                self.tile_id_column: tile_name,
                "product_type": raster.product_type,
                "wavelengths": raster.wavelengths,
            }
            row.update({f"band_{i}": float(value) for i, value in enumerate(summary_vector)})
            records.append(row)
        self.summary_df = pd.DataFrame(records)
        return self.summary_df

    def join_metadata(self, summary_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if summary_df is None:
            if self.summary_df is None:
                raise ValueError("Call summarize_tiles() before join_metadata().")
            summary_df = self.summary_df
        joined = summary_df.merge(
            self.metadata,
            left_on=self.tile_id_column,
            right_on=self.tile_id_column,
            how="left",
            validate="many_to_one",
        )
        missing = joined[self.label_column].isna().sum()
        if missing > 0:
            raise ValueError(
                f"Metadata join failed for {missing} plot summaries. Check tile IDs and metadata keys."
            )
        return joined

    def build_feature_matrix(
        self,
        summary_df: Optional[pd.DataFrame] = None,
        feature_mode: str = "mean",
        normalize: bool = False,
        normalize_method: str = "standard",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        summary_df = self.summarize_tiles(summary_method=feature_mode) if summary_df is None else summary_df
        joined = self.join_metadata(summary_df)
        band_columns = [c for c in joined.columns if c.startswith("band_")]
        X = joined[band_columns].astype(float).values
        if normalize:
            X = normalize_spectra(X, normalize_method)
        y = joined[self.label_column].astype(str).values
        groups = joined[self.site_id_column].astype(str).values
        labels = sorted(np.unique(y).tolist())
        return X, y, groups, labels

    def available_targets(self) -> List[str]:
        return self.metadata.columns.tolist()
