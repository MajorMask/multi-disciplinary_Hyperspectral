"""
Raster data loading for ENVI BSQ/HDR and GeoTIFF plot tiles.

Supports:
- ENVI flightline files (.bsq + .hdr)
- Analysis-ready GeoTIFF plot tiles (.tif)
- Automatic wavelength extraction from ENVI metadata
- Nodata masking

References:
    Rautiainen et al. (2024), ESSD, 16, 5069–5098.
    CASI-1500: 382–1052 nm, 15 nm step, 0.5 m pixels.
    SASI-600: 958–2443 nm, 15 nm step, 1.25 m pixels.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import rasterio
except ImportError:
    rasterio = None
    logger.warning(
        "rasterio not installed. Install with: pip install rasterio"
    )


@dataclass
class RasterTile:
    """Container for a loaded raster tile with metadata."""
    image: np.ndarray                # shape: (bands, rows, cols), float32
    wavelengths: np.ndarray          # shape: (bands,), nm
    nodata_mask: np.ndarray          # shape: (rows, cols), bool — True = valid
    filepath: Path
    stand_id: str
    crs: Optional[str] = None
    transform: Optional[object] = None
    meta: Dict = field(default_factory=dict)

    @property
    def n_bands(self) -> int:
        return self.image.shape[0]

    @property
    def n_rows(self) -> int:
        return self.image.shape[1]

    @property
    def n_cols(self) -> int:
        return self.image.shape[2]

    @property
    def n_valid_pixels(self) -> int:
        return int(self.nodata_mask.sum())

    @property
    def valid_pixel_fraction(self) -> float:
        total = self.nodata_mask.size
        return self.n_valid_pixels / total if total > 0 else 0.0

    def get_valid_spectra(self) -> np.ndarray:
        """Extract spectra of all valid pixels. Shape: (n_valid, n_bands)."""
        pixels = self.image.reshape(self.n_bands, -1).T  # (rows*cols, bands)
        mask_flat = self.nodata_mask.ravel()
        return pixels[mask_flat]


def _ensure_rasterio():
    if rasterio is None:
        raise ImportError(
            "rasterio is required for raster loading. "
            "Install with: pip install rasterio"
        )


def _parse_envi_wavelengths(tags: Dict[str, str]) -> Optional[np.ndarray]:
    """Parse wavelength list from ENVI metadata tags."""
    for key in ("wavelength", "Wavelength", "band names", "band_names"):
        raw = tags.get(key)
        if raw is None:
            continue
        text = str(raw).strip()
        if text.startswith("{") and text.endswith("}"):
            text = text[1:-1]
        parts = [s.strip().strip("'\"") for s in text.split(",") if s.strip()]
        try:
            return np.array([float(v) for v in parts], dtype=np.float64)
        except ValueError:
            continue
    return None


def load_raster(
    filepath: Union[str, Path],
    nodata_value: float = 0,
) -> RasterTile:
    """
    Load a raster tile (ENVI or GeoTIFF) and return a RasterTile object.

    Parameters
    ----------
    filepath : path to .tif, .hdr, or .bsq file
    nodata_value : pixel value to treat as missing

    Returns
    -------
    RasterTile with image array, wavelengths, and validity mask.
    """
    _ensure_rasterio()
    filepath = Path(filepath)

    # For .bsq files, open via the .hdr companion
    open_path = filepath
    if filepath.suffix.lower() == ".bsq":
        hdr = filepath.with_suffix(".hdr")
        if hdr.exists():
            open_path = hdr
        else:
            raise FileNotFoundError(
                f"Expected .hdr companion for {filepath}, not found at {hdr}"
            )

    with rasterio.open(open_path) as src:
        image = src.read().astype(np.float32)  # (bands, rows, cols)
        tags = {**src.tags(), **src.tags(ns="ENVI")}
        crs = str(src.crs) if src.crs else None
        transform = src.transform

        # Try to extract wavelengths
        wavelengths = _parse_envi_wavelengths(tags)
        if wavelengths is None:
            # Fallback: generate band indices
            logger.warning(
                f"No wavelength metadata in {filepath.name}. "
                f"Using band indices 0..{src.count-1}."
            )
            wavelengths = np.arange(src.count, dtype=np.float64)

        meta = {
            "driver": src.driver,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": str(src.dtypes[0]),
        }
        meta.update(tags)

    # Build validity mask: pixel is valid if no band is nodata and all finite
    nodata_mask = np.ones(image.shape[1:], dtype=bool)
    if nodata_value is not None:
        nodata_mask &= ~np.any(image == nodata_value, axis=0)
    nodata_mask &= np.all(np.isfinite(image), axis=0)

    stand_id = filepath.stem

    logger.info(
        f"Loaded {filepath.name}: {image.shape[0]} bands, "
        f"{image.shape[1]}x{image.shape[2]} pixels, "
        f"{nodata_mask.sum()}/{nodata_mask.size} valid "
        f"({nodata_mask.mean()*100:.1f}%)"
    )

    return RasterTile(
        image=image,
        wavelengths=wavelengths,
        nodata_mask=nodata_mask,
        filepath=filepath,
        stand_id=stand_id,
        crs=crs,
        transform=transform,
        meta=meta,
    )


def discover_tiles(
    tile_dir: Union[str, Path],
    pattern: str = "*.tif",
) -> List[Path]:
    """
    Find all raster tile files in a directory.

    Parameters
    ----------
    tile_dir : directory to search
    pattern : glob pattern (e.g., "*.tif", "*.hdr")

    Returns
    -------
    Sorted list of file paths.
    """
    tile_dir = Path(tile_dir)
    if not tile_dir.is_dir():
        raise FileNotFoundError(f"Tile directory not found: {tile_dir}")

    tiles = sorted(tile_dir.glob(pattern))
    if not tiles:
        # Try recursive search
        tiles = sorted(tile_dir.rglob(pattern))

    if not tiles:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {tile_dir}"
        )

    logger.info(f"Found {len(tiles)} tiles in {tile_dir}")
    return tiles


def load_all_tiles(
    tile_dir: Union[str, Path],
    pattern: str = "*.tif",
    nodata_value: float = 0,
    min_valid_fraction: float = 0.5,
) -> List[RasterTile]:
    """
    Load all raster tiles from a directory, filtering out low-quality ones.

    Parameters
    ----------
    tile_dir : directory containing tiles
    pattern : glob pattern
    nodata_value : pixel value to treat as missing
    min_valid_fraction : minimum fraction of valid pixels to keep a tile

    Returns
    -------
    List of RasterTile objects that pass quality filter.
    """
    paths = discover_tiles(tile_dir, pattern)
    tiles = []
    skipped = 0

    for path in paths:
        try:
            tile = load_raster(path, nodata_value=nodata_value)
        except Exception as e:
            logger.warning(f"Failed to load {path.name}: {e}")
            skipped += 1
            continue

        if tile.valid_pixel_fraction < min_valid_fraction:
            logger.warning(
                f"Skipping {path.name}: only {tile.valid_pixel_fraction:.1%} "
                f"valid pixels (threshold: {min_valid_fraction:.1%})"
            )
            skipped += 1
            continue

        tiles.append(tile)

    logger.info(
        f"Loaded {len(tiles)} tiles, skipped {skipped} "
        f"(quality or read errors)"
    )
    return tiles
