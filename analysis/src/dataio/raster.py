from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import rasterio
except ImportError:  # pragma: no cover
    rasterio = None


@dataclass
class RasterData:
    image: np.ndarray
    wavelengths: np.ndarray
    meta: Dict[str, Any]
    product_type: Optional[str] = None
    nodata_value: Optional[float] = None

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.image.shape

    def copy(self) -> RasterData:
        return RasterData(
            image=self.image.copy(),
            wavelengths=self.wavelengths.copy(),
            meta=self.meta.copy(),
            product_type=self.product_type,
            nodata_value=self.nodata_value,
        )


def ensure_rasterio_installed() -> None:
    if rasterio is None:
        raise ImportError("rasterio is required for raster loading. Install with `pip install rasterio`.")


def parse_envi_wavelengths(tags: Dict[str, str]) -> np.ndarray:
    raw = tags.get("wavelength") or tags.get("Wavelength")
    if raw is None:
        raw = tags.get("band names") or tags.get("band_names")
    if raw is None:
        raise ValueError("ENVI metadata does not include wavelengths or band names.")
    if isinstance(raw, (list, tuple)):
        items = raw
    else:
        raw = raw.strip()
        if raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1]
        items = [item.strip().strip('"').strip("'") for item in raw.split(",") if item.strip()]
    try:
        return np.array([float(item) for item in items], dtype=float)
    except ValueError as exc:
        raise ValueError("Unable to parse wavelengths from ENVI metadata.") from exc


def detect_product_type(tags: Dict[str, Any], path: Path) -> Optional[str]:
    description = str(tags.get("description", "") or tags.get("sensor", "") or path.name).upper()
    if "CASI" in description:
        return "CASI"
    if "CS" in description or "SPEX" in description:
        return "CS"
    return None


def apply_scale(image: np.ndarray, scale_factor: float = 0.0001) -> np.ndarray:
    return image.astype(np.float32) * scale_factor


def mask_nodata(image: np.ndarray, nodata_value: float = 10000.0) -> np.ndarray:
    mask = image == nodata_value
    if mask.any():
        image = image.astype(np.float32)
        image[mask] = np.nan
    return image


def read_envi(path: Path, scale_factor: float = 0.0001, nodata_value: float = 10000.0) -> RasterData:
    ensure_rasterio_installed()
    path = Path(path)
    if path.suffix.lower() == ".hdr":
        open_path = path
    elif path.suffix.lower() == ".bsq" or path.suffix.lower() == ".dat":
        open_path = path.with_suffix(".hdr")
    else:
        raise ValueError("ENVI loader expects a .hdr, .bsq, or .dat path.")

    if not open_path.exists():
        raise FileNotFoundError(f"ENVI header file does not exist: {open_path}")

    with rasterio.open(open_path) as src:
        image = src.read().astype(np.float32)
        tags = {**src.tags(), **src.tags(ns="ENVI")}
        wavelengths = parse_envi_wavelengths(tags)
        product_type = detect_product_type(tags, path)
        image = mask_nodata(image, nodata_value=nodata_value)
        image = apply_scale(image, scale_factor=scale_factor)
        meta = {
            "crs": str(src.crs) if src.crs else None,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": str(src.dtypes[0]) if src.dtypes else None,
            **tags,
        }

    return RasterData(
        image=image,
        wavelengths=wavelengths,
        meta=meta,
        product_type=product_type,
        nodata_value=nodata_value,
    )


def read_geotiff(path: Path, scale_factor: float = 0.0001, nodata_value: float = 10000.0) -> RasterData:
    ensure_rasterio_installed()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GeoTIFF path does not exist: {path}")
    with rasterio.open(path) as src:
        image = src.read().astype(np.float32)
        tags = {**src.tags(), **src.tags(ns="TIFF")}
        wavelengths = np.array([])
        if "wavelength" in tags or "band names" in tags or "BandNames" in tags:
            try:
                wavelengths = parse_envi_wavelengths(tags)
            except ValueError:
                wavelengths = np.array([])
        product_type = detect_product_type(tags, path)
        image = mask_nodata(image, nodata_value=nodata_value)
        image = apply_scale(image, scale_factor=scale_factor)
        meta = {
            "crs": str(src.crs) if src.crs else None,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": str(src.dtypes[0]) if src.dtypes else None,
            **tags,
        }
    return RasterData(
        image=image,
        wavelengths=wavelengths,
        meta=meta,
        product_type=product_type,
        nodata_value=nodata_value,
    )
