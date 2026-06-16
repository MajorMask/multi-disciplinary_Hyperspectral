#!/usr/bin/env python3
"""
Utility: inspect a few tiles to verify data loading works.

Reports: shape, wavelengths, nodata stats, value range.

Usage:
    python scripts/inspect_tiles.py --tile-dir D:/Hyperspectral_Data/Hyytiala/CASI/plot_tiles --n 3
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from model.src.dataio.raster_loader import discover_tiles, load_raster


def main():
    parser = argparse.ArgumentParser(description="Inspect raster tiles")
    parser.add_argument("--tile-dir", required=True)
    parser.add_argument("--pattern", default="*.tif")
    parser.add_argument("--n", type=int, default=3, help="Number of tiles to inspect")
    args = parser.parse_args()

    tiles = discover_tiles(args.tile_dir, args.pattern)
    print(f"\nFound {len(tiles)} tiles in {args.tile_dir}\n")

    for path in tiles[:args.n]:
        print(f"--- {path.name} ---")
        tile = load_raster(path)
        print(f"  Shape:       {tile.image.shape} (bands, rows, cols)")
        print(f"  Wavelengths: {tile.wavelengths[:5]}... ({tile.n_bands} total)")
        print(f"  WL range:    {tile.wavelengths.min():.1f} – {tile.wavelengths.max():.1f} nm")
        print(f"  Valid pix:   {tile.n_valid_pixels}/{tile.nodata_mask.size} "
              f"({tile.valid_pixel_fraction:.1%})")
        valid = tile.get_valid_spectra()
        if valid.shape[0] > 0:
            print(f"  Value range: [{valid.min():.4f}, {valid.max():.4f}]")
            print(f"  Mean refl:   {valid.mean():.4f}")
        print(f"  CRS:         {tile.crs}")
        print()


if __name__ == "__main__":
    main()
