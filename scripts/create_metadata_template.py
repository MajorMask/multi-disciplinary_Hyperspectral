#!/usr/bin/env python3
"""
Utility: discover tiles and create a metadata template CSV.

Run this on the Windows laptop after downloading the Hyytiälä data.
It scans the tile directory, extracts stand IDs from filenames,
and writes a CSV template for you to fill in with forest type labels.

Usage:
    python scripts/create_metadata_template.py --tile-dir D:/Hyperspectral_Data/Hyytiala/CASI/plot_tiles
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from model.src.dataio.raster_loader import discover_tiles
from model.src.dataio.metadata_loader import create_template_metadata


def main():
    parser = argparse.ArgumentParser(description="Create metadata template from tile filenames")
    parser.add_argument("--tile-dir", required=True, help="Directory with plot tile files")
    parser.add_argument("--pattern", default="*.tif", help="Glob pattern for tiles")
    parser.add_argument("--output", default="metadata/stand_metadata_template.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    tiles = discover_tiles(args.tile_dir, args.pattern)
    stand_ids = [t.stem for t in tiles]
    print(f"Found {len(stand_ids)} tiles: {stand_ids[:5]}...")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    create_template_metadata(stand_ids, output_path)
    print(f"Template saved to: {output_path}")
    print("Fill in the 'forest_type' column with: coniferous, broadleaved, or mixed")


if __name__ == "__main__":
    main()
