#!/usr/bin/env python3
"""
Setup and Discovery Script — RUN THIS FIRST on the Windows laptop.

Scans the data directory to find:
  1. What file formats exist (GeoTIFF, ENVI, etc.)
  2. Where plot tiles are located
  3. How many stands/tiles are available
  4. Tile shapes, band counts, wavelength ranges
  5. Creates a metadata template CSV

Usage:
    python scripts/setup_and_discover.py --data-dir "D:/Hyperspectral_Data"
    python scripts/setup_and_discover.py --data-dir "D:/Hyperspectral_Data" --site Hyytiala
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def scan_directory(data_dir: Path, max_depth: int = 5):
    """Recursively scan for raster files and report structure."""
    extensions = defaultdict(list)
    dir_tree = defaultdict(list)

    for root, dirs, files in os.walk(data_dir):
        depth = str(root).replace(str(data_dir), "").count(os.sep)
        if depth > max_depth:
            continue
        for f in files:
            ext = Path(f).suffix.lower()
            if ext in (".tif", ".tiff", ".hdr", ".bsq", ".img", ".dat", ".las", ".laz"):
                rel = Path(root).relative_to(data_dir)
                extensions[ext].append(str(rel / f))
                dir_tree[str(rel)].append(f)

    return extensions, dir_tree


def find_tile_directories(dir_tree: dict) -> list:
    """Heuristically find directories that contain plot tiles."""
    candidates = []
    tile_keywords = ["plot", "tile", "stand", "100m", "plot_tile"]
    for dirpath, files in dir_tree.items():
        dirpath_lower = dirpath.lower()
        # Check if directory name suggests tiles
        if any(kw in dirpath_lower for kw in tile_keywords):
            candidates.append((dirpath, len(files), "keyword match"))
        # Or if it has many similarly-named raster files
        elif len(files) >= 5:
            exts = set(Path(f).suffix.lower() for f in files)
            if exts & {".tif", ".tiff", ".hdr", ".bsq"}:
                candidates.append((dirpath, len(files), "file count heuristic"))
    return candidates


def inspect_sample_tiles(data_dir: Path, tile_dir: str, pattern: str = "*.tif", n: int = 2):
    """Load a few tiles and report their properties."""
    try:
        from src.dataio.raster_loader import discover_tiles, load_raster
    except ImportError as e:
        print(f"  [WARN] Cannot load raster_loader: {e}")
        print("  Install rasterio: pip install rasterio")
        return None

    full_path = data_dir / tile_dir
    try:
        tiles = discover_tiles(full_path, pattern)
    except FileNotFoundError:
        # Try other patterns
        for alt_pattern in ["*.tiff", "*.hdr", "*.bsq"]:
            try:
                tiles = discover_tiles(full_path, alt_pattern)
                pattern = alt_pattern
                break
            except FileNotFoundError:
                continue
        else:
            print(f"  [WARN] No raster files found in {full_path}")
            return None

    print(f"\n  Found {len(tiles)} files matching '{pattern}' in {tile_dir}")
    print(f"  Stand IDs (from filenames): {[t.stem for t in tiles[:8]]}{'...' if len(tiles) > 8 else ''}")

    results = []
    for path in tiles[:n]:
        try:
            tile = load_raster(path)
            info = {
                "filename": path.name,
                "stand_id": tile.stand_id,
                "shape": tile.image.shape,
                "n_bands": tile.n_bands,
                "wl_range": (float(tile.wavelengths.min()), float(tile.wavelengths.max())),
                "has_wavelengths": not all(tile.wavelengths == range(tile.n_bands)),
                "valid_fraction": tile.valid_pixel_fraction,
                "value_range": (float(tile.get_valid_spectra().min()),
                                float(tile.get_valid_spectra().max()))
                               if tile.n_valid_pixels > 0 else (0, 0),
                "crs": tile.crs,
            }
            results.append(info)

            print(f"\n  --- {path.name} ---")
            print(f"    Shape:       {info['shape']} (bands, rows, cols)")
            print(f"    Bands:       {info['n_bands']}")
            print(f"    Wavelengths: {'Yes' if info['has_wavelengths'] else 'NO (using indices)'} "
                  f"  Range: {info['wl_range'][0]:.0f}–{info['wl_range'][1]:.0f} nm")
            print(f"    Valid pix:   {info['valid_fraction']:.1%}")
            print(f"    Value range: [{info['value_range'][0]:.4f}, {info['value_range'][1]:.4f}]")
            print(f"    CRS:         {info['crs']}")
        except Exception as e:
            print(f"  [ERROR] Failed to load {path.name}: {e}")

    return {"pattern": pattern, "n_tiles": len(tiles), "stand_ids": [t.stem for t in tiles], "samples": results}


def create_metadata_template(stand_ids: list, output_path: Path, site: str = "Hyytiala"):
    """Create metadata CSV template."""
    import csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stand_id", "site", "forest_type", "dominant_species", "notes"])
        for sid in sorted(stand_ids):
            writer.writerow([sid, site, "", "", ""])
    print(f"\n  Metadata template saved: {output_path}")
    print(f"  {len(stand_ids)} stands — fill in the 'forest_type' column with: coniferous, broadleaved, or mixed")


def update_config(config_path: Path, data_dir: str, tile_dir: str, pattern: str, metadata_path: str):
    """Suggest config updates."""
    print(f"\n{'='*60}")
    print("SUGGESTED CONFIG UPDATES")
    print(f"{'='*60}")
    print(f"Edit: {config_path}\n")
    print(f"  data:")
    print(f'    root_dir: "{data_dir}"')
    print(f'    casi_tile_dir: "{tile_dir}"')
    print(f'    metadata_file: "{metadata_path}"')
    print(f'    tile_filename_pattern: "{pattern}"')
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Discover data layout and set up for experiments"
    )
    parser.add_argument("--data-dir", required=True, help="Root directory with downloaded data")
    parser.add_argument("--site", default="Hyytiala", help="Site name (default: Hyytiala)")
    parser.add_argument("--max-depth", type=int, default=6, help="Max directory scan depth")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        print("Check the path and try again.")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"FREEDLES Data Discovery — {args.site}")
    print(f"{'='*60}")
    print(f"Scanning: {data_dir}\n")

    # Step 1: Scan for files
    extensions, dir_tree = scan_directory(data_dir, args.max_depth)

    print("File types found:")
    for ext, files in sorted(extensions.items(), key=lambda x: -len(x[1])):
        print(f"  {ext:8s} : {len(files)} files")
        if len(files) <= 5:
            for f in files:
                print(f"             {f}")

    print(f"\nDirectories with raster files: {len(dir_tree)}")
    for d, files in sorted(dir_tree.items()):
        raster_count = len([f for f in files if Path(f).suffix.lower() in (".tif", ".tiff", ".hdr", ".bsq")])
        if raster_count > 0:
            print(f"  {d:50s} ({raster_count} raster files)")

    # Step 2: Find tile directories
    print(f"\n{'='*60}")
    print("TILE DIRECTORY CANDIDATES")
    print(f"{'='*60}")
    candidates = find_tile_directories(dir_tree)
    if candidates:
        for dirpath, n_files, reason in candidates:
            print(f"  {dirpath} ({n_files} files, {reason})")
    else:
        print("  No obvious tile directories found.")
        print("  Look at the directory listing above and identify the CASI plot tiles.")
        print("  Then run: python scripts/inspect_tiles.py --tile-dir <path>")

    # Step 3: Inspect sample tiles from best candidate
    tile_info = None
    if candidates:
        best = candidates[0][0]
        print(f"\n{'='*60}")
        print(f"INSPECTING TILES: {best}")
        print(f"{'='*60}")

        # Try common patterns
        for pat in ["*.tif", "*.tiff", "*.hdr"]:
            tile_info = inspect_sample_tiles(data_dir, best, pat)
            if tile_info and tile_info["n_tiles"] > 0:
                break

    # Step 4: Create metadata template
    if tile_info and tile_info["stand_ids"]:
        print(f"\n{'='*60}")
        print("METADATA TEMPLATE")
        print(f"{'='*60}")
        meta_path = project_root / "metadata" / "stand_metadata.csv"
        create_metadata_template(tile_info["stand_ids"], meta_path, args.site)

        # Step 5: Suggest config
        update_config(
            project_root / "config" / "default.yaml",
            str(data_dir),
            candidates[0][0],
            tile_info["pattern"],
            "metadata/stand_metadata.csv",
        )

    # Summary
    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    print("1. Fill in 'forest_type' in metadata/stand_metadata.csv")
    print("   (use FREEDLES companion data or ESSD paper Table 3)")
    print("2. Update config/default.yaml with the paths shown above")
    print("3. Run: python scripts/run_baseline.py --config config/default.yaml")
    print("4. Check outputs/ for results, figures, and metrics")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
