# Airborne Hyperspectral Forest Classification — CLAUDE.md

## What This Project Is

A 10-credit research project investigating whether forest location and forest type can be recovered from airborne hyperspectral spectra alone. Uses the **FREEDLES airborne dataset** (Rautiainen et al. 2024, ESSD 16, 5069–5098).

**Research question**: Can spectral reflectance distinguish coniferous, broadleaved, and mixed forests — and can models generalize across sites spanning boreal, hemiboreal, and temperate biomes?

## Current State

The code repository is **complete and tested** (all 23 Python files compile, imports verified). What's missing is **running it on the actual data** — which only exists on this Windows laptop.

### Deliverables already created (in this repo):
- `analysis/RESEARCH_STRATEGY.md` — comprehensive strategy (READ THIS FIRST for full context)
- `Preliminary_Research_Report.docx` — report shell, needs real results inserted
- `Presentation_Hyperspectral_Forest.pptx` — 9-slide deck, needs real results
- Full Python pipeline under `src/`

### What still needs to happen:
1. **Discover and inspect the Hyytiälä data** (run `scripts/setup_and_discover.py`)
2. **Create the metadata CSV** (the script helps, but forest_type labels need manual entry from FREEDLES companion data)
3. **Run the baseline experiment** (`python scripts/run_baseline.py`)
4. **Run additional experiments** (feature engineering, CNN, ablations)
5. **Update the report and presentation** with actual results

## Dataset Details

**Source**: Etsin/Fairdata, ID `57e630e9-58a2-41a1-ac39-9ebb923040a4`
**DOI**: `10.23729/c6da63dd-f527-4ec9-8401-57c14f77d19f`

### What's on disk (~250 GB, Hyytiälä site only):

The data was downloaded from Etsin. The exact directory structure may vary, but expect something like:

```
D:/Hyperspectral_Data/          (or wherever the user put it)
├── Hyytiala/
│   ├── CASI/
│   │   ├── plot_tiles/          ← 100m×100m analysis-ready tiles (GeoTIFF or ENVI)
│   │   └── flightlines/         ← full swath data (large, optional)
│   ├── SASI/
│   │   ├── plot_tiles/
│   │   └── flightlines/
│   └── ALS/
│       ├── DEM/                 ← 1m digital elevation models
│       └── point_clouds/        ← LAS files (optional)
```

**IMPORTANT**: The actual folder names may differ. Run `scripts/setup_and_discover.py` first — it will scan the data directory and report what it finds.

### Sensors:
- **CASI-1500**: 382–1052 nm, ~45 bands at 15 nm step, **0.5 m pixels** (VIS-NIR)
- **SASI-600**: 958–2443 nm, ~100 bands at 15 nm step, **1.25 m pixels** (NIR-SWIR)
- **ALS**: RIEGL LMS-Q780, 1m DEMs

### Expected tile properties:
- ~200×200 pixels per 100m×100m tile (CASI at 0.5m)
- GeoTIFF (`.tif`) or ENVI (`.bsq` + `.hdr`)
- Wavelengths stored in ENVI header or GeoTIFF metadata
- Nodata value: likely 0 or -9999

### Water vapor bands to EXCLUDE (interpolated by ATCOR-4, not measured):
- 895–1003 nm
- 1092–1168 nm
- 1302–1528 nm
- 1737–2038 nm

The code handles this automatically via `src/preprocessing/band_selection.py`.

## Step-by-Step Execution Guide

### Step 0: Setup
```bash
cd <this_repo_directory>
pip install -r requirements.txt
```

### Step 1: Discover the data
```bash
python scripts/setup_and_discover.py --data-dir "D:/Hyperspectral_Data"
```
This scans for tile files, reports formats/shapes/wavelengths, and creates a metadata template CSV. **Adapt the `--data-dir` path to wherever the data actually lives.**

If the script can't find tiles automatically, look manually:
```bash
python scripts/inspect_tiles.py --tile-dir "D:/Hyperspectral_Data/Hyytiala/CASI/plot_tiles" --n 3
```

### Step 2: Create metadata CSV
The discovery script creates `metadata/stand_metadata_template.csv` with stand IDs pre-filled. You need to add the `forest_type` column values.

**Where to get forest type labels**: The FREEDLES companion dataset includes forest inventory tables. Look for columns like "dominant species", "forest type", or "stand type". At Hyytiälä:
- Stands dominated by Norway spruce or Scots pine → `coniferous`
- Stands dominated by birch or other broadleaved species → `broadleaved`
- Mixed stands → `mixed`

The ESSD paper (Table 3 and supplementary) lists all 28 Hyytiälä stands with species composition.

### Step 3: Update config
Edit `config/default.yaml`:
- Set `data.root_dir` to your actual data location
- Set `data.casi_tile_dir` to the actual CASI tile path (relative to root_dir)
- Set `data.metadata_file` to your filled-in metadata CSV
- Set `data.tile_filename_pattern` to match your file format (`*.tif` or `*.hdr`)

### Step 4: Run baseline experiment
```bash
python scripts/run_baseline.py --config config/default.yaml
```
This runs all 5 classifiers with leave-one-stand-out CV and saves results to `outputs/`.

### Step 5: Generate results summary
```bash
python scripts/summarize_results.py --results-dir outputs/baseline_hyytialä_casi
```
This prints a formatted summary of all metrics, ready for the report.

### Step 6: Additional experiments (time permitting)

The experimental hierarchy from RESEARCH_STRATEGY.md:

| Priority | Experiment | What to change in config |
|----------|-----------|-------------------------|
| **P1** | Baseline (stand-mean, 5 classifiers, LOGO-Stand) | Default config |
| **P2** | PCA reduction (5, 10, 20 components) | `features.pca_components: 10` |
| **P3** | Normalization ablation (SNV, MinMax) | `preprocessing.normalization: "snv"` |
| **P4** | Spectral indices as extra features | `features.include_indices: true` |
| **P5** | 1D CNN on pixel spectra | `features.level: "pixel"` + use cnn_1d model |
| **P6** | ALS fusion | Add ALS features via `src/features/als_features.py` |

For a 10-credit project with only Hyytiälä, **P1–P4 are sufficient**. P5–P6 are bonus.

## Known Pitfalls

1. **Metadata is the bottleneck**: The code can load tiles automatically, but it needs a CSV mapping stand_id → forest_type. Without this, nothing runs. The discovery script helps, but labeling is manual.

2. **File format varies**: Some tiles may be GeoTIFF, others ENVI. The loader handles both — just set the right glob pattern in config.

3. **Wavelength metadata**: If tiles don't have wavelength info in their headers, the loader falls back to band indices. Check with `inspect_tiles.py` first.

4. **Class imbalance**: Hyytiälä is dominated by coniferous stands. All classifiers use `class_weight="balanced"` by default. Still expect poor minority-class performance.

5. **n=28 is small**: Leave-one-stand-out gives 28 folds but each test set has only 1 sample. High variance is expected and honest. Report per-fold results.

6. **Don't use random splits**: Always split by stand (GroupKFold or LOGO). Never split pixels randomly — this leaks spatial autocorrelation and produces fake accuracy.

7. **ATCOR-4 interpolated bands**: The water vapor bands look real but are interpolated. The pipeline excludes them by default. Don't override this.

## Key Files

| File | Purpose |
|------|---------|
| `config/default.yaml` | All experiment parameters — edit paths here |
| `src/experiments/runner.py` | Main orchestrator — runs full pipeline |
| `src/dataio/raster_loader.py` | Loads ENVI/GeoTIFF tiles |
| `src/dataio/metadata_loader.py` | Loads stand metadata CSV |
| `src/dataio/splits.py` | CV split generators |
| `src/preprocessing/band_selection.py` | Water vapor band exclusion |
| `src/models/classical.py` | All 5 classical classifiers + PLS-DA |
| `src/evaluation/metrics.py` | Metrics + all visualization functions |
| `scripts/setup_and_discover.py` | **Run first** — finds data, creates metadata template |
| `scripts/run_baseline.py` | Quick-start for baseline experiment |
| `scripts/summarize_results.py` | Formats results for the report |

## For the Final Report

The `Preliminary_Research_Report.docx` has placeholder sections for results. After running experiments:

1. Run `scripts/summarize_results.py` to get formatted metrics
2. The pipeline auto-generates figures in `outputs/*/figures/`:
   - `mean_spectra_per_class.png` — spectral signatures by forest type
   - `*_confusion_matrix.png` — per-model confusion matrices
   - `*_per_fold_accuracy.png` — per-fold performance bars
   - `rf_band_importance.png` — which wavelengths matter most
3. Insert these into the report sections 3.2 and 3.3
4. Update the presentation slides 6 (scope) with actual numbers

## What Success Looks Like for 10 Credits

For Hyytiälä-only with honest limitations:
- Baseline results from 5 classifiers with LOGO-Stand CV ✓
- At least one ablation (PCA or normalization comparison) ✓
- Confusion matrices and per-fold reporting ✓
- Permutation test confirming significance ✓
- Spectral plots (mean spectra per class, band importance) ✓
- Honest discussion of single-site limitations ✓
- Code that demonstrably works and is well-documented ✓

The research strategy document (`analysis/RESEARCH_STRATEGY.md`) has the full scientific framing, guardrails, and literature references. Read it for any conceptual questions.
