# Etsin Hyperspectral Modeling Pipeline

This analysis is designed to support the Etsin airborne hyperspectral dataset and answer the research question:

> Can forest location and forest type be recovered from airborne hyperspectral spectra alone?

## Goals

- Build a reusable pipeline for ENVI flightline data and analysis-ready plot tiles.
- Use grouped validation (site-level or plot-level) to avoid spatial leakage.
- Benchmark classical spectral-only models before extending to spatial-spectral approaches.
- Evaluate both forest-type classification and location/identity recovery.

## Dataset inputs

- `*.hdr` / `*.bsq`: ENVI flightline raster files.
- `*.tif`: analysis-ready plot tiles.
- metadata CSV/GeoJSON/shapefile: labels for site, plot, forest type, and optional geometry.

## Pipeline structure

The primary script is `etsin_hyperspectral_pipeline.py`.
It includes modular stages:

1. Data loading
   - ENVI and GeoTIFF raster readers
   - metadata ingestion
2. Preprocessing
   - remove noisy water-absorption bands
   - normalize spectra
   - compute plot-level summary spectra
3. Feature extraction
   - mean/median/standard deviation summaries
   - patch-level extraction stubs for future spatial-spectral work
4. Modeling
   - Logistic regression
   - Random forest
   - Gradient boosting
   - PLS-DA baseline
5. Validation
   - Leave-one-site-out
   - GroupKFold
   - balanced accuracy, macro F1, confusion matrices

## Research design

### Problem formulation

- Input representation: reflectance spectra per pixel, patch, or aggregated plot tiles.
- Targets:
  - `forest_type` for forest classification.
  - `site_id` / `plot_id` for location recovery.
- Prediction unit:
  - Start with plot-level summaries as a scientifically defensible baseline.
  - Extend to pixel-level or patch-level models once summary-level performance is established.
- Leakage risks:
  - Random splits mix pixels from the same plot/site across train/test.
  - Grouped validation is required to measure generalization across new sites.

### Method selection

- Baselines:
  - Logistic regression, random forest, gradient boosting, PLS-DA.
- Advanced directions:
  - 1D spectral CNNs on reflectance vectors.
  - 2D/3D spatial-spectral CNNs on plot tiles.
  - Domain generalization / adaptation for unseen sites.

### Experimental progression

1. Start with plot-level mean spectra + grouped CV.
2. Add leave-one-site-out validation.
3. Compare full spectrum vs selected wavelength ranges.
4. Compare spectral-only vs patch-based spatial features.
5. If available, add ALS / structural predictors.

## Usage example

```bash
python analysis/etsin_hyperspectral_pipeline.py \
  --metadata data/etsin_metadata.csv \
  --tile-dir data/plot_tiles \
  --tile-id-col tile_name \
  --plot-id-col plot_id \
  --site-col site_id \
  --label-col forest_type \
  --summary-method mean \
  --cv-mode leave-one-site-out \
  --output-dir analysis/results
```

## Notes

- The script is intentionally generic to support new Etsin file conventions.
- `rasterio` is required for raster loading; `geopandas` is required for geospatial metadata.
- This code is a foundation for reproducible experiments and later extension into deep spectral-spatial models.
