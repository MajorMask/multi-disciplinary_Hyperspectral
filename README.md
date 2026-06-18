# Airborne Hyperspectral Forest Classification

**Research question:** Can forest location and forest type be recovered from airborne hyperspectral spectra alone?

**Dataset:** FREEDLES airborne hyperspectral dataset (Hovi et al., 2024, ESSD 16, 5069–5098)
- Etsin/Fairdata ID: `57e630e9-58a2-41a1-ac39-9ebb923040a4`
- DOI: `10.23729/c6da63dd-f527-4ec9-8401-57c14f77d19f`

**Current scope:** Hyytiälä site only (28 boreal stands, CASI sensor 382–1052 nm).

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download Hyytiälä CASI plot tiles from Etsin to D:/Hyperspectral_Data/

# 3. Create metadata template (run on Windows laptop with data)
python scripts/create_metadata_template.py \
    --tile-dir D:/Hyperspectral_Data/Hyytiala/CASI/plot_tiles

# 4. Fill in forest_type column in the generated CSV
#    (coniferous / broadleaved / mixed — from FREEDLES companion data)

# 5. Inspect a few tiles to verify loading
python scripts/inspect_tiles.py \
    --tile-dir D:/Hyperspectral_Data/Hyytiala/CASI/plot_tiles

# 6. Run baseline experiment
python scripts/run_baseline.py --config config/default.yaml
```

## Project Structure

```
├── config/
│   └── default.yaml          # Experiment configuration
├── src/
│   ├── dataio/               # Data loading (raster, metadata, ALS, CV splits)
│   ├── preprocessing/        # Band selection, normalization, resampling
│   ├── features/             # Stand summaries, spectral indices, PCA, patches
│   ├── models/               # Classical ML + CNN classifiers
│   ├── evaluation/           # Metrics, reporting, shortcut analysis
│   └── experiments/          # Orchestration runner
├── scripts/                  # CLI utilities
├── notebooks/                # EDA and analysis notebooks
├── outputs/                  # Generated results, figures, models
├── analysis/                 # Research strategy and legacy code
└── requirements.txt
```

## Experimental Design

1. **Baseline:** Stand-mean spectra → 5 classical classifiers (LR, RF, SVM, GB, PLS-DA) with leave-one-stand-out CV
2. **Feature engineering:** PCA, spectral indices, normalization ablation
3. **Spatial models:** 1D CNN on pixel spectra, 3D spatial-spectral CNN on patches
4. **Fusion:** Spectral + ALS terrain features

## Limitations

- Only Hyytiälä site processed (storage constraint) — cross-site generalization untested
- All 28 stands are boreal/coniferous-dominated — limited class diversity
- Leave-one-stand-out with 28 samples yields high variance estimates
- No field-measured LAI/biomass validation (companion data not processed)

## References

- Hovi, A., Forsström, P., Mõttus, M., et al. (2024). FREEDLES airborne remote sensing data. ESSD 16, 5069–5098.
- Roberts, D.R., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography 40(8), 913–929.
- Ploton, P., et al. (2020). Spatial validation reveals poor predictive performance of large-scale ecological mapping models. Nature Comms 11, 4540.
- Fassnacht, F.E., et al. (2016). Review of studies on tree species classification from remotely sensed data. RSE 186, 64–87.
- Paoletti, M.E., et al. (2019). Deep learning classifiers for hyperspectral imaging: A review. ISPRS J 158, 279–317.
