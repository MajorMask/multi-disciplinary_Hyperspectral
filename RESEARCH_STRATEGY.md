# Research Strategy: Airborne Hyperspectral Forest Classification

**Dataset**: Hovi, A., Schraik, D., Hanuš, J., Lukeš, P., Lhotáková, Z., Homolová, L., & Rautiainen, M. (2024). *A spectral–structural characterization of European temperate, hemiboreal, and boreal forests: Airborne data.* Aalto University. DOI: 10.23729/c6da63dd-f527-4ec9-8401-57c14f77d19f

**Paper**: Rautiainen et al. (2024). *A spectral–structural characterization of European temperate, hemiboreal, and boreal forests.* Earth System Science Data, 16, 5069–5098.

---

## Critical Audit of Existing Work

### All existing code operates on the wrong dataset

The files `peatland_hyperspectral_analysis.py`, `peatland_eda.ipynb`, and associated outputs use the **Salko et al. (2024) peatland field spectrometer CSV** — a ground-based dataset of 446 peatland plots measured with a portable spectrometer. This is a completely different dataset from a different study, despite overlapping authorship (Rautiainen lab, Aalto University).

The actual project dataset is the **FREEDLES airborne dataset**: CASI + SASI airborne imagery and ALS point clouds from 4 European forest sites. The two datasets differ in every fundamental respect:

| Property | What the code uses (Salko) | What the project requires (Hovi/Rautiainen) |
|----------|---------------------------|---------------------------------------------|
| **Acquisition** | Ground-based field spectrometer | Airborne CASI-1500 + SASI-600 sensors |
| **Data format** | CSV | ENVI BSQ/HDR, GeoTIFF plot tiles, LAS |
| **Spatial dimension** | None (1 spectrum per 1m² plot) | Yes (0.5m / 1.25m pixel imagery) |
| **Sites** | 13 peatland sites (Finland + Estonia) | 4 forest sites (Finland, Estonia, Czech Rep.) |
| **Samples** | 446 plot-level spectra | 58 forest stands with full spatial imagery |
| **Target** | Finnish peatland type | Forest type, tree species, site identity |
| **Volume** | ~10 MB CSV | ~533 GB imagery + point clouds |
| **Spectral range** | 350–2500 nm, 1 nm step (2151 bands) | CASI: 382–1052 nm, SASI: 958–2443 nm, 15 nm step |

**Consequence**: All existing code, notebooks, EDA, and models must be set aside. The pipeline must be rebuilt from scratch for raster imagery.

The `etsin_hyperspectral_pipeline.py` was designed for raster data (ENVI + GeoTIFF loading), which is the right *format* — but it was never tested because the actual data was never loaded. It can serve as a starting scaffold after review and correction.

---

## 1. Research Identity

### One-sentence definition

This project investigates whether forest location (site identity) and forest type (coniferous, broadleaved, mixed) can be predicted from airborne hyperspectral reflectance imagery alone, evaluated under leave-one-site-out cross-validation across four European forest sites spanning boreal, hemiboreal, and temperate biomes.

### The dataset

**4 study sites, 58 stands:**

| Site | Country | Biome | Stands | Dominant species | Coordinates |
|------|---------|-------|--------|------------------|-------------|
| Hyytiälä | Finland | Boreal | 28 | Norway spruce, Scots pine | 61.85°N, 24.3°E |
| Järvselja | Estonia | Hemiboreal | 13 | Mixed broadleaved/coniferous | 58.28°N, 27.3°E |
| Lanzhot | Czech Republic | Temperate | 10 | Broadleaf floodplain forest | 48.68°N, 16.95°E |
| Bily Kriz | Czech Republic | Temperate | 7 | Norway spruce (mountain) | 49.5°N, 18.5°E |

**Airborne sensors:**

| Sensor | Range | Resolution | Pixel size |
|--------|-------|------------|------------|
| CASI-1500 | 382–1052 nm (VIS-NIR) | 15 nm | 0.5 m |
| SASI-600 | 958–2443 nm (NIR-SWIR) | 15 nm | 1.25 m |

**ALS**: RIEGL LMS-Q780, 1064 nm, pulse densities 9–48 pulses/m², 1 m DEM rasters.

**Data products available**:
- Full-site flightlines (~4 km × 4 km extent per site)
- Analysis-ready 100 m × 100 m plot tiles (hyperspectral HDRF)
- ALS point clouds (original + denoised) and 1 m DEMs
- Forest inventory tables with tree species, basal area, stand structure
- Forest floor and canopy spectral measurements (companion dataset)

### Task distinctions

| Task | Target | Input level | Validation |
|------|--------|-------------|------------|
| **Forest type classification** | Coniferous / broadleaved / mixed (categorical) | Stand-level or pixel-level | Leave-one-site-out |
| **Tree species classification** | 15 species (categorical) | Stand-level or pixel-level | Leave-one-site-out |
| **Site identity recovery** | Site (4-class categorical) | Stand-level or pixel-level | Cross-validation within sites |
| **Location regression** | Latitude (continuous) | Stand-level | Leave-one-site-out |

**Prediction unit options** (from most conservative to most ambitious):
1. **Stand-level summary spectra**: mean/median of all pixels within a 100 m × 100 m plot tile. Safest against leakage. n=58.
2. **Pixel-level classification**: each pixel is a sample. Risk of within-stand leakage if not split carefully. n=millions.
3. **Patch-level (e.g., 9×9 or 15×15 pixels)**: captures local spatial structure. Moderate leakage risk.

**Recommended primary approach**: Start with stand-level summaries (n=58, 4 groups) as proof of concept, then extend to pixel-level or patch-level models within the plot tiles.

### Scientific identity

> "Cross-Site Generalization of Airborne Hyperspectral Forest Type Classification Across European Biomes"

The novelty is the **generalization question**: can a model trained on boreal and temperate forests in three countries predict forest type at a fourth, unseen site?

---

## 2. Guardrails

### 2.1 Site leakage (CRITICAL)

**Risk**: With only 4 sites, any random split will mix stands/pixels from the same site into train and test sets. Sites share climate, soil, phenology, and atmospheric conditions. Within-site spectral similarity is extremely high.

**Bias**: Models learn site-specific spectral fingerprints (e.g., "CASI spectra from Hyytiälä always look like X") rather than forest-type signatures. Ploton et al. (2020, Nature Communications) showed that ignoring spatial structure can overestimate performance by factors of 2–5×.

**Rule**: All primary validation must use leave-one-site-out (LOSO) cross-validation with 4 folds. Random splits are **never valid** as the primary result.

**Special concern**: With only 4 folds, variance of the estimate is high. Each fold holds out one entire biome type. Report per-fold results alongside the mean.

### 2.2 Within-stand pixel leakage

**Risk**: If using pixel-level classification, neighboring pixels from the same stand are highly correlated (spatial autocorrelation). Splitting pixels randomly from the same stand into train/test produces near-perfect accuracy that does not generalize.

**Bias**: The model memorizes local spatial patterns rather than learning spectral-type relationships.

**Rule**: When doing pixel-level modeling, always split at the **stand level**: all pixels from a given stand go entirely into train or entirely into test. Never split pixels within a stand.

### 2.3 CASI/SASI harmonization

**Risk**: CASI (0.5 m, VIS-NIR) and SASI (1.25 m, NIR-SWIR) have different spatial resolutions. Merging them requires resampling. The overlap region (958–1052 nm) may show sensor-specific artifacts.

**Bias**: Models may exploit sensor artifacts or resolution differences rather than spectral information.

**Rule**: 
- Option A: Use CASI only (VIS-NIR) for a clean spectral-only experiment.
- Option B: Resample SASI to 0.5 m (or CASI to 1.25 m) and merge, but be explicit about the resampling method.
- Ablation: Compare CASI-only vs. CASI+SASI to quantify the SWIR contribution.

### 2.4 Atmospheric absorption bands

**Risk**: Water vapor absorption regions (895–1003, 1092–1168, 1302–1528, 1737–2038 nm) were interpolated during preprocessing (ATCOR-4). These bands contain interpolated values, not measured reflectance.

**Bias**: Models may learn from interpolation artifacts rather than real surface reflectance.

**Rule**: Exclude or flag the interpolated water vapor bands. The preprocessing paper notes these explicitly. Create a band mask and document which bands are excluded.

### 2.5 Class imbalance and label granularity

**Risk**: 28 of 58 stands are at Hyytiälä (boreal, predominantly coniferous). The Czech sites contribute only 17 stands. Tree species have extreme imbalance — Scots pine and Norway spruce dominate; many broadleaved species have only 1–3 stands.

**Bias**: Overall accuracy dominated by the majority class/site.

**Rule for forest type** (3 classes: coniferous/broadleaved/mixed): Viable — each type has multiple stands across sites. Use `class_weight="balanced"` and report balanced accuracy + macro F1.

**Rule for tree species** (15 species): Too fine-grained for stand-level modeling with n=58. Either aggregate to dominant-species or use only as pixel-level labels within the plot tiles. Many species are site-specific (e.g., English oak at Lanzhot only), making cross-site generalization impossible.

### 2.6 Coordinate and metadata leakage

**Risk**: If latitude, longitude, elevation, site name, or other metadata are included in the feature matrix, the model trivially learns location.

**Rule**: Feature matrix must contain ONLY spectral reflectance bands. All metadata is used exclusively for defining targets, groups, and interpretation.

### 2.7 Confounding of forest type with site

**Risk**: The four sites represent different biomes. Boreal = coniferous, temperate = broadleaved. A model might learn "this is a Lanzhot spectrum" rather than "this is a broadleaved spectrum." The confounding between site and type is inherent and cannot be fully resolved with 4 sites.

**Bias**: Apparent forest-type classification accuracy may partly reflect site identification.

**Rule**: 
1. Always report site-classification accuracy alongside type-classification accuracy.
2. If the site classifier outperforms the type classifier, the model is primarily learning site identity.
3. Discuss this confounding honestly in the paper. It is a genuine limitation of 4-site datasets, acknowledged in the domain generalization literature.

### 2.8 Unjustified use of ALS

**Risk**: ALS data (canopy height, structure, density) are powerful predictors of forest type. Adding them inflates apparent spectral-classification performance.

**Rule**: The core experiment uses spectra only. ALS is a separate ablation, explicitly labeled as a fusion experiment. The research question asks "from spectra alone."

---

## 3. Literature-Backed Solution Path

### 3.1 Classical ML on stand-level summary spectra — BASELINE

**Why it fits**: 58 stands is a very small sample. High-dimensional spectral data (CASI: ~45 bands, SASI: ~100 bands at 15 nm step) with n=58 is the canonical HDSS setting. Classical regularized classifiers are well-suited, interpretable, and directly comparable to the literature.

**Literature**: Random forests are the standard baseline in hyperspectral forest classification (Immitzer et al., 2012; Ghosh et al., 2014). PLS-DA matches the Salko et al. (2024) methodology on related data from the same group. SVM-RBF is a proven strong performer in HDSS remote sensing tasks (Mountrakis et al., 2011).

**Methods**:

| Method | Role |
|--------|------|
| Logistic Regression (L2) | Linear sanity check |
| PLS-DA | Dimensionality-reducing baseline |
| Random Forest (balanced) | Non-linear ensemble baseline |
| SVM (RBF, balanced) | Strong HDSS baseline |
| Gradient Boosting | Complementary ensemble |

**Verdict**: Primary results section. Start here. If these fail on n=58 with LOSO-CV, deeper models will also fail.

### 3.2 1D CNN on spectral vectors — SECONDARY

**Why it might fit**: 1D CNNs learn local spectral features (absorption slopes, band relationships) directly from the spectrum. With pixel-level data from plot tiles, n increases from 58 to potentially hundreds of thousands, making deep learning feasible.

**Literature**: Hu et al. (2015) and Chen et al. (2016) demonstrated 1D CNNs for HSI pixel classification. Paoletti et al. (2019) survey confirms viability for spectral-only classification.

**Practical concern**: At stand level (n=58), a 1D CNN will overfit. At pixel level, pixel leakage must be controlled. Requires careful train/test splitting at the stand level even for pixel-level models.

**Verdict**: Second-tier experiment. Use only on pixel-level data from plot tiles, with stand-level splits. Compare to RF on the same pixel-level data.

### 3.3 2D/3D spatial-spectral CNNs — ABLATION (if time permits)

**Why it fits**: The 100 m × 100 m plot tiles with 0.5 m pixels give ~200×200 spatial extent with ~45–145 spectral bands. This is a 3D hyperspectral data cube suitable for spatial-spectral models.

**Literature**: Li et al. (2019) and recent surveys (Paoletti et al., 2019; comprehensive survey by Ahmad et al., 2024) show that 3D CNNs capturing spectral-spatial features consistently outperform spectral-only models on HSI classification benchmarks.

**Practical concern**: With only 4 sites and 58 stands, training a large 3D CNN is risky. Small patch sizes (e.g., 9×9×B) are feasible. Must split by stand (all patches from one stand in train OR test, never both).

**Verdict**: Only if Experiments 1–6 are complete. Use small architectures. Report as an ablation comparing spatial-spectral to spectral-only.

### 3.4 Domain adaptation / generalization — FRAMING

**Why relevant**: The 4 sites span 3 countries and 3 biome types. LOSO-CV is inherently a domain generalization evaluation. Each site is a distinct spectral domain due to different atmospheric conditions, phenology, and sensor geometry.

**Literature**: Tuia et al. (2016, IEEE GRSM) formalized domain adaptation for remote sensing. Zhu et al. (2021) and the recent C³DG conditional domain generalization work for HSI (2024, arXiv:2407.04100) are relevant.

**Practical concern**: Formal DG methods (DANN, MMD alignment) need sufficient samples per domain. With 4 domains of 7–28 stands each, these methods are unlikely to converge reliably.

**Verdict**: Use DG as the conceptual framing in the report (the problem IS cross-domain generalization). Use LOSO-CV as the evaluation protocol. Do not implement formal DG algorithms unless pixel-level data provides enough samples per domain.

### 3.5 Fusion with ALS — CONTROLLED ABLATION

**Why interesting**: ALS-derived metrics (canopy height, tree density, gap fraction) are strong structural predictors of forest type and species.

**Rule**: Separate experiment. Report spectral-only alongside spectral+ALS. The main result is spectral-only; ALS fusion is supplementary.

**ALS features to derive** (from 1 m DEM and point clouds): max canopy height, mean canopy height, canopy cover fraction, height percentiles (p25, p50, p75, p95), canopy rugosity (height std).

---

## 4. Recommended Experimental Hierarchy

### Phase 1: Data Loading and Sanity Checks (Day 1)

**Experiment 0: Load and inspect plot tiles**
- Load a sample 100 m × 100 m CASI plot tile (ENVI BSQ/HDR or GeoTIFF).
- Verify shape: expect ~200 rows × 200 cols × ~45 bands.
- Plot a false-color composite. Plot the mean spectrum.
- Load the metadata table. Verify stand-level labels (forest type, species, site).
- Produce the Site × Forest Type cross-tabulation.
- **Question answered**: Is the data correctly formatted and loadable?

**Experiment 1: Stand-level summary spectra + linear baseline**
- Extract mean spectrum per plot tile (average all valid pixels within each 100 m × 100 m tile).
- This yields n=58 samples × ~45 CASI bands.
- Train Logistic Regression (L2, balanced) with LOSO-CV (4 folds).
- Target: forest type (coniferous / broadleaved / mixed).
- **Question answered**: Is the task linearly separable from stand-level spectra?

### Phase 2: Classical Baselines (Day 2)

**Experiment 2: Full classical baseline sweep**
- Same stand-level summary spectra.
- Models: PLS-DA, RF (500 trees), SVM (RBF), Gradient Boosting.
- LOSO-CV, 4 folds.
- Metrics: overall accuracy, balanced accuracy, macro F1, per-class precision/recall, confusion matrix.
- Report per-fold accuracy (each fold = one site held out).
- **Question answered**: Which classical model best generalizes across sites?

**Experiment 3: Site classification (shortcut test)**
- Same features.
- Target: site identity (4-class).
- Same models and CV (but here, within-site random splits since the task IS site recognition).
- **Question answered**: Do the spectra primarily encode site identity or forest type? If site accuracy >> forest type accuracy, the spectral features mainly reflect geographic/atmospheric differences.

**Experiment 4: Location regression**
- Target: latitude (continuous).
- Models: PLS regression, Ridge, Lasso.
- LOSO-CV.
- **Question answered**: Is there a spectral gradient encoding geographic location?

### Phase 3: Ablations (Day 3)

**Experiment 5: CASI-only vs. CASI+SASI**
- Compare forest type classification using: (a) CASI bands only (VIS-NIR), (b) SASI bands only (SWIR), (c) CASI+SASI merged.
- Best model from Experiment 2.
- **Question answered**: Does the SWIR region add discriminative power for forest type?

**Experiment 6: Band selection and dimensionality**
- Compare: full spectrum vs. PCA (5, 10, 20 components) vs. ANOVA top-N bands.
- **Question answered**: How many spectral features are needed? Is dimensionality reduction helpful or harmful?

**Experiment 7: ALS fusion**
- Derive structural metrics from ALS point clouds (canopy height stats, cover fraction).
- Compare: spectra-only vs. spectra+ALS vs. ALS-only.
- **Question answered**: Does structural information complement spectral features? Can ALS alone classify forest type?

### Phase 4: Pixel-Level and Spatial Models (Days 4–5, if time permits)

**Experiment 8: Pixel-level classification**
- Extract all pixels from each plot tile. Each pixel = one spectrum.
- Train RF on pixel-level spectra with stand-level LOSO-CV splits.
- Compare to stand-level summary results.
- **Question answered**: Does pixel-level modeling outperform stand aggregation? How much variation exists within stands?

**Experiment 9: 1D CNN on pixel spectra**
- Simple 3-layer 1D CNN with dropout and batch normalization.
- Same pixel data and stand-level LOSO-CV splits.
- Compare to pixel-level RF.
- **Question answered**: Do learned spectral features outperform RF on raw spectra?

**Experiment 10 (optional): Patch-based 2D/3D CNN**
- Extract small patches (e.g., 15×15 pixels × B bands) from each plot tile.
- Train a 3D CNN (e.g., HybridSN or a small custom architecture).
- Stand-level LOSO-CV splits.
- **Question answered**: Does spatial context improve classification beyond spectral-only models?

---

## 5. Validation Design

### Primary: Leave-one-site-out (LOSO-CV)

4 folds. Each fold holds out one entire site and trains on the other 3. This directly measures whether the model generalizes to an unseen biome/country/acquisition.

**Implementation**: `LeaveOneGroupOut()` with site as the group variable.

**Why this is mandatory**: Roberts et al. (2017, Ecography) established that block CV is required when spatial, temporal, or hierarchical structure exists. Meyer & Pebesma (2022, Nature Communications) demonstrated that random splits of spatially structured data produce optimistically biased performance. Ploton et al. (2020) showed that spatial validation can reveal that models with apparent R²>0.5 actually have near-null predictive power.

**Limitation**: 4 folds means high variance. One fold holds out Hyytiälä (28 stands — training on only 30 stands), another holds out Bily Kriz (7 stands — testing on very few). Report per-fold results, not just the mean.

### Secondary: Within-site stand-level splits

For within-site experiments (e.g., "can we classify forest types within Hyytiälä alone?"), use leave-one-stand-out or stratified K-fold by stand within that site. This measures within-biome performance and serves as an upper bound.

### Invalid: Random pixel splits

Randomly splitting pixels across train/test, ignoring stand and site membership, produces inflated accuracy from spatial autocorrelation. This is scientifically indefensible.

### Shortcut detection protocol

1. Run forest-type classification with LOSO-CV → accuracy_type.
2. Run site classification (4-class, same features, random stand-level CV) → accuracy_site.
3. If accuracy_site >> accuracy_type: the model primarily encodes site, not type.
4. Produce a PCA/t-SNE plot colored by (a) forest type and (b) site. If sites form tighter clusters than types, there is confounding.
5. Report both honestly.

---

## 6. Data-Use Recommendations

### Start with: Analysis-ready plot tiles (100 m × 100 m)

These are the pre-extracted, atmospherically corrected HDRF plot tiles. One file per stand, ready to load.
- Manageable size (~58 tiles × ~200×200 pixels × ~45–145 bands).
- Avoid dealing with full-site mosaics, flight-line overlaps, and geo-referencing.
- Extract stand-level summary spectra from these tiles.

### Use CASI data first

- CASI (VIS-NIR, 0.5 m) has higher spatial resolution and is a single sensor product.
- Start with CASI-only experiments. Add SASI later as an ablation.
- The VIS-NIR region contains the primary vegetation signals (chlorophyll absorption, red-edge, NIR plateau).

### Reserve for later: Full-site flightlines

- Needed only if you want to classify areas outside the 58 stands, or need flight-line-level processing.
- Each site is ~4 km × 4 km — potentially hundreds of GB per site.
- Unnecessary for the core research question.

### Reserve for later: ALS point clouds

- Derive structural metrics (canopy height, cover fraction) per stand from the 1 m DEMs.
- Do NOT use as primary features — only for the fusion ablation.
- The DEMs may be simpler to work with than raw point clouds.

### Avoid: Raw TLS point clouds

- Terrestrial laser scanning data are available but not relevant to the airborne classification task.
- Very large (2800 individual trees). Only useful for radiative transfer modeling validation.

### Pragmatic data access plan

1. Download only the CASI plot tiles + metadata initially (~tens of GB, not hundreds).
2. Run all Phase 1–3 experiments on these tiles.
3. Download SASI tiles for the CASI+SASI ablation.
4. Download ALS DEMs for the fusion ablation.
5. Full flightlines only if doing site-wide pixel mapping beyond the 58 stands.

---

## 7. Reusable Pipeline Design

```
pipeline/
├── config/
│   └── experiment.yaml              # All parameters: paths, bands, models, CV strategy
├── data/
│   ├── raster_loader.py             # Load ENVI BSQ/HDR and GeoTIFF (rasterio)
│   ├── metadata_loader.py           # Load stand metadata (CSV/GeoJSON)
│   ├── als_loader.py                # Load ALS DEMs and derive structural features
│   └── splits.py                    # LOSO-CV, within-site stand-level CV
├── preprocessing/
│   ├── band_selection.py            # Exclude water vapor bands, select CASI/SASI/merged
│   ├── normalization.py             # Per-band standardization, SNV, continuum removal
│   ├── resampling.py                # Resample SASI to CASI resolution (or vice versa)
│   └── nodata_mask.py               # Mask invalid/nodata pixels within tiles
├── features/
│   ├── stand_summary.py             # Mean/median/std spectrum per stand (from tile pixels)
│   ├── spectral_indices.py          # NDVI, NDWI, red-edge indices from band math
│   ├── pca_features.py              # PCA reduction
│   ├── patch_extractor.py           # Extract (H×W×B) patches for 2D/3D models
│   └── als_features.py              # Canopy height stats, cover fraction from DEMs
├── models/
│   ├── classical.py                 # LR, RF, SVM, GB, PLS-DA (sklearn pipelines)
│   ├── cnn_1d.py                    # 1D CNN for pixel-level spectral classification
│   └── cnn_3d.py                    # 3D spatial-spectral CNN for patch classification
├── evaluation/
│   ├── metrics.py                   # OA, balanced acc, macro F1, kappa, confusion matrix
│   ├── per_fold_report.py           # Per-site-fold detailed results
│   └── shortcut_analysis.py         # Site classifier vs. type classifier comparison
├── experiments/
│   └── runner.py                    # Config → data → features → model → evaluate → save
├── outputs/
│   ├── metrics/                     # CSV results per experiment
│   ├── figures/                     # Confusion matrices, spectral plots, PCA biplots
│   └── models/                      # Saved model artifacts
└── notebooks/
    ├── 01_data_inspection.ipynb     # Visualize tiles, check metadata
    ├── 02_baseline_results.ipynb    # Run and visualize Experiments 1–4
    └── 03_ablations.ipynb           # Run and visualize Experiments 5–10
```

### Key design differences from the existing code

| Existing code problem | Pipeline solution |
|----------------------|-------------------|
| Monolithic 500-line script | Modular: each concern in its own file |
| Operates on peatland CSV | Operates on ENVI/GeoTIFF raster tiles |
| No spatial dimension | Supports pixel-level, patch-level, and stand-level |
| Hardcoded Windows paths | Config-driven YAML paths |
| EDA mixed with modeling | EDA in notebooks, modeling in pipeline |
| Cannot support deep learning | Models module includes CNN architectures |

---

## 8. Report Framing

### Title options

1. "Cross-Site Generalization of Airborne Hyperspectral Forest Type Classification Across European Biomes"
2. "Can Airborne Hyperspectral Spectra Predict Forest Type Across Boreal, Hemiboreal, and Temperate Sites?"
3. "Spectral Recovery of Forest Type and Location from Multi-Site Airborne CASI and SASI Imagery"

### Abstract angle

The novelty is the **cross-site generalization evaluation** on a new, high-quality multi-site airborne dataset. Frame around:
1. The question: Can hyperspectral reflectance generalize forest type classification across biomes?
2. The data: 58 stands across 4 sites in 3 countries, CASI+SASI imagery, LOSO-CV.
3. The finding: [results will determine] — either "spectral features generalize surprisingly well" or "site-specific atmospheric/phenological effects limit cross-site transfer."
4. The implications: What this means for operational forest mapping from airborne HSI.

### Related work (3 threads)

1. **Hyperspectral forest classification**: Dalponte & Coomes (2016), Fassnacht et al. (2016), Immitzer et al. (2012) — tree species and forest type mapping from airborne HSI. Establishes that HSI works within a site but cross-site transfer is rarely evaluated.

2. **Spatial validation and domain shift**: Roberts et al. (2017), Ploton et al. (2020), Meyer & Pebesma (2022) — why random splits overestimate performance, why grouped/spatial validation is necessary. Tuia et al. (2016) on domain adaptation for remote sensing.

3. **The FREEDLES dataset**: Rautiainen et al. (2024, ESSD) — describes the data collection, sensors, preprocessing. Establishes the spectral-structural diversity across the 4 sites. Position your work as the first classification study on this dataset.

### Methodology phrasing

Describe the study as: "We use analysis-ready plot-level hyperspectral reflectance tiles (CASI: 382–1052 nm, 0.5 m; SASI: 958–2443 nm, 1.25 m) from 58 forest stands across four European sites. We evaluate spectral-only classification using classical machine learning (PLS-DA, RF, SVM) under leave-one-site-out cross-validation, with ablations on spectral region, dimensionality, and ALS fusion."

### Writing results honestly

**If stand-level results are weak** (likely — n=58, 4 folds is tough): Frame the study as an investigation of the generalization challenge, not a failure. "Stand-level classification achieves X% balanced accuracy under LOSO-CV, compared to Y% under within-site random splits. The performance gap of Y−X percentage points quantifies the domain shift across biomes." This is a finding, not a failure.

**If pixel-level results improve substantially**: "Pixel-level classification outperforms stand-level summaries, suggesting that sub-stand spectral variability contains discriminative information that is lost by aggregation."

**If CASI outperforms CASI+SASI**: "The addition of SWIR bands does not improve cross-site generalization, suggesting that the VIS-NIR region alone captures the primary forest-type signatures."

### Limitations (state honestly)

1. Four sites spanning three biomes means forest type is partially confounded with site identity. A model predicting "coniferous" may partly be predicting "Hyytiälä."
2. Leave-one-site-out with 4 folds produces high-variance estimates. Per-fold results vary substantially.
3. The two Czech sites (Lanzhot, Bily Kriz) are geographically close but ecologically distinct. Holding out one still leaves the other in training.
4. Atmospheric correction (ATCOR-4) and vicarious calibration were done per-site, which may introduce site-specific residual calibration differences.
5. Temporal differences: flights were in 2019, but different months (July vs. September). Phenological state differences may confound spectral signatures.

---

## Key References

- Rautiainen, M., Hovi, A., Schraik, D., Hanuš, J., Lukeš, P., Lhotáková, Z., & Homolová, L. (2024). A spectral–structural characterization of European temperate, hemiboreal, and boreal forests. *Earth System Science Data*, 16, 5069–5098.
- Meyer, H. & Pebesma, E. (2022). Machine learning-based global maps of ecological variables and the challenge of assessing them. *Nature Communications*, 13, 2208.
- Ploton, P., et al. (2020). Spatial validation reveals poor predictive performance of large-scale ecological mapping models. *Nature Communications*, 11, 4540.
- Roberts, D.R., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929.
- Tuia, D., et al. (2016). Domain adaptation for the classification of remote sensing data. *IEEE GRSM*, 4(2), 41–57.
- Fassnacht, F.E., et al. (2016). Review of studies on tree species classification from remotely sensed data. *Remote Sensing of Environment*, 186, 64–87.
- Dalponte, M. & Coomes, D. (2016). Tree-centric mapping of forest carbon density from airborne laser scanning and hyperspectral data. *Methods in Ecology and Evolution*, 7(10), 1236–1245.
- Immitzer, M., Atzberger, C., Koukal, T. (2012). Tree species classification with random forest using very high spatial resolution 8-band WorldView-2 satellite data. *Remote Sensing*, 4(9), 2661–2693.
- Paoletti, M.E., et al. (2019). Deep learning classifiers for hyperspectral imaging: A review. *ISPRS Journal*, 158, 279–317.
- Mountrakis, G., Im, J., Ogole, C. (2011). Support vector machines in remote sensing: A review. *ISPRS Journal*, 66(3), 247–259.
- Ahmad, M., et al. (2024). A comprehensive survey for hyperspectral image classification: The evolution from conventional to transformers and Mamba models. *arXiv:2404.14955*.

---

## Summary of Immediate Actions

1. **Discard**: All peatland CSV code (`peatland_hyperspectral_analysis.py`, `peatland_eda.ipynb`). These operate on the wrong dataset.
2. **Download**: The CASI plot tiles + metadata from the Etsin/Fairdata repository. Start with tiles only — not the full flightlines.
3. **Inspect**: Load one tile. Verify dimensions, band count, wavelengths, and nodata handling. Plot a false-color composite.
4. **Build**: The stand-level summary spectrum pipeline (mean of all valid pixels per tile → n=58 feature vectors).
5. **Run**: Experiments 0–4 (stand-level baselines with LOSO-CV). This produces your core results table.
6. **Ablate**: Experiments 5–7 (band selection, dimensionality, ALS fusion).
7. **Extend**: If time permits, Experiments 8–10 (pixel-level, 1D CNN, spatial-spectral CNN).
8. **Write**: Frame the report around cross-site generalization and the LOSO-CV evaluation.
