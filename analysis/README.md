# Etsin Hyperspectral Research Pipeline

This is a modular Python codebase for investigating whether **forest location and forest type can be recovered from airborne hyperspectral spectra alone**.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up your data


Place data in a structure like:

```
data/
  metadata.csv           # plot-level labels
  plot_tiles/            # GeoTIFF .tif files per plot
```

The metadata CSV must include columns for:
- `tile_name` (or similar): identifier matching GeoTIFF filenames
- `site_id`: grouping variable for cross-validation (e.g., acquisition site)
- `forest_type`: target label for classification

### 3. Run a baseline experiment

Update the YAML config to match your data paths:

```bash
# Edit configs/baseline_experiment.yaml
python analysis/scripts/run_experiment.py --config configs/baseline_experiment.yaml
```

Results are saved to `outputs/`.

## Project Structure

```
src/
  dataio/               # ENVI/GeoTIFF loaders, metadata I/O
    raster.py
    metadata.py
  datasets/             # Plot-level dataset abstraction
    dataset.py
  preprocessing/        # Spectral normalization, band masking
    spectra.py
  features/             # Feature engineering (PCA, texture, etc.)
    feature_engineering.py
  models/               # Baseline and deep learning models
    baselines.py        # LR, RF, GB, PLS-DA
    deep.py             # 1D/2D/3D CNNs (optional)
  evaluation/           # Metrics and reporting
    metrics.py
  experiments/          # Experiment runner and orchestration
    runner.py
  utils/                # Config loading, path utilities
    config.py

configs/                # YAML experiment configurations
scripts/                # Runnable scripts (run_experiment.py)
notebooks/              # Jupyter notebooks for analysis and visualization
outputs/                # Experiment results (metrics, predictions, confusion matrices)
```

## Data Loading (`src/dataio/`)

### ENVI flightlines

```python
from src.dataio.raster import read_envi

raster = read_envi(Path("data/flightline.bsq"))
image = raster.image  # shape: (bands, rows, cols)
wavelengths = raster.wavelengths
```

### GeoTIFF plot tiles

```python
from src.dataio.raster import read_geotiff

raster = read_geotiff(Path("data/plot_01.tif"))
```

### Metadata

```python
from src.dataio.metadata import load_metadata

df = load_metadata(Path("data/metadata.csv"))
```

## Dataset Abstraction (`src/datasets/`)

The `PlotTileDataset` class handles the full pipeline:

```python
from src.datasets.dataset import PlotTileDataset

dataset = PlotTileDataset(
    tile_dir="data/plot_tiles",
    metadata_path="data/metadata.csv",
    tile_id_column="tile_name",
    plot_id_column="plot_id",
    site_id_column="site_id",
    label_column="forest_type",
)

# Compute mean spectra per plot
summary_df = dataset.summarize_tiles(summary_method="mean", normalize=True)

# Build feature matrix with labels and group assignments
X, y, groups, labels = dataset.build_feature_matrix(
    feature_mode="mean",
    normalize=True,
)
```

## Preprocessing (`src/preprocessing/`)

```python
from src.preprocessing.spectra import (
    remove_noisy_bands,
    normalize_spectra,
    aggregate_plot_spectra,
)

# Remove water-absorption bands
image, wavelengths = remove_noisy_bands(image, wavelengths)

# Normalize spectra
X_norm = normalize_spectra(X, method="standard")

# Aggregate tile to plot-level mean spectrum
spectrum_mean = aggregate_plot_spectra(image, summary="mean")
```

## Baseline Models (`src/models/baselines.py`)

```python
from src.models.baselines import build_baseline_pipelines

pipelines = build_baseline_pipelines()
# Returns dict: "LogisticRegression", "RandomForest", "GradientBoosting"
```

## Evaluation (`src/evaluation/`)

```python
from src.evaluation.metrics import classification_metrics, save_classification_report
from sklearn.model_selection import LeaveOneGroupOut

cv = LeaveOneGroupOut()
for train_idx, test_idx in cv.split(X, y, groups=groups):
    clf.fit(X[train_idx], y[train_idx])
    y_pred = clf.predict(X[test_idx])
    metrics = classification_metrics(y[test_idx], y_pred, labels=labels)
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.3f}")
    print(f"Macro F1: {metrics['macro_f1']:.3f}")
```

## Experiments (`src/experiments/runner.py`)

Define experiments via YAML configuration:

```yaml
experiment_name: etsin_baseline_plot_level
metadata_path: ../data/metadata.csv
tile_dir: ../data/plot_tiles
tile_id_column: tile_name
plot_id_column: plot_id
site_id_column: site_id
label_column: forest_type
feature_method: mean
normalize: true
normalization_method: standard
cv_strategy: leave-one-site-out
model_names:
  - LogisticRegression
  - RandomForest
  - GradientBoosting
  - PCA
output_dir: ../outputs/baseline_experiment
```

Then run:

```bash
python analysis/scripts/run_experiment.py --config configs/baseline_experiment.yaml
```

The experiment runner:
1. Loads plot metadata and tile summaries
2. Builds the feature matrix with labels
3. Runs cross-validation (leave-one-site-out or group K-fold)
4. Saves per-model metrics, confusion matrices, and predictions
5. Produces a summary JSON file

## Research Design

### Problem Formulation

- **Input**: reflectance spectra (full or subset of wavelengths)
- **Targets**:
  - forest-type classification (e.g., bog, fen, spruce forest)
  - site/plot identity recovery (as a proxy for location)
- **Prediction unit**: plot-level (aggregated mean/median spectra)
- **Validation**: grouped cross-validation by site to avoid spatial leakage

### Method Progression

1. **Baselines** (plot-level summary spectra):
   - Logistic regression (linear classifier)
   - Random forest (ensemble, nonlinear)
   - Gradient boosting (boosted ensemble)
   - PCA + logistic regression (dimensionality reduction)

2. **Advanced** (not yet implemented, can be added):
   - 1D CNN on spectral vectors
   - 2D CNN on selected band composites
   - Spectral-spatial CNNs using patch tensors

### Validation Strategy

- **Leave-one-site-out** evaluation: train on all sites except one, test on the held-out site.
  This measures generalization to new acquisition sites.
- **Group K-fold** option: stratified folds within site groups.
- **Metrics**:
  - Overall accuracy
  - Balanced accuracy (handles class imbalance)
  - Macro F1 (unweighted average per class)
  - Per-class precision, recall, support
  - Cohen's Kappa (inter-rater agreement-like metric)
  - Confusion matrix

### Key Assumptions & Caveats

1. **Spectral-only**: No spatial features (pixel neighborhood, texture).
   Performance may improve with spatial-spectral CNNs.

2. **Plot-level aggregation**: Assumes within-plot homogeneity.
   Pixel-level modeling may reveal sub-plot variation.

3. **Grouped validation is essential**: Without grouping by site, train/test leakage from nearby acquisitions overestimates generalization.

4. **Noisy wavelengths**: Water-absorption bands (1330–1550, 1761–2025, 2310–2501 nm) are removed by default.

## Extending the Pipeline

### Add a new baseline model

Edit `src/models/baselines.py`:

```python
def build_baseline_pipelines(random_state: int = 42):
    return {
        "LogisticRegression": Pipeline([...]),
        "YourNewModel": Pipeline([...]),  # Add here
    }
```

### Use a different aggregation method

In your experiment config, change `feature_method`:

```yaml
feature_method: median  # or "std"
```

### Compare preprocessing options

Create a new config file:

```yaml
experiment_name: etsin_without_noisy_bands
...
# The code will automatically remove noisy bands when loading
```

### Add new labels

Ensure your metadata CSV has the target column, then update the config:

```yaml
label_column: tree_species  # instead of "forest_type"
```

### Use pixel-level spectra (advanced)

Implement pixel extraction in `src/datasets/dataset.py` using `compute_patch_tensors()` from `src/preprocessing/spectra.py`, then adapt the experiment runner to handle 3D tensors instead of 2D feature matrices.

## Optional: PyTorch Models

Deep learning models are in `src/models/deep.py` and require PyTorch:

```bash
pip install torch  # or pip install torch torchvision torchaudio
```

They are not integrated into the baseline runner yet. To use them, you would:

1. Extract patch tensors from tiles
2. Build a PyTorch DataLoader
3. Train with standard deep learning loops

A tutorial notebook in `notebooks/` will demonstrate this later.

## Reproducibility

- All random seeds are fixed at `random_state=42` in model definitions
- Config files are saved alongside results
- Predictions and metrics are saved in both CSV and JSON formats
- Confusion matrices are tabulated for visual inspection

## References & Literature

This work is grounded in:

- **Hyperspectral remote sensing**: Goetz & Srivastava (1985), Govender et al. (2007), Heylen et al. (2015)
- **Forest type / species mapping**: Immitzer et al. (2012), Ghosh et al. (2014), Dalponte & Coomes (2016)
- **Grouped validation & domain generalization**: Zhu et al. (2021), Tuia et al. (2016), Valada & Burgard (2020)
- **Spectral band selection**: Hughes (1968), Pal & Foody (2010)

The research explores the "curse of dimensionality" in hyperspectral classification and the necessity of grouped evaluation to detect domain shift across real-world acquisition sites.

## Questions & Support

- **Data not loading?** Check file formats and metadata column names.
- **Cross-validation failing?** Ensure `site_id_column` is available in metadata.
- **Results look suspicious?** Enable leave-one-site-out to catch spatial leakage.

---

**Last updated**: May 7, 2026  
**Status**: Baseline pipeline implemented and tested. Ready for first experiment.
