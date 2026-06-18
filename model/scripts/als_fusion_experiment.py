"""
ALS Fusion Ablation (Experiment 7)

Compares three feature sets with LOSO-CV using RandomForest:
  (a) ALS-only      — 8 structural metrics
  (b) CASI+ALS      — 40 spectral bands + 8 structural metrics
  (c) CASI-only     — 40 spectral bands (baseline, loaded from prior results)

Also runs binary (coniferous vs. broadleaved) for each feature set.
Outputs: outputs/ablations/als_fusion.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut

# ---- paths ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))

CFG_PATH = ROOT / "config" / "default.yaml"
ALS_FEATS = Path("C:/Users/aggarwm1/Videos/multi-disciplinary_Hyperspectral/outputs/als_features.csv")
META_CSV  = ROOT / "data" / "stand_metadata.csv"
OUT_DIR   = Path("C:/Users/aggarwm1/Videos/multi-disciplinary_Hyperspectral/outputs/ablations")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- load config and spectral features ------------------------------------
with open(CFG_PATH) as f:
    cfg = yaml.safe_load(f)

# Import pipeline modules
from model.src.dataio.raster_loader import load_all_tiles
from model.src.preprocessing.band_selection import get_valid_band_mask, apply_band_selection
from model.src.preprocessing.normalization import apply_normalization
from model.src.features.stand_summary import build_stand_feature_matrix

sensor = "casi"
tile_dir = Path(
    "C:/Users/aggarwm1/Videos/multi-disciplinary_Hyperspectral/model/data"
    "/Hyperspectral/Airborne_data/Airborne_hyperspectral/Analysis_ready_subsets/CASI"
)
pattern = cfg["data"].get("tile_filename_pattern", "*.tif")
nodata = cfg["preprocessing"]["nodata_value"]
scale  = cfg["preprocessing"].get("reflectance_scale_factor", 1.0)

print("Loading CASI tiles...")
tiles = load_all_tiles(str(tile_dir), pattern=pattern, nodata_value=nodata, min_valid_fraction=0.5)

# Inject wavelengths and scale
wl_arr = np.array(cfg["preprocessing"]["casi_wavelengths_nm"])
for t in tiles:
    if np.array_equal(t.wavelengths, np.arange(t.n_bands)):
        if len(wl_arr) == t.n_bands:
            t.wavelengths = wl_arr
    if t.stand_id.endswith("_CASI"):
        t.stand_id = t.stand_id[:-5]
    t.image = t.image / scale

wavelengths = tiles[0].wavelengths
exclude_ranges = [tuple(r) for r in cfg["preprocessing"].get("exclude_wavelength_ranges_nm", [])]
band_mask = get_valid_band_mask(wavelengths, exclude_ranges)

X_spec, stand_ids = build_stand_feature_matrix(tiles, method="mean", band_mask=band_mask)
print(f"Spectral features: {X_spec.shape}")

# ---- load metadata --------------------------------------------------------
meta = pd.read_csv(META_CSV)
id_to_type = dict(zip(meta["stand_id"], meta["forest_type"]))
id_to_site  = dict(zip(meta["stand_id"], meta["site"]))

y_all   = np.array([id_to_type[s] for s in stand_ids])
sites   = np.array([id_to_site[s]  for s in stand_ids])

# ---- load ALS features ----------------------------------------------------
als_df = pd.read_csv(ALS_FEATS)
als_feat_cols = ["h_max", "h_mean", "h_std", "h_p25", "h_p50", "h_p75", "h_p95", "cover_fraction"]
als_map = als_df.set_index("stand_id")[als_feat_cols].to_dict("index")

# Build ALS feature matrix aligned to stand_ids
X_als = np.array([
    [als_map[s][c] for c in als_feat_cols] if s in als_map else [np.nan]*len(als_feat_cols)
    for s in stand_ids
])
missing_als = np.isnan(X_als).any(axis=1)
print(f"Stands with ALS: {(~missing_als).sum()}/{len(stand_ids)}")

# ---- CV setup -------------------------------------------------------------
logo = LeaveOneGroupOut()
rf_params = dict(n_estimators=500, class_weight="balanced", min_samples_leaf=2,
                 n_jobs=-1, random_state=42)

def loso_cv(X, y, groups, binary=False):
    """Run LOSO-CV with RF. binary=True drops 'mixed' class."""
    if binary:
        mask = y != "mixed"
        X, y, groups = X[mask], y[mask], groups[mask]

    fold_ba, fold_f1, preds = [], [], []
    class_names = sorted(np.unique(y).tolist())

    for tr, te in logo.split(X, y, groups):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        # Standardise on train only
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd

        rf = RandomForestClassifier(**rf_params)
        rf.fit(Xtr, ytr)
        yp = rf.predict(Xte)

        ba = balanced_accuracy_score(yte, yp)
        f1 = f1_score(yte, yp, average="macro", zero_division=0)
        fold_ba.append(ba)
        fold_f1.append(f1)

    return {
        "mean_ba": float(np.mean(fold_ba)),
        "std_ba":  float(np.std(fold_ba)),
        "mean_f1": float(np.mean(fold_f1)),
        "fold_ba": [float(v) for v in fold_ba],
        "n_classes": len(class_names),
        "n_samples": len(y),
    }

# ---- Run experiments -------------------------------------------------------
results = {}

# Filter to stands that have ALS data
has_als = ~missing_als
X_spec_als   = X_spec[has_als]
X_als_als    = X_als[has_als]
y_als        = y_all[has_als]
sites_als    = sites[has_als]

# Combine spectral + ALS
X_fused = np.concatenate([X_spec_als, X_als_als], axis=1)

print("\nRunning 3-class experiments...")
results["als_only_3class"]    = loso_cv(X_als_als, y_als, sites_als)
results["casi_only_3class"]   = loso_cv(X_spec_als, y_als, sites_als)
results["casi_als_3class"]    = loso_cv(X_fused,  y_als, sites_als)

print("Running binary (conif. vs. broadleaved) experiments...")
results["als_only_binary"]    = loso_cv(X_als_als, y_als, sites_als, binary=True)
results["casi_only_binary"]   = loso_cv(X_spec_als, y_als, sites_als, binary=True)
results["casi_als_binary"]    = loso_cv(X_fused,  y_als, sites_als, binary=True)

# Save
out_path = OUT_DIR / "als_fusion.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {out_path}")
print("\n=== 3-Class LOSO-CV (Balanced Accuracy) ===")
for k in ["als_only_3class", "casi_only_3class", "casi_als_3class"]:
    r = results[k]
    print(f"  {k:25s}: {r['mean_ba']:.3f} ± {r['std_ba']:.3f}")

print("\n=== Binary LOSO-CV (Conif. vs. Broadleaved) ===")
for k in ["als_only_binary", "casi_only_binary", "casi_als_binary"]:
    r = results[k]
    print(f"  {k:25s}: {r['mean_ba']:.3f} ± {r['std_ba']:.3f}")
