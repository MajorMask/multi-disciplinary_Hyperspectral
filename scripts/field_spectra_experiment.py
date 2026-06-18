"""
Field Spectroscopy Classification Experiment (BilyKriz dataset)

Uses the field-measured canopy reflectance spectra (350-2500 nm, 2151 bands,
15 measurements per stand) from the Excel spreadsheet to answer the same
research question as the airborne CASI experiment:

  "Can spectral reflectance distinguish coniferous from broadleaved forest
   and generalise across sites (Jarvselja, BilyKriz, Lanzhot)?"

Two feature modes:
  (a) stand-level means  — 31 samples, 1 per stand
  (b) measurement-level  — 465 samples (15 per stand), GroupKFold by site

Water-vapour bands are excluded (same ranges as airborne analysis).
PCA-20 is applied (2151 bands → 20 components) to regularise the RF.

Output: outputs/ablations/field_spectra.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

EXCEL = Path(
    "C:/Users/aggarwm1/Videos/multi-disciplinary_Hyperspectral/model/data"
    "/Hyperspectral/Airborne_data"
    "/DatasetOfTreeCanopyStructureUnderstoryReflectanceSpectraAndFractionalCoverInHemiborealAndTemperateForestAreasInEstoniaAndCzechRepublic_V2.xlsx"
)
OUT_DIR = Path("C:/Users/aggarwm1/Videos/multi-disciplinary_Hyperspectral/outputs/ablations")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Water vapour / noise bands to exclude (same as airborne analysis)
EXCLUDE_NM = [(895, 1003), (1092, 1168), (1302, 1528), (1737, 2038)]

RF_PARAMS = dict(n_estimators=500, class_weight="balanced",
                 min_samples_leaf=2, n_jobs=-1, random_state=42)


# ---- Load Excel data -------------------------------------------------------
print("Loading Excel data...")
xl = pd.ExcelFile(EXCEL)
sp_raw = xl.parse("Spectra")
stands_raw = xl.parse("Stand_characteristics", header=1)

# Extract wavelengths (columns named WL350, WL351, ... WL2500)
wl_cols = [c for c in sp_raw.columns if str(c).startswith("WL")]
wavelengths = np.array([int(str(c).replace("WL", "")) for c in wl_cols])

# Build exclusion mask
def valid_band_mask(wl, exclude):
    mask = np.ones(len(wl), dtype=bool)
    for lo, hi in exclude:
        mask &= ~((wl >= lo) & (wl <= hi))
    return mask

band_mask = valid_band_mask(wavelengths, EXCLUDE_NM)
valid_wl = wavelengths[band_mask]
print(f"Wavelengths: {len(wavelengths)} total, {band_mask.sum()} after exclusion")

# ---- Assign forest types ---------------------------------------------------
def assign_type(row):
    pct_conif = row["%-Spruce"] + row["%-Pine"]
    if pct_conif > 70:
        return "coniferous"
    if row["%-Broadleaf"] > 70:
        return "broadleaved"
    return "mixed"

site_map = {"HB": "Jarvselja", "TM": "BilyKriz", "TF": "Lanzhot"}
stands_raw["forest_type"] = stands_raw.apply(assign_type, axis=1)
stands_raw["site"] = stands_raw["Study site"].map(site_map)

stand_nr_to_meta = stands_raw.set_index("Stand nr")[["ID", "forest_type", "site"]].to_dict("index")
print("\nStand distribution:")
print(stands_raw.groupby(["site", "forest_type"]).size().to_string())
print(f"\nTotal stands: {len(stands_raw)}")

# ---- Build measurement-level feature matrix --------------------------------
# Spectra rows have 'Stand nr' (1-31) and 'Position' (1-15)
sp_raw = sp_raw.dropna(subset=["Stand nr"])
sp_raw["Stand nr"] = sp_raw["Stand nr"].astype(int)

X_meas = sp_raw[wl_cols].values[:, band_mask].astype(np.float32)
stand_nrs = sp_raw["Stand nr"].values.astype(int)

y_meas   = np.array([stand_nr_to_meta[n]["forest_type"] for n in stand_nrs])
sites_meas = np.array([stand_nr_to_meta[n]["site"] for n in stand_nrs])
ids_meas   = np.array([stand_nr_to_meta[n]["ID"] for n in stand_nrs])

# ---- Stand-level means (aggregate 15 meas per stand to 1 feature vector) --
stand_nrs_unique = sorted(set(stand_nrs))
X_stand, y_stand, sites_stand, ids_stand = [], [], [], []
for nr in stand_nrs_unique:
    mask_nr = stand_nrs == nr
    X_stand.append(X_meas[mask_nr].mean(axis=0))
    y_stand.append(stand_nr_to_meta[nr]["forest_type"])
    sites_stand.append(stand_nr_to_meta[nr]["site"])
    ids_stand.append(stand_nr_to_meta[nr]["ID"])

X_stand = np.array(X_stand)
y_stand = np.array(y_stand)
sites_stand = np.array(sites_stand)

print(f"\nStand-level matrix: {X_stand.shape}")
print(f"Measurement-level matrix: {X_meas.shape}")


# ---- CV function -----------------------------------------------------------
logo = LeaveOneGroupOut()

def loso_cv(X, y, groups, label="", pca_n=20):
    class_names = sorted(np.unique(y).tolist())
    fold_ba, fold_f1, fold_info = [], [], []

    for tr, te in logo.split(X, y, groups):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        site_test = np.unique(groups[te])[0]

        # Standardise on train
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)

        # PCA on train
        n = min(pca_n, Xtr.shape[1], Xtr.shape[0] - 1)
        pca = PCA(n_components=n, random_state=42)
        Xtr = pca.fit_transform(Xtr)
        Xte = pca.transform(Xte)

        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(Xtr, ytr)
        yp = rf.predict(Xte)

        ba = balanced_accuracy_score(yte, yp)
        f1 = f1_score(yte, yp, average="macro", zero_division=0)
        fold_ba.append(ba)
        fold_f1.append(f1)
        fold_info.append({
            "test_site": site_test,
            "balanced_accuracy": float(ba),
            "macro_f1": float(f1),
            "n_test": int(len(te)),
            "class_dist": {c: int((yte == c).sum()) for c in class_names},
        })

    result = {
        "mean_ba": float(np.mean(fold_ba)),
        "std_ba":  float(np.std(fold_ba)),
        "mean_f1": float(np.mean(fold_f1)),
        "fold_ba": [float(v) for v in fold_ba],
        "folds": fold_info,
        "n_samples": len(y),
        "n_classes": len(class_names),
        "class_names": class_names,
    }
    print(f"  {label}: BA = {result['mean_ba']:.3f} ± {result['std_ba']:.3f}  |  per-fold: {[f'{v:.3f}' for v in fold_ba]}")
    return result


# ---- Run experiments -------------------------------------------------------
results = {}

print("\n=== Stand-level (31 stands, 1 mean spectrum per stand) ===")
results["stand_level_all"]    = loso_cv(X_stand, y_stand, sites_stand, "all classes", pca_n=20)

# Binary only (no mixed class, but there are none anyway)
has_mixed = (y_stand == "mixed").any()
if has_mixed:
    mask_bin = y_stand != "mixed"
    results["stand_level_binary"] = loso_cv(
        X_stand[mask_bin], y_stand[mask_bin], sites_stand[mask_bin], "binary", pca_n=20)
else:
    print("  (no mixed class — all classes = binary)")
    results["stand_level_binary"] = results["stand_level_all"]

print("\n=== Measurement-level (465 spectra, 15 per stand, grouped by site) ===")
results["meas_level_all"] = loso_cv(X_meas, y_meas, sites_meas, "all classes", pca_n=20)


# ---- Per-site breakdown for best experiment --------------------------------
print("\n=== Per-site fold detail (stand-level) ===")
for fold in results["stand_level_all"]["folds"]:
    print(f"  Test site: {fold['test_site']:12s}  BA={fold['balanced_accuracy']:.3f}  "
          f"n={fold['n_test']}  dist={fold['class_dist']}")


# ---- Save results ----------------------------------------------------------
out_path = OUT_DIR / "field_spectra.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")


# ---- Summary table ---------------------------------------------------------
print("\n" + "=" * 65)
print("SUMMARY — Field Spectroscopy LOSO-CV (350-2500 nm)")
print("=" * 65)
print(f"{'Experiment':<35}  {'Mean BA':>8}  {'±std':>6}")
print("-" * 65)
for k, r in results.items():
    print(f"  {k:<33}  {r['mean_ba']:>8.3f}  {r['std_ba']:>6.3f}")
print("=" * 65)
print(f"\nReference — Airborne CASI baseline (48 bands, 4 sites): BA = 0.621")
print(f"Reference — Airborne CASI binary   (48 bands, 4 sites): BA = 0.932")
