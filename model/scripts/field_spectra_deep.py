"""
Field Spectroscopy — Extended Analysis

Experiments:
  1. Leave-one-stand-out at Jarvselja (the only site with both forest types)
     → pure within-site generalization, no site-bias, 13 stands LOSO
  2. Cross-site: Train on all 3 sites, test leave-one-SITE-out
     → compare to airborne CASI LOSO results
  3. Train on JS+LZ → predict BK  (hardest transfer: boreal to temperate conifer)
  4. Train on JS+BK → predict LZ  (cross-site broadleaved)
  5. Spectral separability: between-class vs within-class distance per wavelength
  6. Full 2151-band importance (which regions drive the classification)

Output: outputs/ablations/field_spectra_deep.json
        outputs/figures/field_spectra_*.png
"""

import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler

EXCEL = Path(
    "C:/Users/aggarwm1/Videos/multi-disciplinary_Hyperspectral/model/data"
    "/Hyperspectral/Airborne_data"
    "/DatasetOfTreeCanopyStructureUnderstoryReflectanceSpectraAndFractionalCoverInHemiborealAndTemperateForestAreasInEstoniaAndCzechRepublic_V2.xlsx"
)
OUT_DIR  = Path("C:/Users/aggarwm1/Videos/multi-disciplinary_Hyperspectral/outputs/ablations")
FIG_DIR  = Path("C:/Users/aggarwm1/Videos/multi-disciplinary_Hyperspectral/outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_NM = [(895, 1003), (1092, 1168), (1302, 1528), (1737, 2038)]
RF_PARAMS  = dict(n_estimators=500, class_weight="balanced",
                  min_samples_leaf=2, n_jobs=-1, random_state=42)

# ---- Load ----------------------------------------------------------------
xl = pd.ExcelFile(EXCEL)
sp_raw = xl.parse("Spectra")
stands_raw = xl.parse("Stand_characteristics", header=1)

wl_cols   = [c for c in sp_raw.columns if str(c).startswith("WL")]
wavelengths = np.array([int(str(c).replace("WL", "")) for c in wl_cols])

mask = np.ones(len(wavelengths), dtype=bool)
for lo, hi in EXCLUDE_NM:
    mask &= ~((wavelengths >= lo) & (wavelengths <= hi))
valid_wl = wavelengths[mask]

def assign_type(row):
    conif = row["%-Spruce"] + row["%-Pine"]
    if conif > 70: return "coniferous"
    if row["%-Broadleaf"] > 70: return "broadleaved"
    return "mixed"

site_map = {"HB": "Jarvselja", "TM": "BilyKriz", "TF": "Lanzhot"}
stands_raw["forest_type"] = stands_raw.apply(assign_type, axis=1)
stands_raw["site"] = stands_raw["Study site"].map(site_map)

sp_raw = sp_raw.dropna(subset=["Stand nr"])
sp_raw["Stand nr"] = sp_raw["Stand nr"].astype(int)

# measurement-level matrices
X_meas  = sp_raw[wl_cols].values[:, mask].astype(np.float32)
stand_nrs = sp_raw["Stand nr"].values.astype(int)
nr_to = stands_raw.set_index("Stand nr")[["ID", "forest_type", "site"]].to_dict("index")
y_meas    = np.array([nr_to[n]["forest_type"] for n in stand_nrs])
sites_meas = np.array([nr_to[n]["site"] for n in stand_nrs])

# stand-level means
stand_nrs_u = sorted(set(stand_nrs))
X_stand, y_stand, sites_stand, ids_stand = [], [], [], []
for nr in stand_nrs_u:
    m = stand_nrs == nr
    X_stand.append(X_meas[m].mean(0))
    y_stand.append(nr_to[nr]["forest_type"])
    sites_stand.append(nr_to[nr]["site"])
    ids_stand.append(nr_to[nr]["ID"])
X_stand   = np.array(X_stand)
y_stand   = np.array(y_stand)
sites_stand = np.array(sites_stand)
ids_stand = np.array(ids_stand)


# ---- Helper --------------------------------------------------------------
def run_logo(X, y, groups, pca_n=20, label=""):
    logo = LeaveOneGroupOut()
    folds = []
    for tr, te in logo.split(X, y, groups):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        g_te = np.unique(groups[te])[0]

        sc  = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
        n   = min(pca_n, Xtr.shape[1], Xtr.shape[0]-1)
        pca = PCA(n_components=n, random_state=42)
        Xtr = pca.fit_transform(Xtr); Xte = pca.transform(Xte)

        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(Xtr, ytr)
        yp = rf.predict(Xte)

        ba = balanced_accuracy_score(yte, yp)
        f1 = f1_score(yte, yp, average="macro", zero_division=0)
        folds.append({"group": g_te, "ba": float(ba), "f1": float(f1),
                      "n_test": int(len(te)),
                      "dist": {c: int((yte==c).sum()) for c in sorted(set(y))}})

    mean_ba = float(np.mean([f["ba"] for f in folds]))
    std_ba  = float(np.std([f["ba"]  for f in folds]))
    fold_strs = [f['ba'] for f in folds]
    print(f"  {label}: BA={mean_ba:.3f} folds={[round(v,3) for v in fold_strs]}")
    return {"mean_ba": mean_ba, "std_ba": std_ba,
            "mean_f1": float(np.mean([f["f1"] for f in folds])),
            "folds": folds, "n_samples": len(y)}


results = {}

# ---- Exp 1: Within-Jarvselja LOGO ----------------------------------------
print("\n=== 1. Within-Jarvselja leave-one-stand-out ===")
mask_js = sites_stand == "Jarvselja"
results["JS_logo_stand"] = run_logo(
    X_stand[mask_js], y_stand[mask_js], ids_stand[mask_js],
    pca_n=10, label="JS LOGO (stand-level)")

mask_js_m = sites_meas == "Jarvselja"
results["JS_logo_meas"] = run_logo(
    X_meas[mask_js_m], y_meas[mask_js_m],
    np.array([nr_to[n]["ID"] for n in stand_nrs])[mask_js_m],
    pca_n=10, label="JS LOGO (measurement-level)")

# ---- Exp 2: Cross-site LOSO (LOSO = leave-one-SITE-out) ------------------
print("\n=== 2. Cross-site LOSO (all 3 sites) ===")
results["loso_all_sites"] = run_logo(
    X_stand, y_stand, sites_stand, pca_n=20, label="LOSO 3-site stand-level")

# ---- Exp 3: Train JS+LZ → predict BK (temperate-to-boreal transfer) -----
print("\n=== 3. Directional transfer experiments ===")
mask_train = sites_stand != "BilyKriz"
mask_test  = sites_stand == "BilyKriz"

Xtr = X_stand[mask_train]; ytr = y_stand[mask_train]
Xte = X_stand[mask_test];  yte = y_stand[mask_test]
sc  = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
n   = min(20, Xtr.shape[1], Xtr.shape[0]-1)
pca = PCA(n_components=n, random_state=42)
Xtr = pca.fit_transform(Xtr); Xte = pca.transform(Xte)
rf  = RandomForestClassifier(**RF_PARAMS); rf.fit(Xtr, ytr)
yp  = rf.predict(Xte)
ba_bk = balanced_accuracy_score(yte, yp)
print(f"  Train JS+LZ → test BK: BA={ba_bk:.3f}  pred={dict(zip(*np.unique(yp, return_counts=True)))}")
results["transfer_JS_LZ_to_BK"] = {"ba": float(ba_bk), "pred_dist": {str(k): int(v) for k, v in zip(*np.unique(yp, return_counts=True))}}

mask_train2 = sites_stand != "Lanzhot"
mask_test2  = sites_stand == "Lanzhot"
Xtr = X_stand[mask_train2]; ytr = y_stand[mask_train2]
Xte = X_stand[mask_test2];  yte = y_stand[mask_test2]
sc  = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
n   = min(20, Xtr.shape[1], Xtr.shape[0]-1)
pca = PCA(n_components=n, random_state=42)
Xtr = pca.fit_transform(Xtr); Xte = pca.transform(Xte)
rf  = RandomForestClassifier(**RF_PARAMS); rf.fit(Xtr, ytr)
yp  = rf.predict(Xte)
ba_lz = balanced_accuracy_score(yte, yp)
print(f"  Train JS+BK → test LZ: BA={ba_lz:.3f}  pred={dict(zip(*np.unique(yp, return_counts=True)))}")
results["transfer_JS_BK_to_LZ"] = {"ba": float(ba_lz), "pred_dist": {str(k): int(v) for k, v in zip(*np.unique(yp, return_counts=True))}}


# ---- Exp 4: Spectral separability plot ------------------------------------
print("\n=== 4. Generating spectral separability figure ===")
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Stand-level mean spectra by type
for ax, site_filter, title in [
    (axes[0], None, "All sites — mean spectra by forest type (field spectra 350–2500 nm)"),
    (axes[1], "Jarvselja", "Jarvselja only — coniferous vs. broadleaved"),
]:
    mask_f = np.ones(len(y_stand), dtype=bool)
    if site_filter:
        mask_f = sites_stand == site_filter

    classes = sorted(np.unique(y_stand[mask_f]).tolist())
    colors = {"coniferous": "#2e7d32", "broadleaved": "#f57f17", "mixed": "#7b1fa2"}

    for cls in classes:
        m = mask_f & (y_stand == cls)
        mean_s = X_stand[m].mean(0)
        std_s  = X_stand[m].std(0)
        c = colors.get(cls, "grey")
        ax.plot(valid_wl, mean_s, color=c, label=f"{cls} (n={m.sum()})", lw=1.5)
        ax.fill_between(valid_wl, mean_s - std_s, mean_s + std_s, color=c, alpha=0.15)

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(350, 2500)
    # Mark water vapour exclusions
    for lo, hi in EXCLUDE_NM:
        ax.axvspan(lo, hi, alpha=0.07, color="grey")

plt.tight_layout()
fig_path = FIG_DIR / "field_spectra_mean_by_type.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {fig_path}")


# ---- Exp 5: Separability index per wavelength ----------------------------
print("\n=== 5. Computing separability index ===")
# J-M distance proxy: (mu1 - mu2)^2 / (sigma1^2 + sigma2^2) per band
con_mask = y_stand == "coniferous"
brd_mask = y_stand == "broadleaved"

mu_c = X_stand[con_mask].mean(0); sd_c = X_stand[con_mask].std(0) + 1e-8
mu_b = X_stand[brd_mask].mean(0); sd_b = X_stand[brd_mask].std(0) + 1e-8
sep_idx = (mu_c - mu_b)**2 / (sd_c**2 + sd_b**2)

top10_idx = np.argsort(sep_idx)[-10:][::-1]
top10_wl  = valid_wl[top10_idx]
print(f"  Top-10 most separable wavelengths: {top10_wl} nm")

fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.fill_between(valid_wl, sep_idx, alpha=0.6, color="steelblue")
ax2.scatter(top10_wl, sep_idx[top10_idx], color="tomato", zorder=5,
            s=30, label="Top-10 bands")
for lo, hi in EXCLUDE_NM:
    ax2.axvspan(lo, hi, alpha=0.07, color="grey")
ax2.set_xlabel("Wavelength (nm)")
ax2.set_ylabel("Separability index (JM proxy)")
ax2.set_title("Spectral separability: coniferous vs. broadleaved (field spectra, all sites)")
ax2.legend()
ax2.set_xlim(350, 2500)
plt.tight_layout()
fig2_path = FIG_DIR / "field_spectra_separability.png"
fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  Saved: {fig2_path}")
results["top10_separable_wavelengths_nm"] = [int(w) for w in top10_wl]


# ---- Save ----------------------------------------------------------------
out_path = OUT_DIR / "field_spectra_deep.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nAll results saved to {out_path}")
print("\n" + "="*70)
print("SUMMARY — Field Spectra Extended Analysis")
print("="*70)
exps = [
    ("Within-JS LOGO (stand-level)",      "JS_logo_stand"),
    ("Within-JS LOGO (15-meas)",          "JS_logo_meas"),
    ("LOSO 3-site cross-site",            "loso_all_sites"),
]
for label, key in exps:
    r = results[key]
    print(f"  {label:<45}: BA = {r['mean_ba']:.3f} ± {r['std_ba']:.3f}")
print(f"  {'Transfer JS+LZ → BK (conifer)':<45}: BA = {results['transfer_JS_LZ_to_BK']['ba']:.3f}")
print(f"  {'Transfer JS+BK → LZ (broadleaved)':<45}: BA = {results['transfer_JS_BK_to_LZ']['ba']:.3f}")
print("="*70)
print(f"\nTop-10 separable wavelengths: {results['top10_separable_wavelengths_nm']} nm")
print(f"\nAirborne CASI reference: Binary BA = 0.932  |  3-class BA = 0.621")
