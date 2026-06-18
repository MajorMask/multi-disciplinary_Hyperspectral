"""
Cross-sensor prediction: train on CASI airborne (HY+JS+LZ), predict on
BilyKriz field spectra resampled to CASI band centres.

This directly answers "can we apply the airborne model to BK field data?"
and shows whether the BK Norway spruce spectral signature is consistent with
the model trained on pine/spruce from 3 other biomes.

Outputs:
  - Predictions for all 8 BK field-spectra stands
  - Probability estimates per class
  - outputs/ablations/cross_sensor_bk.json
  - outputs/figures/cross_sensor_bk_spectra.png
"""

import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ---- Paths ----------------------------------------------------------------
ROOT = Path("C:/Users/aggarwm1/Videos/multi-disciplinary_Hyperspectral/model")
EXCEL = Path(
    "C:/Users/aggarwm1/Videos/multi-disciplinary_Hyperspectral/model/data"
    "/Hyperspectral/Airborne_data"
    "/DatasetOfTreeCanopyStructureUnderstoryReflectanceSpectraAndFractionalCoverInHemiborealAndTemperateForestAreasInEstoniaAndCzechRepublic_V2.xlsx"
)
OUT_DIR = Path("C:/Users/aggarwm1/Videos/multi-disciplinary_Hyperspectral/outputs/ablations")
FIG_DIR = Path("C:/Users/aggarwm1/Videos/multi-disciplinary_Hyperspectral/outputs/figures")

sys.path.insert(0, str(ROOT.parent))
with open(ROOT / "config" / "default.yaml") as f:
    cfg = yaml.safe_load(f)

from model.src.dataio.raster_loader import load_all_tiles
from model.src.preprocessing.band_selection import get_valid_band_mask
from model.src.preprocessing.normalization import apply_normalization
from model.src.features.stand_summary import build_stand_feature_matrix

CASI_WL = np.array(cfg["preprocessing"]["casi_wavelengths_nm"])
EXCLUDE_NM = [(895, 1003), (1092, 1168), (1302, 1528), (1737, 2038)]

tile_dir = Path(
    "C:/Users/aggarwm1/Videos/multi-disciplinary_Hyperspectral/model/data"
    "/Hyperspectral/Airborne_data/Airborne_hyperspectral/Analysis_ready_subsets/CASI"
)
nodata = cfg["preprocessing"]["nodata_value"]
scale  = cfg["preprocessing"]["reflectance_scale_factor"]

# ---- Load CASI tiles -------------------------------------------------------
print("Loading CASI airborne tiles...")
tiles = load_all_tiles(str(tile_dir), pattern="*_CASI.tif", nodata_value=nodata, min_valid_fraction=0.5)
for t in tiles:
    if np.array_equal(t.wavelengths, np.arange(t.n_bands)):
        if len(CASI_WL) == t.n_bands:
            t.wavelengths = CASI_WL
    if t.stand_id.endswith("_CASI"):
        t.stand_id = t.stand_id[:-5]
    t.image = t.image / scale

band_mask = get_valid_band_mask(CASI_WL, [tuple(r) for r in EXCLUDE_NM])
valid_casi_wl = CASI_WL[band_mask]

X_casi, stand_ids = build_stand_feature_matrix(tiles, method="mean", band_mask=band_mask)
print(f"CASI feature matrix: {X_casi.shape}  ({len(valid_casi_wl)} valid bands)")

# ---- Load metadata ---------------------------------------------------------
meta = pd.read_csv(ROOT / "data" / "stand_metadata.csv")
id_to_type = dict(zip(meta["stand_id"], meta["forest_type"]))
id_to_site  = dict(zip(meta["stand_id"], meta["site"]))

y_casi   = np.array([id_to_type[s] for s in stand_ids])
sites_casi = np.array([id_to_site[s] for s in stand_ids])

# ---- Load field spectra ----------------------------------------------------
print("Loading field spectra...")
xl = pd.ExcelFile(EXCEL)
sp_raw   = xl.parse("Spectra")
stands   = xl.parse("Stand_characteristics", header=1)

wl_cols     = [c for c in sp_raw.columns if str(c).startswith("WL")]
field_wl    = np.array([int(str(c).replace("WL", "")) for c in wl_cols])
sp_raw = sp_raw.dropna(subset=["Stand nr"])
sp_raw["Stand nr"] = sp_raw["Stand nr"].astype(int)

nr_to = stands.set_index("Stand nr")[["ID"]].to_dict("index")

# BilyKriz stands only (Stand nr 14-21 = BK_Spruce1-8)
bk_nrs = stands.loc[stands["Study site"] == "TM", "Stand nr"].tolist()
bk_mask = sp_raw["Stand nr"].isin(bk_nrs)
bk_sp   = sp_raw.loc[bk_mask, wl_cols].values.astype(np.float64)
bk_nrs_per_row = sp_raw.loc[bk_mask, "Stand nr"].values.astype(int)
bk_ids  = np.array([nr_to[n]["ID"] for n in bk_nrs_per_row])

# Stand-level means for BK
bk_stand_ids = []
bk_stand_means = []
for nr in sorted(set(bk_nrs_per_row)):
    m = bk_nrs_per_row == nr
    bk_stand_means.append(bk_sp[m].mean(0))
    bk_stand_ids.append(nr_to[nr]["ID"])
bk_stand_means = np.array(bk_stand_means)  # (8, 2151)
bk_stand_ids   = np.array(bk_stand_ids)

# ---- Resample field spectra to CASI wavelengths ----------------------------
print("Resampling field spectra to CASI band centres...")
resampled = np.zeros((bk_stand_means.shape[0], len(valid_casi_wl)))
for i, spec in enumerate(bk_stand_means):
    interp = interp1d(field_wl, spec, kind="linear", fill_value="extrapolate")
    full_resampled = interp(CASI_WL)
    resampled[i] = full_resampled[band_mask]

# ---- Train on CASI (HY + JS + LZ, exclude BK) and predict BK field --------
train_mask = ~np.isin(sites_casi, ["BilyKriz"])
X_train = X_casi[train_mask]
y_train = y_casi[train_mask]

print(f"Training on {train_mask.sum()} stands (HY+JS+LZ): {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"Predicting {len(resampled)} BK stands (all Norway spruce -> expected: coniferous)")

sc  = StandardScaler()
X_train_s   = sc.fit_transform(X_train)
X_predict_s = sc.transform(resampled)

rf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                             min_samples_leaf=2, n_jobs=-1, random_state=42)
rf.fit(X_train_s, y_train)

y_pred_bk  = rf.predict(X_predict_s)
y_proba_bk = rf.predict_proba(X_predict_s)
class_labels = rf.classes_

print("\n=== Cross-sensor predictions on BilyKriz field spectra ===")
print(f"{'Stand':15s}  {'Pred':12s}  {'P(conif)':10s}  {'P(broadlv)':12s}  {'P(mixed)':10s}")
print("-" * 65)
for sid, pred, prob in zip(bk_stand_ids, y_pred_bk, y_proba_bk):
    p_dict = dict(zip(class_labels, prob))
    print(f"  {sid:13s}  {pred:12s}  "
          f"{p_dict.get('coniferous',0):.3f}       "
          f"{p_dict.get('broadleaved',0):.3f}          "
          f"{p_dict.get('mixed',0):.3f}")

correct = (y_pred_bk == "coniferous").sum()
print(f"\n  Correctly predicted as coniferous: {correct}/{len(y_pred_bk)}")
print(f"  BA on BK (all coniferous ground truth): {correct/len(y_pred_bk):.3f}")


# ---- Figure: CASI vs resampled field spectra comparison -------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: mean airborne CASI spectra vs mean resampled BK field spectra
for ax, title, X_c, y_c, X_bk, label_bk in [
    (axes[0], "CASI airborne vs BK field spectra (CASI bands)",
     X_casi, y_casi, resampled, "BK field (resampled)"),
]:
    ax.set_title(title, fontsize=10)
    colors = {"coniferous": "#2e7d32", "broadleaved": "#f57f17", "mixed": "#7b1fa2"}

    for cls in ["coniferous", "broadleaved", "mixed"]:
        m = y_c == cls
        if m.sum() == 0: continue
        ax.plot(valid_casi_wl, X_c[m].mean(0), color=colors[cls],
                label=f"CASI {cls} (n={m.sum()})", lw=1.5)

    ax.plot(valid_casi_wl, X_bk.mean(0), color="black", linestyle="--",
            lw=2, label=f"BK field spectra (n={len(X_bk)}, resampled)")
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Reflectance")
    ax.legend(fontsize=8)

# Right: probability breakdown per BK stand
ax2 = axes[1]
x = np.arange(len(bk_stand_ids))
w = 0.25
for i, cls in enumerate(class_labels):
    probs = [p[i] for p in y_proba_bk]
    ax2.bar(x + i*w, probs, w, label=cls, alpha=0.8,
            color={"coniferous": "#2e7d32", "broadleaved": "#f57f17", "mixed": "#7b1fa2"}.get(cls, "grey"))
ax2.set_xticks(x + w)
ax2.set_xticklabels([sid.replace("BK_", "") for sid in bk_stand_ids], rotation=45, ha="right")
ax2.set_ylabel("Probability")
ax2.set_title("RF predicted probabilities for BK field spectra\n(model trained on HY+JS+LZ airborne)")
ax2.legend()
ax2.set_ylim(0, 1)

plt.tight_layout()
fig.savefig(FIG_DIR / "cross_sensor_bk_prediction.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved: {FIG_DIR / 'cross_sensor_bk_prediction.png'}")


# ---- Save JSON -------------------------------------------------------------
out = {
    "n_train_stands": int(train_mask.sum()),
    "train_class_dist": {k: int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
    "n_bk_stands": len(bk_stand_ids),
    "bk_predictions": [
        {"stand_id": sid, "prediction": pred,
         "probabilities": {cls: float(p) for cls, p in zip(class_labels, prob)}}
        for sid, pred, prob in zip(bk_stand_ids, y_pred_bk, y_proba_bk)
    ],
    "n_correct_coniferous": int(correct),
    "ba_bk": float(correct / len(y_pred_bk)),
}
with open(OUT_DIR / "cross_sensor_bk.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"Results saved: {OUT_DIR / 'cross_sensor_bk.json'}")
