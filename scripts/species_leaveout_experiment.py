"""
Leave-Species-Out Analysis: Norway Spruce vs Scots Pine
=======================================================
Addresses the limitation that coniferous species are lumped together.

Experiments:
  E1  Species separability  — can CASI tell spruce from pine? (binary, LOSO by site)
  E2  Leave-spruce-out      — train on pine+broad, test on spruce+broad
  E3  Leave-pine-out        — train on spruce+broad, test on pine+broad
  E4  Full binary baseline  — all coniferous vs broadleaved (reproduces 0.932 reference)
  E5  Site-species matrix   — per-site recall breakdown for E2 and E3
  E6  Leave-HY-spruce-out   — most aggressive: HY spruce excluded from training
  E7  Leave-BK-spruce-out   — remove plantation spruce (most different biome)

Outputs:
  outputs/ablations/species_leaveout.json
  outputs/figures/species_leaveout_*.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                              confusion_matrix, f1_score, recall_score)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT    = Path("C:/Users/aggarwm1/Videos/multi-disciplinary_Hyperspectral/model")
OUT_DIR = Path("C:/Users/aggarwm1/Videos/multi-disciplinary_Hyperspectral/outputs")
ABL_DIR = OUT_DIR / "ablations"
FIG_DIR = OUT_DIR / "figures"
ABL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(ROOT / "config" / "default.yaml") as f:
    cfg = yaml.safe_load(f)

import sys
sys.path.insert(0, str(ROOT.parent))
from model.src.dataio.raster_loader import load_all_tiles
from model.src.preprocessing.band_selection import get_valid_band_mask
from model.src.features.stand_summary import build_stand_feature_matrix

CASI_WL    = np.array(cfg["preprocessing"]["casi_wavelengths_nm"])
NODATA     = cfg["preprocessing"]["nodata_value"]
SCALE      = cfg["preprocessing"]["reflectance_scale_factor"]
EXCLUDE_NM = [(895,1003),(1092,1168),(1302,1528),(1737,2038)]
TILE_DIR   = Path(
    "C:/Users/aggarwm1/Videos/multi-disciplinary_Hyperspectral/model/data"
    "/Hyperspectral/Airborne_data/Airborne_hyperspectral/Analysis_ready_subsets/CASI"
)
RF = dict(n_estimators=500, class_weight="balanced",
          min_samples_leaf=2, n_jobs=-1, random_state=42)

# ── Load tiles ─────────────────────────────────────────────────────────────
print("Loading CASI tiles...")
tiles = load_all_tiles(str(TILE_DIR), pattern="*_CASI.tif",
                       nodata_value=NODATA, min_valid_fraction=0.5)
for t in tiles:
    if np.array_equal(t.wavelengths, np.arange(t.n_bands)):
        t.wavelengths = CASI_WL
    if t.stand_id.endswith("_CASI"):
        t.stand_id = t.stand_id[:-5]
    t.image = t.image / SCALE

band_mask    = get_valid_band_mask(CASI_WL, [tuple(r) for r in EXCLUDE_NM])
X_all, sids  = build_stand_feature_matrix(tiles, method="mean", band_mask=band_mask)
print(f"Feature matrix: {X_all.shape}  ({band_mask.sum()} valid bands)")

# ── Metadata ───────────────────────────────────────────────────────────────
meta = pd.read_csv(ROOT / "data" / "stand_metadata.csv")
id2type    = dict(zip(meta.stand_id, meta.forest_type))
id2site    = dict(zip(meta.stand_id, meta.site))
id2species = dict(zip(meta.stand_id, meta.dominant_species))

types   = np.array([id2type.get(s, "unknown") for s in sids])
sites   = np.array([id2site.get(s, "unknown") for s in sids])
species = np.array([id2species.get(s, "unknown") for s in sids])

print("\nConiferous species breakdown:")
conif_mask = types == "coniferous"
uniq, cnts = np.unique(species[conif_mask], return_counts=True)
for u, c in zip(uniq, cnts):
    print(f"  {u}: {c} stands")

print("\nDataset sizes:")
print(f"  Total stands        : {len(sids)}")
print(f"  Coniferous          : {conif_mask.sum()}")
print(f"  - Norway spruce     : {(species=='Norway spruce').sum()}")
print(f"  - Scots pine        : {(species=='Scots pine').sum()}")
print(f"  Broadleaved         : {(types=='broadleaved').sum()}")
print(f"  Mixed               : {(types=='mixed').sum()}")

results = {}


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def fit_predict(X_tr, y_tr, X_te):
    sc  = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)
    clf  = RandomForestClassifier(**RF)
    clf.fit(X_tr, y_tr)
    return clf.predict(X_te), clf.predict_proba(X_te), clf.classes_

def binary_metrics(y_true, y_pred, pos_label=None):
    classes = sorted(set(y_true) | set(y_pred))
    ba = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class_recall = {}
    for c in set(y_true):
        mask = y_true == c
        if mask.sum() > 0:
            per_class_recall[c] = float((y_pred[mask] == c).sum() / mask.sum())
    return {"balanced_accuracy": float(ba),
            "macro_f1": float(f1),
            "per_class_recall": per_class_recall,
            "n_test": int(len(y_true))}


# ═══════════════════════════════════════════════════════════════════════════
# E4 — Full binary baseline (all coniferous vs broadleaved, LOSO)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("E4 — Full binary LOSO baseline (coniferous vs broadleaved)")
binary_mask = types != "mixed"
Xb = X_all[binary_mask]
yb = np.where(types[binary_mask] == "coniferous", "coniferous", "broadleaved")
sb = sites[binary_mask]

logo = LeaveOneGroupOut()
e4_folds = []
for tr, te in logo.split(Xb, yb, sb):
    yp, _, _ = fit_predict(Xb[tr], yb[tr], Xb[te])
    fold_site = np.unique(sb[te])[0]
    m = binary_metrics(yb[te], yp)
    m["site"] = fold_site
    e4_folds.append(m)
    print(f"  test={fold_site:12s}  BA={m['balanced_accuracy']:.3f}  "
          f"recall_conif={m['per_class_recall'].get('coniferous',0):.3f}  "
          f"recall_broad={m['per_class_recall'].get('broadleaved',0):.3f}")

e4_mean = float(np.mean([f["balanced_accuracy"] for f in e4_folds]))
print(f"  MEAN BA = {e4_mean:.3f}")
results["E4_full_binary_loso"] = {
    "description": "All coniferous vs broadleaved, LOSO by site",
    "mean_ba": e4_mean,
    "folds": e4_folds
}


# ═══════════════════════════════════════════════════════════════════════════
# E1 — Species separability: Norway spruce vs Scots pine
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("E1 — Species separability: spruce vs pine (within coniferous)")

pine_mask   = species == "Scots pine"
spruce_mask = species == "Norway spruce"
sp_mask     = pine_mask | spruce_mask
X_sp = X_all[sp_mask]
y_sp = species[sp_mask]   # "Norway spruce" | "Scots pine"
s_sp = sites[sp_mask]

# LOSO by site (HY: 13 pine + 9 spruce; JS: 2 pine + 2 spruce; BK: 7 spruce)
e1_folds = []
for tr, te in logo.split(X_sp, y_sp, s_sp):
    fold_site = np.unique(s_sp[te])[0]
    classes_in_test = np.unique(y_sp[te])
    if len(classes_in_test) < 2:
        print(f"  test={fold_site:12s}  SKIP (single species in test: {classes_in_test[0]})")
        e1_folds.append({"site": fold_site, "skipped": True,
                          "reason": f"single species: {classes_in_test[0]}"})
        continue
    yp, _, _ = fit_predict(X_sp[tr], y_sp[tr], X_sp[te])
    m = binary_metrics(y_sp[te], yp)
    m["site"] = fold_site
    e1_folds.append(m)
    print(f"  test={fold_site:12s}  BA={m['balanced_accuracy']:.3f}  "
          f"recall_spruce={m['per_class_recall'].get('Norway spruce',0):.3f}  "
          f"recall_pine={m['per_class_recall'].get('Scots pine',0):.3f}  "
          f"n={m['n_test']}")

valid_bas = [f["balanced_accuracy"] for f in e1_folds if not f.get("skipped")]
e1_mean = float(np.mean(valid_bas)) if valid_bas else 0.0
print(f"  MEAN BA (valid folds) = {e1_mean:.3f}")
results["E1_species_separability"] = {
    "description": "Norway spruce vs Scots pine, LOSO by site (coniferous stands only)",
    "mean_ba": e1_mean,
    "folds": e1_folds
}


# ═══════════════════════════════════════════════════════════════════════════
# E2 — Leave-spruce-out: train pine+broad, test spruce+broad
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("E2 — Leave-spruce-out: train on [Scots pine + broadleaved], test on [Norway spruce + broadleaved]")

broad_mask = types == "broadleaved"
# Training: pine + broadleaved
train_mask_e2 = pine_mask | broad_mask
X_tr_e2 = X_all[train_mask_e2]
y_tr_e2 = np.where(types[train_mask_e2]=="coniferous", "coniferous", "broadleaved")
print(f"  Training: {(y_tr_e2=='coniferous').sum()} pine + {(y_tr_e2=='broadleaved').sum()} broadleaved")

# Test: spruce + broadleaved
test_mask_e2 = spruce_mask | broad_mask
X_te_e2 = X_all[test_mask_e2]
y_te_e2 = np.where(types[test_mask_e2]=="coniferous", "coniferous", "broadleaved")
sids_te_e2 = np.array(sids)[test_mask_e2]
sites_te_e2 = sites[test_mask_e2]
print(f"  Test    : {(y_te_e2=='coniferous').sum()} spruce + {(y_te_e2=='broadleaved').sum()} broadleaved")

yp_e2, proba_e2, cls_e2 = fit_predict(X_tr_e2, y_tr_e2, X_te_e2)
m_e2 = binary_metrics(y_te_e2, yp_e2)
print(f"\n  Overall BA = {m_e2['balanced_accuracy']:.3f}")
print(f"  Recall(coniferous=spruce) = {m_e2['per_class_recall'].get('coniferous',0):.3f}")
print(f"  Recall(broadleaved)       = {m_e2['per_class_recall'].get('broadleaved',0):.3f}")

# Per-site recall on spruce test stands
print("\n  Per-site spruce recall:")
site_recall_e2 = {}
for site_name in ["Hyytiala","Jarvselja","BilyKriz"]:
    mask = (sites_te_e2 == site_name) & (y_te_e2 == "coniferous")
    if mask.sum() == 0:
        continue
    n_correct = (yp_e2[mask] == "coniferous").sum()
    recall = n_correct / mask.sum()
    site_recall_e2[site_name] = float(recall)
    print(f"    {site_name:12s}: {n_correct}/{mask.sum()} correct  (recall={recall:.3f})")

# Per-stand predictions
stand_preds_e2 = {}
for sid, true, pred in zip(sids_te_e2, y_te_e2, yp_e2):
    if true == "coniferous":  # only report spruce stands
        stand_preds_e2[sid] = {"true": true, "predicted": pred,
                                 "correct": bool(pred == true)}

results["E2_leave_spruce_out"] = {
    "description": "Train on [Scots pine + broadleaved], test on [Norway spruce + broadleaved]",
    "n_train": len(X_tr_e2),
    "n_test": len(X_te_e2),
    "metrics": m_e2,
    "spruce_site_recall": site_recall_e2,
    "spruce_stand_predictions": stand_preds_e2
}


# ═══════════════════════════════════════════════════════════════════════════
# E3 — Leave-pine-out: train spruce+broad, test pine+broad
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("E3 — Leave-pine-out: train on [Norway spruce + broadleaved], test on [Scots pine + broadleaved]")

train_mask_e3 = spruce_mask | broad_mask
X_tr_e3 = X_all[train_mask_e3]
y_tr_e3 = np.where(types[train_mask_e3]=="coniferous", "coniferous", "broadleaved")
print(f"  Training: {(y_tr_e3=='coniferous').sum()} spruce + {(y_tr_e3=='broadleaved').sum()} broadleaved")

test_mask_e3 = pine_mask | broad_mask
X_te_e3 = X_all[test_mask_e3]
y_te_e3 = np.where(types[test_mask_e3]=="coniferous", "coniferous", "broadleaved")
sids_te_e3 = np.array(sids)[test_mask_e3]
sites_te_e3 = sites[test_mask_e3]
print(f"  Test    : {(y_te_e3=='coniferous').sum()} pine + {(y_te_e3=='broadleaved').sum()} broadleaved")

yp_e3, proba_e3, cls_e3 = fit_predict(X_tr_e3, y_tr_e3, X_te_e3)
m_e3 = binary_metrics(y_te_e3, yp_e3)
print(f"\n  Overall BA = {m_e3['balanced_accuracy']:.3f}")
print(f"  Recall(coniferous=pine)   = {m_e3['per_class_recall'].get('coniferous',0):.3f}")
print(f"  Recall(broadleaved)       = {m_e3['per_class_recall'].get('broadleaved',0):.3f}")

print("\n  Per-site pine recall:")
site_recall_e3 = {}
for site_name in ["Hyytiala","Jarvselja"]:
    mask = (sites_te_e3 == site_name) & (y_te_e3 == "coniferous")
    if mask.sum() == 0:
        continue
    n_correct = (yp_e3[mask] == "coniferous").sum()
    recall = n_correct / mask.sum()
    site_recall_e3[site_name] = float(recall)
    print(f"    {site_name:12s}: {n_correct}/{mask.sum()} correct  (recall={recall:.3f})")

stand_preds_e3 = {}
for sid, true, pred in zip(sids_te_e3, y_te_e3, yp_e3):
    if true == "coniferous":
        stand_preds_e3[sid] = {"true": true, "predicted": pred,
                                 "correct": bool(pred == true)}

results["E3_leave_pine_out"] = {
    "description": "Train on [Norway spruce + broadleaved], test on [Scots pine + broadleaved]",
    "n_train": len(X_tr_e3),
    "n_test": len(X_te_e3),
    "metrics": m_e3,
    "pine_site_recall": site_recall_e3,
    "pine_stand_predictions": stand_preds_e3
}


# ═══════════════════════════════════════════════════════════════════════════
# E6 — Leave-HY-spruce-out (biome-species combo)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("E6 — Leave-HY-spruce-out: remove Hyytiala Norway spruce from training")

hy_spruce_mask = (species == "Norway spruce") & (sites == "Hyytiala")
print(f"  Holding out {hy_spruce_mask.sum()} HY spruce stands")

# Train on everything except HY spruce AND mixed
train_mask_e6 = ~hy_spruce_mask & (types != "mixed")
X_tr_e6 = X_all[train_mask_e6]
y_tr_e6 = np.where(types[train_mask_e6]=="coniferous","coniferous","broadleaved")
print(f"  Training: {(y_tr_e6=='coniferous').sum()} conif + {(y_tr_e6=='broadleaved').sum()} broad")

# Test: HY spruce + all broadleaved
test_mask_e6 = hy_spruce_mask | broad_mask
X_te_e6 = X_all[test_mask_e6]
y_te_e6 = np.where(types[test_mask_e6]=="coniferous","coniferous","broadleaved")
sids_te_e6 = np.array(sids)[test_mask_e6]
print(f"  Test    : {(y_te_e6=='coniferous').sum()} HY spruce + {(y_te_e6=='broadleaved').sum()} broadleaved")

yp_e6, _, _ = fit_predict(X_tr_e6, y_tr_e6, X_te_e6)
m_e6 = binary_metrics(y_te_e6, yp_e6)
hy_spruce_recall = float((yp_e6[y_te_e6=="coniferous"]=="coniferous").mean())
print(f"  BA={m_e6['balanced_accuracy']:.3f}  HY_spruce recall={hy_spruce_recall:.3f}")

results["E6_leave_HY_spruce_out"] = {
    "description": "Remove Hyytiala Norway spruce from training; test on [HY spruce + broadleaved]",
    "n_train": int(train_mask_e6.sum()),
    "metrics": m_e6,
    "hy_spruce_recall": hy_spruce_recall
}


# ═══════════════════════════════════════════════════════════════════════════
# E7 — Leave-BK-spruce-out (plantation spruce, most distinct biome)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("E7 — Leave-BK-spruce-out: remove BilyKriz spruce from training")

bk_spruce_mask = (species == "Norway spruce") & (sites == "BilyKriz")
print(f"  Holding out {bk_spruce_mask.sum()} BK spruce stands (plantation, temperate montane)")

train_mask_e7 = ~bk_spruce_mask & (types != "mixed")
X_tr_e7 = X_all[train_mask_e7]
y_tr_e7 = np.where(types[train_mask_e7]=="coniferous","coniferous","broadleaved")

test_mask_e7 = bk_spruce_mask | broad_mask
X_te_e7 = X_all[test_mask_e7]
y_te_e7 = np.where(types[test_mask_e7]=="coniferous","coniferous","broadleaved")
sids_te_e7 = np.array(sids)[test_mask_e7]
print(f"  Training: {(y_tr_e7=='coniferous').sum()} conif + {(y_tr_e7=='broadleaved').sum()} broad")
print(f"  Test    : {(y_te_e7=='coniferous').sum()} BK spruce + {(y_te_e7=='broadleaved').sum()} broadleaved")

yp_e7, proba_e7, cls_e7 = fit_predict(X_tr_e7, y_tr_e7, X_te_e7)
m_e7 = binary_metrics(y_te_e7, yp_e7)
bk_recall = float((yp_e7[y_te_e7=="coniferous"]=="coniferous").mean())

stand_preds_e7 = {}
for sid, true, pred in zip(sids_te_e7, y_te_e7, yp_e7):
    if true == "coniferous":
        stand_preds_e7[sid] = {"predicted": pred, "correct": bool(pred == true)}

print(f"  BA={m_e7['balanced_accuracy']:.3f}  BK_spruce recall={bk_recall:.3f}")
print(f"  Per-stand: {stand_preds_e7}")

results["E7_leave_BK_spruce_out"] = {
    "description": "Remove BilyKriz Norway spruce from training; test on [BK spruce + broadleaved]",
    "n_train": int(train_mask_e7.sum()),
    "metrics": m_e7,
    "bk_spruce_recall": bk_recall,
    "bk_stand_predictions": stand_preds_e7
}


# ═══════════════════════════════════════════════════════════════════════════
# E5 — Species-site interaction summary (mean spectra comparison)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("E5 — Computing species mean spectra and spectral distance")

valid_wl = CASI_WL[band_mask]
spruce_spectra = X_all[spruce_mask]
pine_spectra   = X_all[pine_mask]
broad_spectra  = X_all[broad_mask]

mu_spr = spruce_spectra.mean(0)
mu_pin = pine_spectra.mean(0)
mu_brd = broad_spectra.mean(0)
sd_spr = spruce_spectra.std(0)
sd_pin = pine_spectra.std(0)

# Separability index (JM proxy) between spruce and pine
sep_sp = (mu_spr - mu_pin)**2 / (sd_spr**2 + sd_pin**2 + 1e-8)
top5_sp = np.argsort(sep_sp)[-5:][::-1]
print(f"  Top-5 spruce/pine separable wavelengths: {valid_wl[top5_sp]} nm")

# Euclidean distances in feature space (after normalisation)
from numpy.linalg import norm
sc_ref = StandardScaler().fit(X_all[types!="mixed"])
mu_spr_n = sc_ref.transform(mu_spr.reshape(1,-1))[0]
mu_pin_n = sc_ref.transform(mu_pin.reshape(1,-1))[0]
mu_brd_n = sc_ref.transform(mu_brd.reshape(1,-1))[0]
d_spr_pin = float(norm(mu_spr_n - mu_pin_n))
d_spr_brd = float(norm(mu_spr_n - mu_brd_n))
d_pin_brd = float(norm(mu_pin_n - mu_brd_n))
print(f"  Euclidean distance (normalised feature space):")
print(f"    Spruce vs Pine       : {d_spr_pin:.3f}")
print(f"    Spruce vs Broadleaved: {d_spr_brd:.3f}")
print(f"    Pine   vs Broadleaved: {d_pin_brd:.3f}")

results["E5_species_spectra"] = {
    "description": "Spectral distances between spruce, pine, broadleaved in normalised feature space",
    "euclidean_distances": {
        "spruce_vs_pine": d_spr_pin,
        "spruce_vs_broadleaved": d_spr_brd,
        "pine_vs_broadleaved": d_pin_brd
    },
    "top5_spruce_pine_separable_nm": [int(w) for w in valid_wl[top5_sp]],
    "spruce_pine_separability_index_mean": float(sep_sp.mean())
}


# ═══════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════

# ── Figure 1: Summary bar chart ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
exp_labels = [
    "E4: Full binary\n(all conif. vs broad.)",
    "E2: Leave-spruce-out\n(train pine, test spruce)",
    "E3: Leave-pine-out\n(train spruce, test pine)",
    "E6: Leave-HY-spruce\n(hardest site combo)",
    "E7: Leave-BK-spruce\n(plantation spruce)",
]
ba_vals = [
    results["E4_full_binary_loso"]["mean_ba"],
    results["E2_leave_spruce_out"]["metrics"]["balanced_accuracy"],
    results["E3_leave_pine_out"]["metrics"]["balanced_accuracy"],
    results["E6_leave_HY_spruce_out"]["metrics"]["balanced_accuracy"],
    results["E7_leave_BK_spruce_out"]["metrics"]["balanced_accuracy"],
]
conif_recall = [
    np.mean([f["per_class_recall"].get("coniferous", 0) for f in results["E4_full_binary_loso"]["folds"]]),
    results["E2_leave_spruce_out"]["metrics"]["per_class_recall"].get("coniferous", 0),
    results["E3_leave_pine_out"]["metrics"]["per_class_recall"].get("coniferous", 0),
    results["E6_leave_HY_spruce_out"]["metrics"]["per_class_recall"].get("coniferous", 0),
    results["E7_leave_BK_spruce_out"]["metrics"]["per_class_recall"].get("coniferous", 0),
]
colors_exp = ["#2E75B6","#ED7D31","#70AD47","#7B1FA2","#C00000"]
x = np.arange(len(exp_labels)); w = 0.38
b1 = ax.bar(x - w/2, ba_vals,      w, color=colors_exp, alpha=0.88, label="Balanced Accuracy")
b2 = ax.bar(x + w/2, conif_recall, w, color=colors_exp, alpha=0.45, label="Coniferous Recall",
            edgecolor=[c for c in colors_exp], linewidth=1.5)
ax.axhline(0.5, color="#888", linestyle=":", lw=1, label="Chance")
ax.axhline(results["E4_full_binary_loso"]["mean_ba"],
           color="#2E75B6", linestyle="--", lw=1.4, label="Full binary baseline")
ax.set_xticks(x); ax.set_xticklabels(exp_labels, fontsize=8.5)
ax.set_ylim(0, 1.15); ax.set_ylabel("Score", fontsize=11)
ax.set_title("Leave-Species-Out: Balanced Accuracy and Coniferous Recall", fontsize=10, fontweight="bold")
ax.legend(fontsize=8.5, loc="lower right"); ax.grid(axis="y", alpha=0.3)
for bar, v in zip(b1, ba_vals):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.3f}", ha="center", fontsize=8.5, fontweight="bold")

# ── Figure right: species mean spectra ───────────────────────────────────
ax2 = axes[1]
ax2.plot(valid_wl, mu_spr, color="#2E7D32", lw=2, label=f"Norway spruce (n={spruce_spectra.shape[0]})")
ax2.fill_between(valid_wl, mu_spr-sd_spr, mu_spr+sd_spr, color="#2E7D32", alpha=0.15)
ax2.plot(valid_wl, mu_pin, color="#1565C0", lw=2, label=f"Scots pine (n={pine_spectra.shape[0]})")
ax2.fill_between(valid_wl, mu_pin-sd_pin, mu_pin+sd_pin, color="#1565C0", alpha=0.15)
ax2.plot(valid_wl, mu_brd, color="#F57F17", lw=2, label=f"Broadleaved (n={broad_spectra.shape[0]})")
ax2.fill_between(valid_wl, mu_brd-broad_spectra.std(0), mu_brd+broad_spectra.std(0),
                 color="#F57F17", alpha=0.15)
for lo, hi in EXCLUDE_NM:
    overlap_lo = max(lo, valid_wl.min()); overlap_hi = min(hi, valid_wl.max())
    if overlap_lo < overlap_hi:
        ax2.axvspan(overlap_lo, overlap_hi, alpha=0.07, color="grey")
ax2.set_xlabel("Wavelength (nm)", fontsize=11)
ax2.set_ylabel("Reflectance", fontsize=11)
ax2.set_title("Mean Reflectance: Norway Spruce vs Scots Pine vs Broadleaved\n(CASI 40 valid bands)", fontsize=10, fontweight="bold")
ax2.legend(fontsize=9); ax2.grid(alpha=0.2)
plt.tight_layout()
fig.savefig(FIG_DIR / "species_leaveout_summary.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved: {FIG_DIR / 'species_leaveout_summary.png'}")


# ── Figure 2: Separability index spruce vs pine ──────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

ax = axes[0]
ax.fill_between(valid_wl, sep_sp, alpha=0.6, color="steelblue", label="Spruce vs Pine")
sep_sp_brd = (mu_spr - mu_brd)**2 / (sd_spr**2 + broad_spectra.std(0)**2 + 1e-8)
sep_pin_brd = (mu_pin - mu_brd)**2 / (sd_pin**2 + broad_spectra.std(0)**2 + 1e-8)
ax.plot(valid_wl, sep_sp_brd, color="#2E7D32", lw=1.5, label="Spruce vs Broadleaved")
ax.plot(valid_wl, sep_pin_brd, color="#1565C0",  lw=1.5, label="Pine vs Broadleaved")
ax.scatter(valid_wl[top5_sp], sep_sp[top5_sp], color="tomato", zorder=5, s=40, label="Top-5 Spruce/Pine")
ax.set_ylabel("Separability index (JM proxy)", fontsize=11)
ax.set_title("Spectral Separability: Spruce vs Pine vs Broadleaved (CASI 40 bands)", fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.2)

ax2 = axes[1]
# Per-site spruce means
for site_name, col, ls in [("Hyytiala","#1B5E20","--"),
                             ("Jarvselja","#2E7D32","-."),
                             ("BilyKriz","#66BB6A",":")]:
    m_site = (spruce_mask) & (sites == site_name)
    if m_site.sum() == 0: continue
    ax2.plot(valid_wl, X_all[m_site].mean(0), color=col, ls=ls, lw=1.8,
             label=f"Spruce — {site_name} (n={m_site.sum()})")
for site_name, col, ls in [("Hyytiala","#0D47A1","--"),
                             ("Jarvselja","#1565C0","-.")]:
    m_site = (pine_mask) & (sites == site_name)
    if m_site.sum() == 0: continue
    ax2.plot(valid_wl, X_all[m_site].mean(0), color=col, ls=ls, lw=1.8,
             label=f"Pine   — {site_name} (n={m_site.sum()})")
ax2.set_xlabel("Wavelength (nm)", fontsize=11)
ax2.set_ylabel("Reflectance", fontsize=11)
ax2.set_title("Per-Site Mean Spectra: Norway Spruce and Scots Pine", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9); ax2.grid(alpha=0.2)
plt.tight_layout()
fig.savefig(FIG_DIR / "species_separability_spectra.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Figure saved: {FIG_DIR / 'species_separability_spectra.png'}")


# ── Figure 3: Per-stand prediction heatmap for E2 and E3 ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, stand_dict, title, species_name in [
    (axes[0], stand_preds_e2, "E2: Leave-spruce-out\n(test spruce stands)", "Norway spruce"),
    (axes[1], stand_preds_e3, "E3: Leave-pine-out\n(test pine stands)",   "Scots pine"),
]:
    stands_sorted = sorted(stand_dict.keys())
    correct_vals  = [1 if stand_dict[s]["correct"] else 0 for s in stands_sorted]
    colors_bar    = ["#2E7D32" if c else "#C62828" for c in correct_vals]
    ax.barh(range(len(stands_sorted)), correct_vals, color=colors_bar, edgecolor="white")
    ax.set_yticks(range(len(stands_sorted)))
    ax.set_yticklabels([s.replace("HY_","HY/").replace("JS_","JS/").replace("BK_","BK/")
                        for s in stands_sorted], fontsize=9)
    ax.set_xlim(0, 1.2); ax.set_xticks([0, 1])
    ax.set_xticklabels(["Wrong\n(broadleaved)", "Correct\n(coniferous)"], fontsize=9)
    total = len(stands_sorted); n_correct = sum(correct_vals)
    ax.set_title(f"{title}\n{n_correct}/{total} correctly labelled coniferous", fontsize=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for i, (s, c) in enumerate(zip(stands_sorted, correct_vals)):
        label = "✓" if c else "✗"
        color = "white" if c else "white"
        ax.text(0.02, i, label, va="center", fontsize=11, fontweight="bold",
                color="white" if c else "white")

plt.tight_layout()
fig.savefig(FIG_DIR / "species_stand_predictions.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Figure saved: {FIG_DIR / 'species_stand_predictions.png'}")


# ── Figure 4: E1 confusion — spruce vs pine ──────────────────────────────
# Pool all valid fold predictions for a summary confusion
all_true_e1, all_pred_e1 = [], []
for f in e1_folds:
    if f.get("skipped"):
        continue
# Re-run to collect arrays (LOSO re-run)
for tr, te in logo.split(X_sp, y_sp, s_sp):
    fold_site = np.unique(s_sp[te])[0]
    if len(np.unique(y_sp[te])) < 2:
        continue
    yp, _, _ = fit_predict(X_sp[tr], y_sp[tr], X_sp[te])
    all_true_e1.extend(y_sp[te])
    all_pred_e1.extend(yp)

if all_true_e1:
    cls_labels = ["Norway spruce","Scots pine"]
    cm_e1 = confusion_matrix(all_true_e1, all_pred_e1, labels=cls_labels)
    row_sum = cm_e1.sum(axis=1, keepdims=True); row_sum[row_sum==0] = 1
    cm_norm_e1 = cm_e1 / row_sum

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm_e1, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(2)); ax.set_yticks(range(2))
    ax.set_xticklabels(["Norway\nspruce","Scots\npine"], fontsize=11)
    ax.set_yticklabels(["Norway\nspruce","Scots\npine"], fontsize=11)
    ax.set_xlabel("Predicted", fontsize=11); ax.set_ylabel("True", fontsize=11)
    ax.set_title("E1: Spruce vs Pine Confusion Matrix\n(LOSO, valid folds only)", fontsize=10, fontweight="bold")
    for i in range(2):
        for j in range(2):
            v = cm_norm_e1[i,j]; raw = cm_e1[i,j]
            ax.text(j, i, f"{v:.2f}\n(n={raw})", ha="center", va="center",
                    fontsize=10, color="white" if v>0.5 else "black")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "species_confusion_e1.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {FIG_DIR / 'species_confusion_e1.png'}")


# ── Save JSON ──────────────────────────────────────────────────────────────
out_path = ABL_DIR / "species_leaveout.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved: {out_path}")


# ── Print final summary ────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY — Leave-Species-Out Analysis")
print("="*70)
print(f"  {'Experiment':<45}  {'BA':>6}  {'Conif. Recall':>13}")
print("-"*70)
print(f"  {'E4: Full binary LOSO (baseline)':<45}  {results['E4_full_binary_loso']['mean_ba']:>6.3f}  {'—':>13}")
print(f"  {'E1: Spruce vs pine separability':<45}  {results['E1_species_separability']['mean_ba']:>6.3f}  {'—':>13}")
print(f"  {'E2: Leave-spruce-out (train pine, test spruce)':<45}  "
      f"{results['E2_leave_spruce_out']['metrics']['balanced_accuracy']:>6.3f}  "
      f"{results['E2_leave_spruce_out']['metrics']['per_class_recall'].get('coniferous',0):>13.3f}")
print(f"  {'E3: Leave-pine-out (train spruce, test pine)':<45}  "
      f"{results['E3_leave_pine_out']['metrics']['balanced_accuracy']:>6.3f}  "
      f"{results['E3_leave_pine_out']['metrics']['per_class_recall'].get('coniferous',0):>13.3f}")
print(f"  {'E6: Leave-HY-spruce-out (hardest combo)':<45}  "
      f"{results['E6_leave_HY_spruce_out']['metrics']['balanced_accuracy']:>6.3f}  "
      f"{results['E6_leave_HY_spruce_out']['hy_spruce_recall']:>13.3f}")
print(f"  {'E7: Leave-BK-spruce-out (plantation, new biome)':<45}  "
      f"{results['E7_leave_BK_spruce_out']['metrics']['balanced_accuracy']:>6.3f}  "
      f"{results['E7_leave_BK_spruce_out']['bk_spruce_recall']:>13.3f}")
print("="*70)
print(f"\nSpectral distances (normalised feature space):")
d = results["E5_species_spectra"]["euclidean_distances"]
print(f"  Spruce vs Pine       : {d['spruce_vs_pine']:.3f}")
print(f"  Spruce vs Broadleaved: {d['spruce_vs_broadleaved']:.3f}")
print(f"  Pine   vs Broadleaved: {d['pine_vs_broadleaved']:.3f}")
