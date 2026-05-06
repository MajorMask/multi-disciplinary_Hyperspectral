"""
Peatland Hyperspectral Analysis — Topic F
==========================================
Research question:
  Can peatland type (Finnish_peatland_type) and geographic location
  (Coordinate_y / Coordinate_x) be recovered from reflectance spectra?

Dataset: Salko et al. 2024. Mendeley Data doi:10.17632/3866tj3w8v.1
Columns confirmed from header: see COLUMN_NOTES below.

Usage
-----
Place the raw CSV (e.g. 'spectra_raw.csv' or 'spectra_smoothed.csv')
in the same directory and run:

    python peatland_hyperspectral_analysis.py

All figures are saved to ./figures/ ; all model results to ./results/.
"""

# ── COLUMN NOTES ─────────────────────────────────────────────────────────────
# Metadata       : Plot_ID, Country, Site, Date, Coordinate_y, Coordinate_x
# Finnish target : Finnish_peatland_type   (NaN for Estonian plots)
# Tree basal area: BA_Pine_living/dead, BA_Spruce_living/dead,
#                  BA_Deciduous_living/dead
# PFT fractions  : PFT_bare_peat, PFT_brown_mosses, PFT_graminoids,
#                  PFT_herbaceous, PFT_lichens, PFT_litter,
#                  PFT_other_mosses, PFT_sphagnum_mosses,
#                  PFT_woody_stemmed, PFT_water, PFT_unidentified
# Quality flag   : SWIR_class
# Spectra        : wl350 … wl2500  (2151 bands, 1 nm step)
# ─────────────────────────────────────────────────────────────────────────────

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from scipy.signal import savgol_filter

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Ridge, Lasso, LassoCV
from sklearn.model_selection import (GroupKFold, cross_val_score,
                                     cross_val_predict, StratifiedGroupKFold)
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, r2_score, mean_squared_error)
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"figure.dpi": 150, "font.size": 9,
                             "axes.spines.top": False, "axes.spines.right": False})

os.makedirs("figures", exist_ok=True)
os.makedirs("results",  exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 0. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CSV_FILE  = "C:\\Users\\aggarwm1\\Videos\\multi-disciplinary_Hyperspectral\\data\\Hyperspectral\\Airborne_data\\Reflectance_spectra_of_peatland_vegetation_Finland_Estonia_smoothed.csv"   # change to spectra_smoothed.csv if available
SEPARATOR = ","
TARGET_COL = "Finnish_peatland_type"
SITE_COL   = "Site"
LAT_COL    = "Coordinate_y"
LON_COL    = "Coordinate_x"

# Noisy water-absorption bands (nm) — always exclude from modelling
NOISY_BANDS = list(range(1330, 1550)) + list(range(1761, 2025)) + list(range(2311, 2501))
NOISY_COLS  = {f"wl{b}" for b in NOISY_BANDS}

# Spectral regions of biological interest
REGIONS = {
    "VIS (350–700)"   : (350,  700),
    "Red-edge (700–800)": (700, 800),
    "NIR (800–1330)"  : (800, 1330),
    "SWIR-1 (1550–1760)": (1550, 1760),
    "SWIR-2 (2025–2310)": (2025, 2311),
}

PFT_COLS = [
    "PFT_bare_peat", "PFT_brown_mosses", "PFT_graminoids", "PFT_herbaceous",
    "PFT_lichens", "PFT_litter", "PFT_other_mosses", "PFT_sphagnum_mosses",
    "PFT_woody_stemmed", "PFT_water", "PFT_unidentified",
]
BA_COLS = [
    "BA_Pine_living", "BA_Pine_dead", "BA_Spruce_living", "BA_Spruce_dead",
    "BA_Deciduous_living", "BA_Deciduous_dead",
]

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & AUDIT
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=SEPARATOR, low_memory=False)
    print(f"\n{'─'*60}")
    print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Sites   : {df[SITE_COL].nunique()} unique — {sorted(df[SITE_COL].unique())}")
    print(f"Country : {df['Country'].value_counts().to_dict()}")
    print(f"\nMissing values (non-spectral):")
    meta_cols = [TARGET_COL, SITE_COL, LAT_COL, LON_COL, "SWIR_class"] + PFT_COLS + BA_COLS
    print(df[meta_cols].isna().sum().to_string())
    
    # Convert coordinate and basal area columns to numeric
    df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors='coerce')
    df[LON_COL] = pd.to_numeric(df[LON_COL], errors='coerce')
    for col in BA_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def get_spectral_cols(df: pd.DataFrame, exclude_noisy: bool = True) -> list:
    all_spec = [c for c in df.columns if c.startswith("wl")]
    if exclude_noisy:
        return [c for c in all_spec if c not in NOISY_COLS]
    return all_spec


def wavelengths(cols: list) -> np.ndarray:
    return np.array([int(c[2:]) for c in cols])


# ══════════════════════════════════════════════════════════════════════════════
# 2. EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def plot_class_distribution(df: pd.DataFrame):
    """Bar chart of peatland types (Finnish plots only)."""
    fi = df.dropna(subset=[TARGET_COL])
    counts = fi[TARGET_COL].value_counts()
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(counts.index, counts.values, color="#3B7DBE", edgecolor="white", height=0.7)
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_xlabel("Number of plots")
    ax.set_title(f"Peatland type distribution (n={len(fi)} Finnish plots)")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig("figures/01_class_distribution.png", bbox_inches="tight")
    plt.close()
    print(f"\nClass counts:\n{counts.to_string()}")
    return counts


def plot_geographic_distribution(df: pd.DataFrame):
    """Scatter map of all plots coloured by Site."""
    sites = df[SITE_COL].unique()
    cmap  = plt.cm.get_cmap("tab20", len(sites))
    site2color = {s: cmap(i) for i, s in enumerate(sites)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: by Site
    ax = axes[0]
    for site in sites:
        sub = df[df[SITE_COL] == site]
        ax.scatter(sub[LON_COL], sub[LAT_COL], s=18, alpha=0.75,
                   label=site, color=site2color[site])
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Plots by site")
    ax.legend(fontsize=6, ncol=2, loc="lower right")

    # Panel 2: by peatland type (Finnish only)
    ax = axes[1]
    fi  = df.dropna(subset=[TARGET_COL])
    est = df[df["Country"] == "EE"]
    types = sorted(fi[TARGET_COL].unique())
    tcmap = plt.cm.get_cmap("Set2", len(types))
    for i, t in enumerate(types):
        sub = fi[fi[TARGET_COL] == t]
        ax.scatter(sub[LON_COL], sub[LAT_COL], s=18, alpha=0.8,
                   label=t, color=tcmap(i))
    ax.scatter(est[LON_COL], est[LAT_COL], s=18, alpha=0.5,
               color="gray", marker="^", label="Estonia (no type)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Plots by Finnish peatland type")
    ax.legend(fontsize=6, ncol=2, loc="lower right")

    plt.suptitle("Geographic distribution of 446 peatland plots", fontsize=10)
    plt.tight_layout()
    fig.savefig("figures/02_geographic_distribution.png", bbox_inches="tight")
    plt.close()


def plot_mean_spectra_by_type(df: pd.DataFrame, spec_cols: list, wls: np.ndarray):
    """Mean ± std spectra for each peatland type."""
    fi = df.dropna(subset=[TARGET_COL])
    types = sorted(fi[TARGET_COL].unique())
    cmap  = plt.cm.get_cmap("Set2", len(types))

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, t in enumerate(types):
        sub = fi[fi[TARGET_COL] == t][spec_cols].values.astype(float)
        mu  = np.nanmean(sub, axis=0)
        sd  = np.nanstd(sub, axis=0)
        c   = cmap(i)
        ax.plot(wls, mu, label=f"{t} (n={len(sub)})", color=c, lw=1.2)
        ax.fill_between(wls, mu - sd, mu + sd, color=c, alpha=0.12)

    # Shade noisy regions
    for r in [(1330, 1549), (1761, 2024), (2311, 2500)]:
        ax.axvspan(r[0], r[1], color="gray", alpha=0.15)
    ax.text(1420, ax.get_ylim()[1] * 0.95, "noise", ha="center",
            fontsize=7, color="gray", va="top")
    ax.text(1890, ax.get_ylim()[1] * 0.95, "noise", ha="center",
            fontsize=7, color="gray", va="top")

    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Reflectance factor")
    ax.set_title("Mean (±1 SD) reflectance spectra by peatland type")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    plt.tight_layout()
    fig.savefig("figures/03_mean_spectra_by_type.png", bbox_inches="tight")
    plt.close()


def plot_mean_spectra_by_site(df: pd.DataFrame, spec_cols: list, wls: np.ndarray):
    """Mean spectra per site — reveals geographic spectral gradient."""
    sites = sorted(df[SITE_COL].unique())
    lats  = df.groupby(SITE_COL)[LAT_COL].mean().reindex(sites).values
    norm  = plt.Normalize(lats.min(), lats.max())
    cmap  = plt.cm.RdYlGn_r  # warm = south, cool = north

    fig, ax = plt.subplots(figsize=(13, 5))
    for s, lat in zip(sites, lats):
        sub = df[df[SITE_COL] == s][spec_cols].values.astype(float)
        mu  = np.nanmean(sub, axis=0)
        ax.plot(wls, mu, color=cmap(norm(lat)), lw=1.0, label=f"{s} ({lat:.1f}°N)")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Mean latitude (°N)")
    for r in [(1330, 1549), (1761, 2024), (2311, 2500)]:
        ax.axvspan(r[0], r[1], color="gray", alpha=0.15)
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Reflectance factor")
    ax.set_title("Mean spectra by site, colour-coded by latitude")
    ax.legend(fontsize=5.5, ncol=2, loc="upper right")
    plt.tight_layout()
    fig.savefig("figures/04_mean_spectra_by_site_latitude.png", bbox_inches="tight")
    plt.close()


def plot_pft_composition(df: pd.DataFrame):
    """Stacked bar chart of mean PFT cover by peatland type."""
    fi = df.dropna(subset=[TARGET_COL]).copy()
    pft_labels = [c.replace("PFT_", "").replace("_", " ") for c in PFT_COLS]
    grp = fi.groupby(TARGET_COL)[PFT_COLS].mean()

    cmap   = plt.cm.get_cmap("Paired", len(PFT_COLS))
    colors = [cmap(i) for i in range(len(PFT_COLS))]

    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = np.zeros(len(grp))
    for j, (col, lbl) in enumerate(zip(PFT_COLS, pft_labels)):
        vals = grp[col].fillna(0).values
        ax.bar(grp.index, vals, bottom=bottom, label=lbl, color=colors[j])
        bottom += vals

    ax.set_ylabel("Mean fractional cover"); ax.set_ylim(0, 1.05)
    ax.set_title("Mean PFT composition by peatland type")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.tight_layout()
    fig.savefig("figures/05_pft_composition_by_type.png", bbox_inches="tight")
    plt.close()


def plot_basal_area(df: pd.DataFrame):
    """Total basal area (living pine + spruce + deciduous) by site."""
    fi = df.dropna(subset=[TARGET_COL]).copy()
    fi["BA_total_living"] = (fi["BA_Pine_living"].fillna(0)
                             + fi["BA_Spruce_living"].fillna(0)
                             + fi["BA_Deciduous_living"].fillna(0))
    fig, ax = plt.subplots(figsize=(9, 4))
    fi.boxplot(column="BA_total_living", by=TARGET_COL, ax=ax,
               vert=True, patch_artist=True)
    ax.set_xlabel("Peatland type"); ax.set_ylabel("Total living basal area (m²/ha)")
    ax.set_title("Living basal area by peatland type")
    plt.suptitle("")
    plt.xticks(rotation=30, ha="right", fontsize=7)
    plt.tight_layout()
    fig.savefig("figures/06_basal_area_by_type.png", bbox_inches="tight")
    plt.close()


def plot_spectral_band_discriminability(df: pd.DataFrame,
                                        spec_cols: list, wls: np.ndarray):
    """ANOVA F-score per wavelength — shows which bands discriminate types."""
    fi    = df.dropna(subset=[TARGET_COL]).copy()
    X     = fi[spec_cols].values.astype(float)
    y     = LabelEncoder().fit_transform(fi[TARGET_COL])
    valid = ~np.isnan(X).any(axis=0)
    f_sc  = np.full(len(spec_cols), np.nan)
    mi    = np.full(len(spec_cols), np.nan)

    f_sc[valid], _  = f_classif(X[:, valid], y)
    mi[valid]        = mutual_info_classif(X[:, valid], y, random_state=42)

    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    for ax, vals, label, color in zip(
            axes, [f_sc, mi],
            ["ANOVA F-score", "Mutual information (bits)"],
            ["#3B7DBE", "#E07B39"]):
        ax.plot(wls, vals, color=color, lw=0.8)
        ax.fill_between(wls, 0, vals, color=color, alpha=0.25)
        for r in [(1330, 1549), (1761, 2024), (2311, 2500)]:
            ax.axvspan(r[0], r[1], color="gray", alpha=0.15)
        ax.set_ylabel(label)

    # Annotate key regions
    for band_nm, label in [(720, "Red-edge"), (970, "Water abs."), (1200, "NIR shoulder"),
                            (2100, "SWIR")]:
        if wls.min() < band_nm < wls.max():
            axes[0].axvline(band_nm, color="k", lw=0.6, linestyle="--", alpha=0.4)
            axes[0].text(band_nm + 5, axes[0].get_ylim()[1] * 0.9, label,
                         fontsize=6.5, color="k", va="top", rotation=90)

    axes[1].set_xlabel("Wavelength (nm)")
    axes[0].set_title("Per-band discriminability for Finnish_peatland_type classification")
    plt.tight_layout()
    fig.savefig("figures/07_band_discriminability.png", bbox_inches="tight")
    plt.close()

    # Print top 10 most discriminative bands
    top10 = wls[np.argsort(f_sc)[::-1][:10]]
    print(f"\nTop 10 most discriminative wavelengths (ANOVA): {top10} nm")
    return f_sc, mi


# ══════════════════════════════════════════════════════════════════════════════
# 3. DIMENSIONALITY REDUCTION
# ══════════════════════════════════════════════════════════════════════════════

def run_pca(df: pd.DataFrame, spec_cols: list) -> tuple:
    """PCA on all 446 plots. Returns (pca, X_pca, scaler)."""
    X      = df[spec_cols].values.astype(float)
    # Impute column-wise mean for any residual NaNs
    col_means = np.nanmean(X, axis=0)
    inds   = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    pca    = PCA(n_components=30, random_state=42)
    X_pca  = pca.fit_transform(X_sc)
    return pca, X_pca, scaler, X_sc


def plot_pca_variance(pca: PCA):
    ev  = pca.explained_variance_ratio_
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(range(1, len(ev)+1), ev * 100, color="#3B7DBE")
    axes[0].set_xlabel("Principal component"); axes[0].set_ylabel("Variance explained (%)")
    axes[0].set_title("Scree plot")
    axes[1].plot(range(1, len(ev)+1), np.cumsum(ev)*100, marker="o", ms=4, color="#3B7DBE")
    axes[1].axhline(95, color="r", lw=0.8, linestyle="--", label="95%")
    axes[1].set_xlabel("# components"); axes[1].set_ylabel("Cumulative variance (%)")
    axes[1].set_title("Cumulative variance"); axes[1].legend()
    plt.tight_layout()
    fig.savefig("figures/08_pca_variance.png", bbox_inches="tight")
    plt.close()
    n95 = int(np.argmax(np.cumsum(ev) >= 0.95)) + 1
    print(f"\nPCA: {n95} components explain ≥95% of spectral variance")
    return n95


def plot_pca_biplots(df: pd.DataFrame, X_pca: np.ndarray):
    """PC1 vs PC2, colored by type and by latitude."""
    fi_mask = df[TARGET_COL].notna().values
    types   = df.loc[fi_mask, TARGET_COL].values
    utypes  = sorted(set(types))
    tcmap   = plt.cm.get_cmap("Set2", len(utypes))
    t2c     = {t: tcmap(i) for i, t in enumerate(utypes)}

    lats    = df[LAT_COL].values
    lat_norm = plt.Normalize(np.nanmin(lats), np.nanmax(lats))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: by type
    ax = axes[0]
    ax.scatter(X_pca[~fi_mask, 0], X_pca[~fi_mask, 1],
               s=12, alpha=0.35, color="lightgray", label="Estonia")
    for t in utypes:
        idx = fi_mask & (df[TARGET_COL] == t).values
        ax.scatter(X_pca[idx, 0], X_pca[idx, 1],
                   s=16, alpha=0.75, color=t2c[t], label=t)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("PCA — coloured by peatland type")
    ax.legend(fontsize=6, ncol=2)

    # Panel 2: by latitude (all plots)
    ax = axes[1]
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=lats, s=14,
                    alpha=0.75, cmap="RdYlGn_r", norm=lat_norm)
    plt.colorbar(sc, ax=ax, label="Latitude (°N)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("PCA — coloured by latitude")

    plt.tight_layout()
    fig.savefig("figures/09_pca_biplots.png", bbox_inches="tight")
    plt.close()


def plot_pca_loadings(pca: PCA, spec_cols: list, wls: np.ndarray, n_comp: int = 4):
    """PC loading vectors — which wavelengths drive each component."""
    fig, axes = plt.subplots(n_comp, 1, figsize=(13, 3 * n_comp), sharex=True)
    for i, ax in enumerate(axes):
        load = pca.components_[i]
        ax.plot(wls, load, lw=0.9, color="#3B7DBE")
        ax.axhline(0, color="k", lw=0.5)
        ax.fill_between(wls, load, 0,
                        where=(load > 0), color="#3B7DBE", alpha=0.25)
        ax.fill_between(wls, load, 0,
                        where=(load < 0), color="#E07B39", alpha=0.25)
        for r in [(1330, 1549), (1761, 2024), (2311, 2500)]:
            ax.axvspan(r[0], r[1], color="gray", alpha=0.15)
        ev_pct = pca.explained_variance_ratio_[i] * 100
        ax.set_ylabel(f"PC{i+1} loading\n({ev_pct:.1f}%)", fontsize=8)
    axes[-1].set_xlabel("Wavelength (nm)")
    axes[0].set_title("PCA loading vectors")
    plt.tight_layout()
    fig.savefig("figures/10_pca_loadings.png", bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 4. CLASSIFICATION — Peatland type recovery
# ══════════════════════════════════════════════════════════════════════════════

def prepare_classification_data(df: pd.DataFrame, spec_cols: list,
                                 min_class_size: int = 5):
    """
    Returns (X_scaled, y, groups, label_names, scaler)
    using only Finnish plots with class size >= min_class_size.
    """
    fi   = df.dropna(subset=[TARGET_COL]).copy()

    # Drop very rare classes (can't do leave-one-site-out with < min_class_size)
    counts = fi[TARGET_COL].value_counts()
    keep   = counts[counts >= min_class_size].index
    fi     = fi[fi[TARGET_COL].isin(keep)].copy()
    dropped = set(counts.index) - set(keep)
    if dropped:
        print(f"\nDropped rare classes (n < {min_class_size}): {dropped}")

    X      = fi[spec_cols].values.astype(float)
    col_means = np.nanmean(X, axis=0)
    inds   = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

    le     = LabelEncoder()
    y      = le.fit_transform(fi[TARGET_COL])
    groups = fi[SITE_COL].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    print(f"\nClassification dataset: {len(fi)} samples, "
          f"{len(le.classes_)} classes, "
          f"{fi[SITE_COL].nunique()} sites")
    return X_sc, y, groups, le.classes_, scaler


def site_leave_one_out_cv(clf, X, y, groups, label_names):
    """
    GroupKFold = leave-one-site-out cross-validation.
    Returns dict with per-fold accuracy and aggregated report.
    """
    gkf   = GroupKFold(n_splits=len(np.unique(groups)))
    y_pred_all = np.empty_like(y)
    accs  = []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        clf.fit(X[tr], y[tr])
        preds = clf.predict(X[te])
        y_pred_all[te] = preds
        acc = (preds == y[te]).mean()
        accs.append(acc)
        site = np.unique(groups[te])[0]
        print(f"  Held-out site: {site:20s}  acc={acc:.3f}")
    mean_acc = np.mean(accs)
    std_acc  = np.std(accs)
    print(f"\n  Mean accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
    report = classification_report(y, y_pred_all,
                                   target_names=label_names, output_dict=True)
    return mean_acc, std_acc, y_pred_all, report


def run_all_classifiers(X, y, groups, label_names):
    """Benchmark: PLS-DA (via LDA on PLS scores), RF, SVM."""
    results = {}

    # ── PLS-DA ────────────────────────────────────────────────────────────
    print("\n── PLS-DA (10 components) ──")
    pls = PLSRegression(n_components=10, max_iter=500)
    # Encode y as one-hot for PLS then take max column as prediction
    from sklearn.preprocessing import OneHotEncoder
    ohe  = OneHotEncoder(sparse_output=False)
    Y_oh = ohe.fit_transform(y.reshape(-1, 1))

    gkf = GroupKFold(n_splits=len(np.unique(groups)))
    y_pred_pls = np.empty_like(y)
    for tr, te in gkf.split(X, y, groups):
        pls.fit(X[tr], Y_oh[tr])
        pred_oh = pls.predict(X[te])
        y_pred_pls[te] = pred_oh.argmax(axis=1)

    acc_pls = (y_pred_pls == y).mean()
    print(f"  Overall accuracy: {acc_pls:.3f}")
    print(classification_report(y, y_pred_pls, target_names=label_names))
    results["PLS-DA"] = {"acc": acc_pls, "y_pred": y_pred_pls}

    # ── Random Forest ──────────────────────────────────────────────────────
    print("\n── Random Forest (500 trees) ──")
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                random_state=42, n_jobs=-1)
    acc_rf, std_rf, y_pred_rf, rep_rf = site_leave_one_out_cv(rf, X, y, groups, label_names)
    results["Random Forest"] = {"acc": acc_rf, "std": std_rf,
                                 "y_pred": y_pred_rf, "report": rep_rf}

    # ── SVM (RBF) ──────────────────────────────────────────────────────────
    print("\n── SVM (RBF kernel) ──")
    svm = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced",
              random_state=42)
    acc_sv, std_sv, y_pred_sv, rep_sv = site_leave_one_out_cv(svm, X, y, groups, label_names)
    results["SVM (RBF)"] = {"acc": acc_sv, "std": std_sv,
                             "y_pred": y_pred_sv, "report": rep_sv}

    return results


def plot_confusion_matrices(results: dict, y: np.ndarray, label_names: np.ndarray):
    n   = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y, res["y_pred"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(label_names)))
        ax.set_yticks(range(len(label_names)))
        ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(label_names, fontsize=6)
        for i in range(len(label_names)):
            for j in range(len(label_names)):
                ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                        fontsize=6, color="white" if cm_norm[i,j] > 0.5 else "black")
        ax.set_title(f"{name}\nOA={res['acc']:.3f}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle("Confusion matrices — leave-one-site-out CV", fontsize=10)
    plt.tight_layout()
    fig.savefig("figures/11_confusion_matrices.png", bbox_inches="tight")
    plt.close()


def plot_rf_feature_importance(rf_model, spec_cols: list, wls: np.ndarray):
    """Train RF on all data, plot permutation-like feature importance."""
    fi_plot = plt.subplots(figsize=(13, 4))
    ax = fi_plot[1]
    imp = rf_model.feature_importances_
    ax.fill_between(wls[:len(spec_cols)], imp, color="#3B7DBE", alpha=0.7, lw=0)
    ax.plot(wls[:len(spec_cols)], imp, color="#1a4a7a", lw=0.6)
    for r in [(1330, 1549), (1761, 2024), (2311, 2500)]:
        ax.axvspan(r[0], r[1], color="gray", alpha=0.15)
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Feature importance")
    ax.set_title("Random Forest feature importance by wavelength")
    plt.tight_layout()
    fi_plot[0].savefig("figures/12_rf_feature_importance.png", bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 5. REGRESSION — Geographic location recovery
# ══════════════════════════════════════════════════════════════════════════════

def run_location_regression(df: pd.DataFrame, spec_cols: list):
    """
    Regress latitude (and longitude) from spectra.
    Uses leave-one-site-out CV.
    High R² → spectra encode geographic location.
    """
    X      = df[spec_cols].values.astype(float)
    col_means = np.nanmean(X, axis=0)
    inds   = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    groups = df[SITE_COL].values

    results = {}
    for target_name in [LAT_COL, LON_COL]:
        y_full = df[target_name].values
        valid = ~np.isnan(y_full)
        if not valid.any():
            print(f"No valid {target_name} values, skipping")
            continue
        y = y_full[valid]
        X_sc_valid = X_sc[valid]
        groups_valid = groups[valid]
        gkf = GroupKFold(n_splits=len(np.unique(groups_valid)))

        print(f"\n── PLS regression → {target_name} ──")
        pls = PLSRegression(n_components=10)
        y_pred_full = np.full_like(y_full, np.nan)
        valid_indices = np.where(valid)[0]
        for tr, te in gkf.split(X_sc_valid, y, groups_valid):
            pls.fit(X_sc_valid[tr], y[tr])
            y_pred_full[valid_indices[te]] = pls.predict(X_sc_valid[te]).ravel()
        valid_pred = valid & ~np.isnan(y_pred_full)
        r2   = r2_score(y_full[valid_pred], y_pred_full[valid_pred])
        rmse = np.sqrt(mean_squared_error(y_full[valid_pred], y_pred_full[valid_pred]))
        print(f"  R² = {r2:.4f},  RMSE = {rmse:.4f}")
        results[target_name] = {"y": y_full, "y_pred": y_pred_full, "r2": r2, "rmse": rmse}

        # Lasso for sparse band selection
        print(f"── Lasso → {target_name} ──")
        lasso = LassoCV(cv=5, max_iter=5000, random_state=42)
        y_pred_lasso_full = np.full_like(y_full, np.nan)
        for tr, te in gkf.split(X_sc_valid, y, groups_valid):
            lasso.fit(X_sc_valid[tr], y[tr])
            y_pred_lasso_full[valid_indices[te]] = lasso.predict(X_sc_valid[te])
        valid_pred_l = valid & ~np.isnan(y_pred_lasso_full)
        r2_l   = r2_score(y_full[valid_pred_l], y_pred_lasso_full[valid_pred_l])
        rmse_l = np.sqrt(mean_squared_error(y_full[valid_pred_l], y_pred_lasso_full[valid_pred_l]))
        print(f"  R² = {r2_l:.4f},  RMSE = {rmse_l:.4f}")
        results[f"{target_name}_lasso"] = {
            "y": y_full, "y_pred": y_pred_lasso_full,
            "r2": r2_l, "rmse": rmse_l,
            "coef": lasso.coef_,
        }

    return results


def plot_location_regression(reg_results: dict, df: pd.DataFrame,
                              spec_cols: list, wls: np.ndarray):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for row, tgt in enumerate([LAT_COL, LON_COL]):
        res = reg_results[tgt]
        ax  = axes[row][0]
        ax.scatter(res["y"], res["y_pred"], s=10, alpha=0.5,
                   c=df[SITE_COL].astype("category").cat.codes, cmap="tab20")
        lims = [min(res["y"].min(), res["y_pred"].min()) - 0.5,
                max(res["y"].max(), res["y_pred"].max()) + 0.5]
        ax.plot(lims, lims, "r--", lw=0.8)
        ax.set_xlabel(f"Observed {tgt}"); ax.set_ylabel(f"Predicted {tgt}")
        ax.set_title(f"PLS → {tgt}   R²={res['r2']:.3f}  RMSE={res['rmse']:.4f}")

        # Lasso coefficients
        res_l = reg_results[f"{tgt}_lasso"]
        ax    = axes[row][1]
        coef  = res_l["coef"]
        ax.fill_between(wls[:len(spec_cols)], coef, color="#E07B39", alpha=0.5, lw=0)
        ax.axhline(0, color="k", lw=0.5)
        for r in [(1330, 1549), (1761, 2024), (2311, 2500)]:
            ax.axvspan(r[0], r[1], color="gray", alpha=0.15)
        ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Lasso coefficient")
        ax.set_title(f"Lasso wavelength selection → {tgt}   R²={res_l['r2']:.3f}")

    plt.suptitle("Geographic location recovery from spectra", fontsize=10)
    plt.tight_layout()
    fig.savefig("figures/13_location_regression.png", bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 6. PFT CORRELATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def plot_pft_spectral_correlations(df: pd.DataFrame,
                                   spec_cols: list, wls: np.ndarray):
    """Pearson r between each PFT fractional cover and each wavelength."""
    X   = df[spec_cols].values.astype(float)
    fig, axes = plt.subplots(len(PFT_COLS), 1, figsize=(13, 2.2 * len(PFT_COLS)),
                             sharex=True)
    cmap = plt.cm.get_cmap("tab10", len(PFT_COLS))
    for i, (col, ax) in enumerate(zip(PFT_COLS, axes)):
        pft = df[col].values.astype(float)
        valid = ~np.isnan(pft)
        r_vals = np.array([
            np.corrcoef(pft[valid], X[valid, j])[0, 1]
            for j in range(X.shape[1])
        ])
        ax.fill_between(wls, r_vals, color=cmap(i), alpha=0.5, lw=0)
        ax.plot(wls, r_vals, color=cmap(i), lw=0.7)
        ax.axhline(0, color="k", lw=0.4)
        ax.set_ylim(-1, 1)
        ax.set_ylabel(col.replace("PFT_", "").replace("_", " "), fontsize=7)
        for r in [(1330, 1549), (1761, 2024), (2311, 2500)]:
            ax.axvspan(r[0], r[1], color="gray", alpha=0.12)
    axes[-1].set_xlabel("Wavelength (nm)")
    axes[0].set_title("Pearson r: PFT fractional cover vs. reflectance")
    plt.tight_layout()
    fig.savefig("figures/14_pft_spectral_correlations.png", bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 7. RESULTS SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def save_results_summary(clf_results: dict, reg_results: dict):
    rows = []
    for name, res in clf_results.items():
        rows.append({
            "Task": "Classification (peatland type)",
            "Method": name,
            "Metric": "Overall accuracy (LOSO-CV)",
            "Value": f"{res['acc']:.3f}",
        })
    for tgt in [LAT_COL, LON_COL]:
        for suffix, method in [("", "PLS"), ("_lasso", "Lasso")]:
            res = reg_results.get(f"{tgt}{suffix}")
            if res:
                rows.append({
                    "Task": f"Regression ({tgt})",
                    "Method": method,
                    "Metric": "R² (LOSO-CV)",
                    "Value": f"{res['r2']:.3f}",
                })
    df_res = pd.DataFrame(rows)
    df_res.to_csv("results/summary_table.csv", index=False)
    print(f"\n{'═'*60}")
    print("RESULTS SUMMARY")
    print(df_res.to_string(index=False))
    return df_res


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Peatland Hyperspectral Analysis — Topic F")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────
    df = load_data(CSV_FILE)

    spec_cols_all   = get_spectral_cols(df, exclude_noisy=False)
    spec_cols_clean = get_spectral_cols(df, exclude_noisy=True)
    wls_all         = wavelengths(spec_cols_all)
    wls_clean       = wavelengths(spec_cols_clean)

    print(f"\nTotal spectral bands : {len(spec_cols_all)}")
    print(f"After noise removal  : {len(spec_cols_clean)} bands")

    # ── EDA ───────────────────────────────────────────────────────────────
    print("\n[Step 2] Exploratory data analysis …")
    plot_class_distribution(df)
    plot_geographic_distribution(df)
    plot_mean_spectra_by_type(df, spec_cols_clean, wls_clean)
    plot_mean_spectra_by_site(df, spec_cols_clean, wls_clean)
    plot_pft_composition(df)
    plot_basal_area(df)
    f_scores, mi_scores = plot_spectral_band_discriminability(
        df, spec_cols_clean, wls_clean)

    # ── PCA ───────────────────────────────────────────────────────────────
    print("\n[Step 3] PCA …")
    pca, X_pca, scaler, X_sc = run_pca(df, spec_cols_clean)
    n95 = plot_pca_variance(pca)
    plot_pca_biplots(df, X_pca)
    plot_pca_loadings(pca, spec_cols_clean, wls_clean, n_comp=4)

    # ── Classification ────────────────────────────────────────────────────
    print("\n[Step 4] Classification — peatland type recovery …")
    X_clf, y_clf, groups_clf, label_names, _ = prepare_classification_data(
        df, spec_cols_clean)
    clf_results = run_all_classifiers(X_clf, y_clf, groups_clf, label_names)
    plot_confusion_matrices(clf_results, y_clf, label_names)

    # Train RF on all data for feature importance plot
    rf_full = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                     random_state=42, n_jobs=-1)
    rf_full.fit(X_clf, y_clf)
    plot_rf_feature_importance(rf_full, spec_cols_clean, wls_clean)

    # ── Location regression ────────────────────────────────────────────────
    print("\n[Step 5] Regression — geographic location recovery …")
    reg_results = run_location_regression(df, spec_cols_clean)
    plot_location_regression(reg_results, df, spec_cols_clean, wls_clean)

    # ── PFT correlations ──────────────────────────────────────────────────
    print("\n[Step 6] PFT–spectra correlations …")
    plot_pft_spectral_correlations(df, spec_cols_clean, wls_clean)

    # ── Summary ───────────────────────────────────────────────────────────
    save_results_summary(clf_results, reg_results)

    print("\nAll figures saved to ./figures/")
    print("All result tables saved to ./results/")
    print("Done.")


if __name__ == "__main__":
    main()