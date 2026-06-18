"""
Extract stand-level ALS structural features from raw LAS point clouds.

For each stand: merges all flight-line LAS files, computes ground-normalized
heights using classified ground returns (class=2), then derives structural
metrics: height percentiles, cover fraction, rugosity.

Output: outputs/als_features.csv  (58 rows × 9 feature columns + stand_id)
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import laspy
except ImportError:
    sys.exit("laspy not installed. Run: pip install laspy[lazrs]")

ALS_ROOT = Path(
    "C:/Users/aggarwm1/Videos/multi-disciplinary_Hyperspectral/model/data"
    "/Airborne_laser_scanning/Airborne_data/Airborne_laser_scanning"
    "/Analysis_ready_subsets"
)
OUT_CSV = Path("C:/Users/aggarwm1/Videos/multi-disciplinary_Hyperspectral/outputs/als_features.csv")
MIN_VEG_HEIGHT = 2.0   # metres above ground — below this = ground/shrub, not canopy
COVER_THRESHOLD = 2.0  # metres for cover fraction computation


def extract_stand_features(stand_dir: Path) -> dict:
    las_files = sorted(stand_dir.glob("*.las"))
    if not las_files:
        return None

    z_all, ret_all, cls_all = [], [], []
    for lf_path in las_files:
        lf = laspy.read(lf_path)
        z_all.extend(np.asarray(lf.z, dtype=np.float32))
        ret_all.extend(np.asarray(lf.return_number, dtype=np.uint8))
        cls_all.extend(np.asarray(lf.classification, dtype=np.uint8))

    z = np.array(z_all)
    ret = np.array(ret_all)
    cls = np.array(cls_all)

    ground_mask = cls == 2
    veg_mask = ~ground_mask  # all non-ground points (class 1=unclassified, 4=medium veg, etc.)

    if ground_mask.sum() < 10:
        return None

    # Ground elevation estimate: 5th percentile of ground-classified points
    ground_z = np.percentile(z[ground_mask], 5)

    # Normalised heights for vegetation points
    h_veg = z[veg_mask] - ground_z
    h_veg = h_veg[h_veg >= 0]  # drop below-ground artefacts

    if len(h_veg) < 10:
        return None

    # First returns only for cover fraction
    first_mask = ret == 1
    h_first = z[first_mask] - ground_z
    h_first = h_first[h_first >= 0]

    n_first_above = (h_first >= COVER_THRESHOLD).sum()
    cover_fraction = n_first_above / max(len(h_first), 1)

    feats = {
        "h_max": float(h_veg.max()),
        "h_mean": float(h_veg.mean()),
        "h_std": float(h_veg.std()),
        "h_p25": float(np.percentile(h_veg, 25)),
        "h_p50": float(np.percentile(h_veg, 50)),
        "h_p75": float(np.percentile(h_veg, 75)),
        "h_p95": float(np.percentile(h_veg, 95)),
        "cover_fraction": cover_fraction,
        "n_veg_pts": int(veg_mask.sum()),
    }
    return feats


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    records = []
    stand_dirs = sorted(ALS_ROOT.iterdir())
    print(f"Processing {len(stand_dirs)} stands...")

    for stand_dir in stand_dirs:
        if not stand_dir.is_dir():
            continue
        stand_id = stand_dir.name
        feats = extract_stand_features(stand_dir)
        if feats is None:
            print(f"  SKIP {stand_id}: insufficient data")
            continue
        feats["stand_id"] = stand_id
        records.append(feats)
        print(f"  {stand_id}: h_max={feats['h_max']:.1f}m  cover={feats['cover_fraction']:.2f}")

    df = pd.DataFrame(records)
    cols = ["stand_id", "h_max", "h_mean", "h_std", "h_p25", "h_p50", "h_p75", "h_p95", "cover_fraction", "n_veg_pts"]
    df = df[cols]
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} stands to {OUT_CSV}")


if __name__ == "__main__":
    main()
