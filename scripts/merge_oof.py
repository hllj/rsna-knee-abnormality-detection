"""
Merge per-fold OOF predictions into one teacher file for pseudo-labelling.

    python scripts/merge_oof.py --exp coatnet2_384      # -> models/oof/coatnet2_384.csv
"""
import argparse
import glob
import os

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True)
ap.add_argument("--oof_dir", default="models/oof")
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(a.oof_dir, f"{a.exp}_oof_fold*.csv")))
assert files, f"no OOF files for {a.exp} in {a.oof_dir}"
df = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates("StudyInstanceUID")
out = os.path.join(a.oof_dir, f"{a.exp}.csv")
df.to_csv(out, index=False)
print(f"merged {len(files)} folds, {len(df)} studies -> {out}")
