"""Build a pseudo-label teacher file from a model's 5-fold out-of-fold predictions.

Pools <exp>_oof_fold{0..4}.csv (each study predicted by the fold model that never trained on it) and quantile-maps
every finding onto the report-label distribution of the same studies. The teacher keeps its ranking, but its values
live on the label scale. pos_weight-trained teachers (Raptor) predict systematically high probabilities, and a raw
50/50 mix would shift every target upward.

  python scripts/make_pseudo_teacher.py --oof runs/EXP-051/oof/raptor_repro_oof_fold{0,1,2,3,4}.csv \
      --targets runs/EXP-041/models/oof/raptor_repro_report_targets.csv --out configs/teachers/raptor_repro5_qmap.csv
"""
import argparse

import numpy as np
import pandas as pd

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
          "Synovitis", "Baker's", "Contusion", "Fracture"]
ID = "StudyInstanceUID"

ap = argparse.ArgumentParser()
ap.add_argument("--oof", nargs="+", required=True)
ap.add_argument("--targets", required=True, help="report-label targets (uid + 12 columns) of the same studies")
ap.add_argument("--out", required=True)
a = ap.parse_args()

oof = pd.concat([pd.read_csv(p, dtype={ID: str}) for p in a.oof]).drop_duplicates(ID).set_index(ID)[LABELS]
tgt = pd.read_csv(a.targets, dtype={ID: str}).set_index(ID).reindex(oof.index)[LABELS]
assert tgt.notna().all().all(), "teacher studies without report targets"
out = pd.DataFrame(index=oof.index, columns=LABELS, dtype=float)
for c in LABELS:
    q = oof[c].rank(pct=True, method="average").to_numpy()        # teacher ranking in (0, 1]
    ref = np.sort(tgt[c].to_numpy())                                # label distribution of the same studies
    out[c] = np.interp(q, np.linspace(1 / len(ref), 1, len(ref)), ref)
out.reset_index().to_csv(a.out, index=False, float_format="%.5f")
print(f"{len(out)} studies -> {a.out}")
print("mean teacher (raw) vs mapped vs labels:")
for c in LABELS:
    print(f"  {c:17s} {oof[c].mean():.3f} -> {out[c].mean():.3f} | labels {tgt[c].mean():.3f}")
