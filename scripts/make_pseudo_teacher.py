"""Build a pseudo-label teacher file from 5-fold out-of-fold predictions.

Each --oof group is one model family: its <exp>_oof_fold{0..4}.csv are pooled (every study predicted by the fold model
that never trained on it). Several groups are rank-averaged per finding (a multi-architecture teacher). The result is
quantile-mapped onto the report-label distribution of the same studies: the teacher keeps its ranking, but its values
live on the label scale. pos_weight-trained teachers (Raptor) predict systematically high probabilities, and a raw
50/50 mix would shift every target upward.

Label reference: --targets (a csv) or --exp (the experiment's own label blend via build_targets, pseudo disabled; needs
the label tables, i.e. run on a training box). Without either, the rank-averaged teacher is written as is (values =
percentile ranks) so the mapping can be done later from --rank_file.

  python scripts/make_pseudo_teacher.py --oof runs/EXP-051/oof/raptor_repro_oof_fold{0,1,2,3,4}.csv \
      --targets runs/EXP-041/models/oof/raptor_repro_report_targets.csv --out configs/teachers/raptor_repro5_qmap.csv
  python scripts/make_pseudo_teacher.py --oof runs/EXP-027/oof/convnext_small_288_dense_oof_fold?.csv \
      --oof runs/EXP-051/oof/raptor_repro_oof_fold?.csv --out configs/teachers/cnx5_raptor5_rank.csv
  python scripts/make_pseudo_teacher.py --rank_file configs/teachers/cnx5_raptor5_rank.csv \
      --exp raptor_v3_d96_alldata_pl --out configs/teachers/cnx5_raptor5_qmap_jev.csv
"""
import argparse

import numpy as np
import pandas as pd

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
          "Synovitis", "Baker's", "Contusion", "Fracture"]
ID = "StudyInstanceUID"

ap = argparse.ArgumentParser()
ap.add_argument("--oof", nargs="+", action="append", default=[], help="one model family's fold OOF csvs (repeatable)")
ap.add_argument("--rank_file", default=None, help="a rank-averaged teacher written earlier (instead of --oof)")
ap.add_argument("--targets", default=None, help="report-label targets csv (uid + 12 columns)")
ap.add_argument("--exp", default=None, help="map onto this experiment's label blend (build_targets, pseudo disabled)")
ap.add_argument("--out", required=True)
a = ap.parse_args()

if a.rank_file:
    teacher = pd.read_csv(a.rank_file, dtype={ID: str}).set_index(ID)[LABELS]
else:
    fams = []
    for group in a.oof:
        oof = pd.concat([pd.read_csv(p, dtype={ID: str}) for p in group]).drop_duplicates(ID).set_index(ID)[LABELS]
        fams.append(oof.rank(pct=True))
        print(f"family: {len(group)} files, {len(oof)} studies")
    common = sorted(set.intersection(*(set(f.index) for f in fams)))
    teacher = sum(f.loc[common] for f in fams) / len(fams)

if a.exp:
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root: import src.*
    from omegaconf import OmegaConf
    from src.config import get_config
    from src.labeling.blend import build_targets
    cfg = get_config([f"exp={a.exp}"])
    tgt_all = build_targets(OmegaConf.merge(cfg, {"labels": {"pseudo": {"enabled": False}}}))
elif a.targets:
    tgt_all = pd.read_csv(a.targets, dtype={ID: str}).set_index(ID)
else:
    teacher.reset_index().to_csv(a.out, index=False, float_format="%.6f")
    print(f"{len(teacher)} studies (rank-averaged, not mapped) -> {a.out}")
    raise SystemExit

tgt = tgt_all.reindex(teacher.index)[LABELS]
assert tgt.notna().all().all(), "teacher studies without report targets"
out = pd.DataFrame(index=teacher.index, columns=LABELS, dtype=float)
for c in LABELS:
    q = teacher[c].rank(pct=True, method="average").to_numpy()      # teacher ranking in (0, 1]
    ref = np.sort(tgt[c].to_numpy())                                # label distribution of the same studies
    out[c] = np.interp(q, np.linspace(1 / len(ref), 1, len(ref)), ref)
out.reset_index().to_csv(a.out, index=False, float_format="%.5f")
print(f"{len(out)} studies -> {a.out}")
print("mean mapped teacher vs labels:")
for c in LABELS:
    print(f"  {c:17s} {out[c].mean():.3f} | labels {tgt[c].mean():.3f}")
