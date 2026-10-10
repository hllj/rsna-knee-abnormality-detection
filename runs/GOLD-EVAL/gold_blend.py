"""Gold-58 evaluation of every model family and of family-weighted rank blends, from an inference --save_raw npz
(run with --split train --fold_eval 99 --mirror_tta, i.e. the 58 gold studies only).

Reproduces src/models/inference.py: per checkpoint (plain + mirror)/2 when a mirror pass exists, rank01 per finding,
family weight split evenly over the family's checkpoints, weighted sum. Macro AUC = mean over the 12 findings.
  python gold_blend.py raw_gold.npz train.csv
"""
import itertools
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
          "Synovitis", "Baker's", "Contusion", "Fracture"]
FAM = [  # (regex on the checkpoint path, family, gold used for epoch selection?)
    (r"convnext-s-288-dense", "cnx_dense5", False),
    (r"raptor-repro", "raptor_repro5", False),
    (r"reader-repro", "reader_v1", False),
    (r"reader-v3", "reader_v3", True),
    (r"coatnet2-384-dense", "coatnet_f0", False),
    (r"raptor-alldata", "raptor_v1_all", False),
    (r"raptor-v2__pytorch__exp-064", "raptor_v2_80", True),
    (r"raptor-v2-d96__pytorch__exp-065", "d96_s1", True),
    (r"raptor-v2-d96__pytorch__exp-072", "d96_s2", True),
    (r"convnext-v2-d96", "cnx_d96", True),
]

raw = np.load(sys.argv[1])
ids = [str(x) for x in raw["ids"]]
tr = pd.read_csv(sys.argv[2], dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
y = tr.loc[ids, LABELS].to_numpy(float)
assert not np.isnan(y).any(), "non-gold studies in the raw file"


def rank01(p):
    return np.stack([(rankdata(p[:, j]) - 1) / max(len(p) - 1, 1) for j in range(p.shape[1])], 1)


def auc(p):
    return float(np.mean([roc_auc_score(y[:, j], p[:, j]) for j in range(len(LABELS))]))


ck = {}  # family -> list of (plain, mirror-averaged) predictions
for k in raw.files:
    if k == "ids" or k.endswith("#mirror"):
        continue
    fam = next((f for pat, f, _ in FAM if re.search(pat, k)), None)
    assert fam, f"unmapped checkpoint {k}"
    plain = raw[k]
    mir = (plain + raw[k + "#mirror"]) / 2 if k + "#mirror" in raw.files else plain
    ck.setdefault(fam, []).append((plain, mir))
biased = {f for _, f, b in FAM if b}

print(f"{len(ids)} gold studies | checkpoints per family:", {f: len(v) for f, v in ck.items()})
print("\n## 1. Families (rank-average of a family's checkpoints)")
famp = {}
for f in [f for _, f, _ in FAM if f in ck]:
    a_plain = auc(sum(rank01(p) for p, _ in ck[f]))
    famp[f] = sum(rank01(m) for _, m in ck[f]) / len(ck[f])
    print(f"  {f:15s} plain {a_plain:.4f} | with mirror {auc(famp[f]):.4f}"
          + ("   (gold used for epoch selection: optimistic)" if f in biased else ""))


def blend(w):
    tot = sum(w.values())
    return sum(wi * famp[f] for f, wi in w.items() if wi) / tot


print("\n## 2. Blends already scored on the public LB (calibration)")
known = {
    "EXP-055 (0.936)": {"cnx_dense5": .3, "raptor_repro5": .3, "reader_v1": .3, "coatnet_f0": .1},
    "EXP-070 (0.936)": {"cnx_dense5": .3, "raptor_repro5": .3, "reader_v1": .15, "reader_v3": .15, "coatnet_f0": .1},
    "EXP-060 (0.934)": {"cnx_dense5": .5, "raptor_repro5": .5},
    "EXP-058 (0.930)": {"raptor_v1_all": 1},
    "EXP-067 (0.933)": {"raptor_v2_80": 1},
    "EXP-068 (0.938)": {"d96_s1": 1},
    "EXP-071 (0.940)": {"cnx_dense5": .18, "raptor_repro5": .18, "reader_v1": .18, "coatnet_f0": .06, "d96_s1": .4},
}
for n, w in known.items():
    if all(f in famp for f in w):
        print(f"  {n:16s} gold {auc(blend(w)):.4f}")

print("\n## 3. Grid search over family weights (step 0.1)")
cand = [f for f in ["d96", "cnx_dense5", "raptor_repro5", "reader_v1", "coatnet_f0", "cnx_d96", "raptor_v2_80"]
        if f == "d96" or f in famp]
if "d96_s1" in famp and "d96_s2" in famp:
    famp["d96"] = (famp["d96_s1"] + famp["d96_s2"]) / 2  # the two seeds as one family (equal split)
    print(f"  d96 seeds 1+2 averaged: gold {auc(famp['d96']):.4f}")
else:
    famp["d96"] = famp["d96_s1"]
res = []
for combo in itertools.product(range(11), repeat=len(cand)):
    if sum(combo) != 10:
        continue
    w = {f: c / 10 for f, c in zip(cand, combo)}
    res.append((auc(blend(w)), w))
res.sort(key=lambda r: -r[0])
for a, w in res[:10]:
    print(f"  gold {a:.4f}  " + " ".join(f"{f}={v:.1f}" for f, v in w.items() if v))
top = res[:25]
avg = {f: float(np.mean([w[f] for _, w in top])) for f in cand}
print("  mean of the top-25 weight vectors (smoother pick): gold "
      f"{auc(blend(avg)):.4f}  " + " ".join(f"{f}={v:.2f}" for f, v in avg.items() if v >= 0.005))

print("\n## 4. Bootstrap: P(blend beats EXP-071 weights on resampled gold), 2,000 resamples")
ref = blend(known["EXP-071 (0.940)"])
rng = np.random.default_rng(0)
for name, p in [("grid best", blend(res[0][1])), ("top-25 mean", blend(avg)), ("d96 seeds alone", famp["d96"])]:
    wins = n = 0
    for _ in range(2000):
        i = rng.integers(0, len(ids), len(ids))
        try:
            d = np.mean([roc_auc_score(y[i, j], p[i, j]) - roc_auc_score(y[i, j], ref[i, j]) for j in range(12)])
        except ValueError:  # a resample without positives for some finding
            continue
        wins += d > 0
        n += 1
    print(f"  {name:16s} {wins / n:.2f}")
