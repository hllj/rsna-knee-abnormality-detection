"""Gold-58 AUC of every nartaa-anchored candidate (submitted EXP-077/083-086 and new EXP-092-095), reproducing the
notebook fusion: per-finding percentile ranks, weighted sum.

Inputs: goldout/raw_gold.npz (our older families), goldout_ext/ext/{n384,n224,hel}.csv, goldout_ext/raw_partners.npz.
Caveat: nartaa selected its epochs on gold-58 (and heliosli may have), so absolute values are optimistic for any
blend dominated by nartaa; compare candidates with each other and against their LB where known.
"""
import re
import sys

import numpy as np
import pandas as pd

T = "/Users/hllj/.claude/jobs/171aabfa/tmp/"
sys.argv = ["", T + "goldout/raw_gold.npz", T + "gold/train.csv"]
import os
os.chdir(T)
exec(open("gold_blend.py").read().split('print(f"{len(ids)} gold studies')[0])  # raw, ids, y, ck, rank01, auc, LABELS

idx = pd.Index(ids)


def rk(df):
    return rank01(df.loc[idx, LABELS].to_numpy(float))


def ext(tag):
    return pd.read_csv(f"goldout_ext/ext/{tag}.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")


fam = {f: sum(rank01(m) for _, m in v) / len(v) for f, v in ck.items()}         # mirror where available
plain = {f: sum(rank01(p) for p, _ in v) / len(v) for f, v in ck.items()}
N384, N224, HEL = rk(ext("n384")), rk(ext("n224")), rk(ext("hel"))

rp = np.load("goldout_ext/raw_partners.npz")
assert [str(x) for x in rp["ids"]] == ids
pl5 = [rank01(rp[k]) for k in rp.files if "dense-pl" in k and not k.endswith("#mirror")]
rd5 = [rank01((rp[k] + rp[k + "#mirror"]) / 2 if k + "#mirror" in rp.files else rp[k])
       for k in rp.files if "reader-repro" in k and not k.endswith("#mirror")]
PL5, RD5 = sum(pl5) / len(pl5), sum(rd5) / len(rd5)
print(f"members: pseudo ConvNeXt {len(pl5)} folds, reader {len(rd5)} folds")


def fuse(anchor, ours, w, hel_w=0.0, hel=None):
    """w: scalar or per-finding vector of the anchor (nartaa) share."""
    w = np.broadcast_to(np.asarray(w, float), (len(LABELS),))
    out = anchor * w + ours * (1 - w - hel_w)
    if hel_w:
        out = out + hel_w * hel
    return out


route = np.array([{"Lateral OA": .6, "PF OA": .6, "Medial OA": .6, "Fracture": .6, "Contusion": .6,
                   "Effusion": .75, "Synovitis": .75}.get(c, .85) for c in LABELS])
seeds = (fam["d96_s1"] + fam["d96_s2"]) / 2
cand = {
    "nartaa 384 alone (EXP-092 control)": (N384, None, 0.949),
    "heliosli compact alone": (HEL, None, None),
    "nartaa 224 alone": (N224, None, 0.945),
    "EXP-077 nartaa 0.8 + seeds 0.2": (fuse(N384, seeds, .8), None, None),
    "EXP-083 nartaa 0.7 + ConvNeXt dense 5f": (fuse(N384, plain["cnx_dense5"], .7), None, 0.947),
    "EXP-084 nartaa 0.7 + (ConvNeXt 5f + reader f0)": (fuse(N384, (plain["cnx_dense5"] + fam["reader_v1"]) / 2, .7), None, 0.948),
    "EXP-085 routing + ConvNeXt/reader/CoAtNet": (fuse(N384, .4 * plain["cnx_dense5"] + .4 * fam["reader_v1"] + .2 * plain["coatnet_f0"], route), None, 0.948),
    "EXP-086 EXP-085 + nartaa 224 (0.1)": (fuse(.9 * N384 + .1 * N224, .4 * plain["cnx_dense5"] + .4 * fam["reader_v1"] + .2 * plain["coatnet_f0"], route), None, 0.948),
    "EXP-093 nartaa 0.7 + heliosli 0.3": (fuse(N384, 0, .7, .3, HEL), None, None),
    "EXP-094 nartaa .65 + heliosli .25 + pseudo ConvNeXt 5f .1": (fuse(N384, PL5, .65, .25, HEL), None, None),
    "EXP-095 nartaa .65 + heliosli .25 + reader 5f .1": (fuse(N384, RD5, .65, .25, HEL), None, None),
    "pseudo ConvNeXt 5f alone": (PL5, None, None),
    "reader 5f alone": (RD5, None, None),
}
print(f'\n{"candidate":58s} {"gold-58":>8s} {"LB":>6s}')
for n, (p, _, lb) in cand.items():
    print(f"{n:58s} {auc(p):8.4f} {'' if lb is None else lb:>6}")
print("\nPer-finding gold, nartaa alone vs heliosli vs reader 5f vs pseudo ConvNeXt 5f:")
from sklearn.metrics import roc_auc_score
for j, c in enumerate(LABELS):
    a = [roc_auc_score(y[:, j], m[:, j]) for m in (N384, HEL, RD5, PL5)]
    print(f"  {c:17s} " + " ".join(f"{v:.3f}" for v in a))
