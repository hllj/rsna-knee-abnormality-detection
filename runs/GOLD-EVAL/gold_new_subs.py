"""Gold-58 AUC of the submissions not yet scored on gold: EXP-069, 077, 082, 083-086 (same rank pipeline)."""
import json
import os
import sys

import numpy as np
import pandas as pd

T = "/Users/hllj/.claude/jobs/171aabfa/tmp/"
os.chdir(T)
sys.argv = ["", T + "goldout/raw_gold.npz", T + "gold/train.csv"]
exec(open("gold_blend.py").read().split('print(f"{len(ids)} gold studies')[0])  # raw, ids, y, ck, rank01, auc, LABELS

idx = pd.Index(ids)
fam = {f: sum(rank01(m) for _, m in v) / len(v) for f, v in ck.items()}
plain = {f: sum(rank01(p) for p, _ in v) / len(v) for f, v in ck.items()}


def rk_csv(p):
    return rank01(pd.read_csv(p, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[idx, LABELS].to_numpy(float))


N384, N224 = rk_csv("goldout_ext/ext/n384.csv"), rk_csv("goldout_ext/ext/n224.csv")
V3 = rk_csv("/Users/hllj/Projects/rsna-knee-abnormality-detection/.claude/worktrees/dense-f0/runs/ITER-07/iter11/models/oof/"
            "raptor_v3_d96_alldata_pl_gold_fold-1.csv")
route = np.array([{"Lateral OA": .6, "PF OA": .6, "Medial OA": .6, "Fracture": .6, "Contusion": .6,
                   "Effusion": .75, "Synovitis": .75}.get(c, .85) for c in LABELS])
exp055 = .3 * plain["cnx_dense5"] + .3 * plain["raptor_repro5"] + .3 * plain["reader_v1"] + .1 * plain["coatnet_f0"]
seeds = (fam["d96_s1"] + fam["d96_s2"]) / 2
part85 = .4 * plain["cnx_dense5"] + .4 * fam["reader_v1"] + .2 * plain["coatnet_f0"]
g = {
    "EXP-069": auc(.65 * N384 + .35 * rank01(exp055)),
    "EXP-077": auc(.8 * N384 + .2 * seeds),
    "EXP-082": auc((fam["d96_s1"] + fam["d96_s2"] + V3) / 3),
    "EXP-083": auc(.7 * N384 + .3 * plain["cnx_dense5"]),
    "EXP-084": auc(.7 * N384 + .3 * (plain["cnx_dense5"] + fam["reader_v1"]) / 2),
    "EXP-085": auc(N384 * route + part85 * (1 - route)),
    "EXP-086": auc((.9 * N384 + .1 * N224) * route + part85 * (1 - route)),
    "nartaa 384 alone": auc(N384),
    "Raptor v3 alone (no mirror)": auc(V3),
}
for k, v in g.items():
    print(f"{k:28s} {v:.4f}")
json.dump({k: round(v, 4) for k, v in g.items()}, open(T + "gold_new_subs.json", "w"), indent=1)
