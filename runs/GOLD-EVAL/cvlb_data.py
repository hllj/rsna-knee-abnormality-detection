"""Assemble every LB-scored experiment with its OOF CV and gold-58 AUC; compute correlations; write cvlb.json."""
import json
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

T = "/Users/hllj/.claude/jobs/171aabfa/tmp/"
REPO = "/Users/hllj/Projects/rsna-knee-abnormality-detection/.claude/worktrees/dense-f0/"
sys.argv = ["", T + "goldout/raw_gold.npz", T + "gold/train.csv"]
import os
os.chdir(T)
exec(open("gold_blend.py").read().split('print(f"{len(ids)} gold studies')[0])  # raw, y, rank01, auc


def ck(pat):  # plain prediction of one checkpoint by path pattern
    ks = [k for k in raw.files if re.search(pat, k) and not k.endswith("#mirror")]
    assert len(ks) == 1, (pat, ks)
    return rank01(raw[ks[0]])


cnx5 = sum(ck(p) for p in [r"exp-024-fold0", r"exp-026-fold1", r"exp-026-fold2", r"exp-026-fold3", r"exp-026-fold4"]) / 5
rap0, rea0, coat0, rapall = ck(r"exp-041-fold0"), ck(r"exp-042-fold0"), ck(r"exp-028-fold0"), ck(r"exp-054")
rap5 = sum(ck(p) for p in [r"exp-041-fold0", r"exp-051-fold1", r"exp-051-fold2", r"exp-051-fold3", r"exp-051-fold4"]) / 5
cnx0 = ck(r"exp-024-fold0")
computed_gold = {
    "EXP-047": auc((cnx0 + rap0 + rea0) / 3),
    "EXP-048": auc((cnx5 + rap0 + rea0) / 3),
    "EXP-049": auc(0.3 * cnx5 + 0.3 * rap0 + 0.3 * rea0 + 0.1 * coat0),
    "EXP-059": auc(0.3 * cnx5 + 0.15 * rap5 + 0.15 * rapall + 0.3 * rea0 + 0.1 * coat0),
}
# consistent gold-58 for blends from the gold-eval re-scoring (same rank pipeline, mirror where the submission had it)
gold_eval = {"EXP-055": .9202, "EXP-070": .9203, "EXP-060": .9133, "EXP-058": .9080, "EXP-067": .9153,
             "EXP-068": .9147, "EXP-071": .9199, "EXP-074": .9206, "EXP-075": .9226, "EXP-076": .9259}

r = pd.read_csv(REPO + "experiments/registry.csv", dtype=str)
for c in ["oof_auc", "gold_auc_mean_fold", "gold_auc_ensemble", "lb_public"]:
    r[c] = pd.to_numeric(r[c], errors="coerce")
r = r[r.lb_public.notna()].set_index("exp_id")

META = {  # exp: (short label, group, fold of the OOF if single-fold, OOF override)
    "EXP-000": ("EffNet-B0 dry run", "excluded", None, None),
    "EXP-007": ("EffNet-B0 f0", "single fold", 0, None),
    "EXP-008": ("EffNet-B0 f1", "single fold", 1, None),
    "EXP-009": ("EffNet-B0 5f", "5-fold", None, None),
    "EXP-012": ("EffNet-B0 5f (fast infer)", "excluded", None, None),
    "EXP-017": ("ConvNeXt 288 f0 (stale weights)", "excluded", None, None),
    "EXP-019": ("ConvNeXt 288 f0", "single fold", 0, None),
    "EXP-021": ("ConvNeXt 288 f1", "single fold", 1, None),
    "EXP-020": ("ConvNeXt 288 5f", "5-fold", None, None),
    "EXP-022": ("ConvNeXt 5f + EffNet 5f", "blend", None, None),
    "EXP-025": ("ConvNeXt dense f0", "single fold", 0, None),
    "EXP-029": ("ConvNeXt dense f1", "single fold", 1, None),
    "EXP-027": ("ConvNeXt dense 5f", "5-fold", None, None),
    "EXP-040": ("CoAtNet-2 dense f0", "single fold", 0, None),
    "EXP-045": ("Raptor repro f0", "single fold", 0, None),
    "EXP-046": ("Reader v1 f0", "single fold", 0, None),
    "EXP-047": ("3-way f0 blend", "blend", 0, None),
    "EXP-048": ("ConvNeXt 5f + Raptor f0 + reader", "blend", None, None),
    "EXP-049": ("EXP-048 + CoAtNet f0", "blend", None, None),
    "EXP-055": ("ConvNeXt 5f + Raptor 5f + reader + CoAtNet", "blend", None, None),
    "EXP-056": ("Reader v2 f0", "single fold", 0, None),
    "EXP-060": ("ConvNeXt 5f + Raptor 5f", "blend", None, None),
    "EXP-058": ("Raptor v1 all-data", "all-data", None, None),
    "EXP-059": ("EXP-055 + Raptor all-data", "blend", None, None),
    "EXP-069": ("OAI: nartaa 0.65 + EXP-055", "excluded", None, None),
    "EXP-067": ("Raptor v2 80-slice all-data", "all-data", None, None),
    "EXP-070": ("EXP-055 + reader v3", "blend", None, None),
    "EXP-068": ("Raptor v2 96-slice", "all-data", None, None),
    "EXP-071": ("EXP-055 ×0.6 + 96-slice ×0.4", "blend", None, None),
    "EXP-074": ("96-slice seeds 1+2", "all-data", None, None),
    "EXP-075": ("EXP-055 ×0.6 + 2 seeds ×0.4", "blend", None, None),
    "EXP-076": ("Gold-tuned blend", "blend", None, None),
    "EXP-078": ("Pseudo-label ConvNeXt f0", "single fold", 0, 0.8825),
}
GOLD_OVERRIDE = {"EXP-078": 0.9044}
rows = []
for e, (lab, grp, fold, oof_ovr) in META.items():
    x = r.loc[e]
    gold = gold_eval.get(e, computed_gold.get(e, GOLD_OVERRIDE.get(e)))
    if gold is None:
        g = x.gold_auc_ensemble if pd.notna(x.gold_auc_ensemble) else x.gold_auc_mean_fold
        gold = None if pd.isna(g) else float(g)
    oof = oof_ovr if oof_ovr is not None else (None if pd.isna(x.oof_auc) else float(x.oof_auc))
    if e in ("EXP-047",):
        oof = 0.8828
    rows.append(dict(exp=e, label=lab, group=grp, fold=fold, lb=float(x.lb_public),
                     gold=None if gold is None else round(float(gold), 4), oof=oof))
df = pd.DataFrame(rows)


def corr(d, a, b):
    d = d.dropna(subset=[a, b])
    if len(d) < 3:
        return None
    p, s = pearsonr(d[a], d[b])[0], spearmanr(d[a], d[b])[0]
    k, c = np.polyfit(d[a], d[b], 1)
    resid = d[b] - (k * d[a] + c)
    return dict(n=len(d), pearson=round(p, 3), spearman=round(s, 3), slope=round(k, 3), icpt=round(c, 4),
                rmse=round(float(np.sqrt((resid ** 2).mean())), 4))


use = df[df.group != "excluded"]
comp = use[use.lb >= 0.925]
f0 = use[(use.fold == 0) & use.oof.notna()]
stats = {
    "gold_lb_all": corr(use, "gold", "lb"),
    "gold_lb_competitive": corr(comp, "gold", "lb"),
    "oof_lb_fold0": corr(f0, "oof", "lb"),
    "oof_lb_all_with_oof": corr(use, "oof", "lb"),
    "gold_oof": corr(use, "gold", "oof"),
    "lb_minus_gold_mean": round(float((use.lb - use.gold).mean()), 4),
    "lb_minus_gold_sd": round(float((use.lb - use.gold).std()), 4),
}
fit = np.polyfit(comp.dropna(subset=["gold"]).gold, comp.dropna(subset=["gold"]).lb, 1)
df["pred_from_gold"] = df.gold.apply(lambda g: None if g is None or pd.isna(g) else round(float(np.polyval(fit, g)), 4))
df["resid"] = df.apply(lambda x: None if x.pred_from_gold is None or pd.isna(x.pred_from_gold) else round(x.lb - x.pred_from_gold, 4), axis=1)
json.dump({"rows": df.where(df.notna(), None).to_dict("records"), "stats": stats,
           "fit_competitive": [round(float(fit[0]), 4), round(float(fit[1]), 4)],
           "computed_gold": {k: round(v, 4) for k, v in computed_gold.items()}},
          open(T + "cvlb.json", "w"), indent=1)
print(json.dumps(stats, indent=1))
print({k: round(v, 4) for k, v in computed_gold.items()})
print(df[["exp", "label", "group", "oof", "gold", "lb", "resid"]].to_string(index=False))
