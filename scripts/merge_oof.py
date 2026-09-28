"""
Merge per-fold predictions of one experiment and summarise its cross-validation.

    python scripts/merge_oof.py --exp effnet_b0_224 --oof_dir models/oof --train data/raw/train.csv

Writes
    <oof_dir>/<exp>.csv              OOF teacher file (one prediction per training study) for pseudo-labels
    <oof_dir>/<exp>_gold_ens.csv     gold-58 predictions averaged over all folds (= what a fold ensemble does)
    <oof_dir>/<exp>_cv.json          summary read by scripts/track.py

OOF macro AUC is scored against the report-derived targets the folds were trained on (config labels,
without pseudo-labels); gold AUC is scored against the radiologists' labels.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants import ID_COL, LABELS  # noqa: E402
from src.evaluation.metrics import macro_auc  # noqa: E402


def rank01(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(pct=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--oof_dir", default="models/oof")
    ap.add_argument("--train", default="data/raw/train.csv")
    ap.add_argument("--targets", default=None, help="csv of report targets to score OOF against (optional)")
    a = ap.parse_args()

    oof_files = sorted(glob.glob(os.path.join(a.oof_dir, f"{a.exp}_oof_fold*.csv")))
    gold_files = sorted(glob.glob(os.path.join(a.oof_dir, f"{a.exp}_gold_fold*.csv")))
    assert oof_files or gold_files, f"no prediction files for {a.exp} in {a.oof_dir}"
    summary = {"exp_name": a.exp, "folds": len(oof_files)}

    if oof_files:
        oof = pd.concat([pd.read_csv(f, dtype={ID_COL: str}) for f in oof_files]).drop_duplicates(ID_COL)
        oof.to_csv(os.path.join(a.oof_dir, f"{a.exp}.csv"), index=False)
        summary["oof_studies"] = len(oof)
        if a.targets:
            t = pd.read_csv(a.targets, dtype={ID_COL: str}).set_index(ID_COL).reindex(oof[ID_COL])
            m, per = macro_auc(t[LABELS].values, oof[LABELS].values)
            summary.update(oof_auc=round(m, 5), oof_per_label={k: round(v, 4) for k, v in per.items()})

    if gold_files:
        train = pd.read_csv(a.train, dtype={ID_COL: str})
        gold = train.dropna(subset=LABELS).set_index(ID_COL)[LABELS]
        preds = [pd.read_csv(f, dtype={ID_COL: str}).set_index(ID_COL).reindex(gold.index)[LABELS] for f in gold_files]
        singles = [macro_auc(gold.values, p.values)[0] for p in preds]
        ens = sum(rank01(p) for p in preds) / len(preds)
        ens.reset_index().to_csv(os.path.join(a.oof_dir, f"{a.exp}_gold_ens.csv"), index=False)
        m, per = macro_auc(gold.values, ens.values)
        summary.update(gold_auc_per_fold=[round(x, 4) for x in singles], gold_auc_mean_fold=round(float(np.mean(singles)), 5),
                       gold_auc_ensemble=round(m, 5), gold_per_label={k: round(v, 4) for k, v in per.items()})

    with open(os.path.join(a.oof_dir, f"{a.exp}_cv.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if "per_label" not in k}, indent=1))


if __name__ == "__main__":
    main()
