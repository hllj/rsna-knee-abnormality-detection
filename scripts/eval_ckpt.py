"""
Re-score a trained checkpoint on its held-out fold and the gold-58 at different evaluation settings
(no training) — e.g. more windows per study.

    python scripts/eval_ckpt.py --ckpt fold0.pt --k_eval 48 64 --out eval_fold0.json [config overrides...]

OOF AUC is scored against report-derived targets (same ruler as merge_oof.py), gold AUC against the
radiologist labels. Config overrides (e.g. paths.data_dir=..., paths.cache_dir=...) point at the data.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_config  # noqa: E402
from src.constants import ID_COL, LABELS  # noqa: E402
from src.dataset.folds import gold_ids  # noqa: E402
from src.evaluation.metrics import macro_auc  # noqa: E402
from src.labeling.blend import build_targets  # noqa: E402
from src.models.inference import predict_ckpt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--k_eval", type=int, nargs="+", default=[48, 64])
    ap.add_argument("--out", default="eval_ckpt.json")
    a, overrides = ap.parse_known_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    fold = int(ck["cfg"]["fold"])
    cfg = get_config(overrides)                       # data locations from this environment
    base = OmegaConf.merge(cfg, {"labels": {"pseudo": {"enabled": False}}})
    targets = build_targets(base)
    train = pd.read_csv(cfg.data.train_file, dtype={ID_COL: str})
    folds = pd.read_csv(cfg.data.folds_file, dtype={ID_COL: str})
    cached = {f[:-4] for f in os.listdir(cfg.paths.cache_dir) if f.endswith(".npz")}
    va = [u for u in folds.loc[folds.fold == fold, ID_COL] if u in cached]
    gold = train[train[ID_COL].isin(gold_ids(train))].set_index(ID_COL)[LABELS]
    gd = [u for u in gold.index if u in cached]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    report = {"ckpt": a.ckpt, "fold": fold, "n_val": len(va), "n_gold": len(gd), "results": {}}
    for k in a.k_eval:
        p_va = predict_ckpt(a.ckpt, va, cfg.paths.cache_dir, device, batch_size=2, workers=4, k_eval=k)
        p_gd = predict_ckpt(a.ckpt, gd, cfg.paths.cache_dir, device, batch_size=2, workers=4, k_eval=k)
        oof, oof_per = macro_auc(targets.reindex(va)[LABELS].values, p_va)
        gauc, g_per = macro_auc(gold.loc[gd].values, p_gd)
        report["results"][str(k)] = {"oof_auc": round(oof, 5), "gold_auc": round(gauc, 5),
                                     "oof_per_label": {x: round(v, 4) for x, v in oof_per.items()},
                                     "gold_per_label": {x: round(v, 4) for x, v in g_per.items()}}
        print(f"[eval] fold {fold} k_eval={k}: OOF {oof:.4f} | gold {gauc:.4f}", flush=True)
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
