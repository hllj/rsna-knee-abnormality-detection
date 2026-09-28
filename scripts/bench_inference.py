"""
Benchmark inference variants on training DICOMs (Kaggle GPU notebook: rsna-knee-bench).

  A  cache   full decode        — the original path (build npz cache, one DataLoader pass per checkpoint)
  B  stream  full decode        — ideas 2-4: threaded/raw reads, CPU/GPU overlap, one pass for all checkpoints
  C  stream  selective, pct 0   — + idea 1: decode only the sampled slices, window from those slices
  D  stream  selective, pct 4   — idea 1 with every 4th slice added to the intensity window

Timing: every variant gets its own, disjoint set of `--n_time` non-gold studies, so none of them reads
DICOMs already in the OS file cache. Agreement: all variants then predict the same 58 gold studies and are
compared with A (per-model |Δp|, correlation, gold-58 AUC of the rank-averaged ensemble).

    python scripts/bench_inference.py --ckpts /kaggle/input/rsna-knee-weights/**/*.pt --n_time 150
"""
import argparse
import glob
import json
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants import ID_COL, LABELS  # noqa: E402
from src.dataset.folds import gold_ids  # noqa: E402
from src.evaluation.metrics import macro_auc  # noqa: E402
from src.kaggle_paths import find_competition_root  # noqa: E402
from src.models.inference import build_cache, pp_key, predict_ckpt, rank01, run_stream  # noqa: E402

VARIANTS = {
    "A_cache_full": ("cache", {"selective": False, "pct_stride": 0}),
    "B_stream_full": ("stream", {"selective": False, "pct_stride": 0}),
    "C_stream_selective": ("stream", {"selective": True, "pct_stride": 0}),
    "D_stream_selective_pct4": ("stream", {"selective": True, "pct_stride": 4}),
}


def run(variant, ids, dicom_dir, series, ckpts, device, workers, threads):
    mode, opts = VARIANTS[variant]
    opts = dict(opts, threads=threads)
    t = time.time()
    if mode == "stream":
        by = run_stream(ids, dicom_dir, series, ckpts, device, workers, opts, batch=8)
        preds = [by[c] for c in ckpts]
    else:
        pp = torch.load(ckpts[0], map_location="cpu", weights_only=False)["cfg"]["preprocess"]
        assert all(pp_key(torch.load(c, map_location="cpu", weights_only=False)["cfg"]["preprocess"]) == pp_key(pp)
                   for c in ckpts)
        cache = tempfile.mkdtemp(prefix=f"bench_{variant}_")
        build_cache(ids, dicom_dir, series, pp, cache, workers, opts)
        preds = [predict_ckpt(c, ids, cache, device, workers=min(4, workers)) for c in ckpts]
    return preds, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--n_time", type=int, default=150, help="fresh studies per variant for timing")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--test_studies", type=int, default=1300, help="for the extrapolated runtime")
    ap.add_argument("--out", default="bench_inference.json")
    a = ap.parse_args()

    ckpts = sorted({p for pat in a.ckpts for p in glob.glob(pat, recursive=True)})
    data = a.data_dir or find_competition_root()
    train = pd.read_csv(os.path.join(data, "train.csv"), dtype={ID_COL: str})
    series = pd.read_csv(os.path.join(data, "train_series.csv"), dtype={ID_COL: str, "SeriesInstanceUID": str})
    dicom_dir = os.path.join(data, "train_series")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gold = train[train[ID_COL].isin(gold_ids(train))].set_index(ID_COL)[LABELS]
    others = train.loc[~train[ID_COL].isin(gold.index), ID_COL].sample(frac=1.0, random_state=0).tolist()
    print(f"{len(ckpts)} checkpoints | device {device} | CPUs {os.cpu_count()} | variants {a.variants}")

    report = {"n_time": a.n_time, "ckpts": [os.path.basename(c) for c in ckpts], "variants": {}}
    for i, v in enumerate(a.variants):   # timing on disjoint, never-read studies
        ids = others[i * a.n_time:(i + 1) * a.n_time]
        _, sec = run(v, ids, dicom_dir, series, ckpts, device, a.workers, a.threads)
        per = sec / len(ids)
        report["variants"][v] = {"seconds": round(sec, 1), "s_per_study": round(per, 3),
                                 "est_test_min": round(per * a.test_studies / 60, 1)}
        print(f"[time] {v:26s} {sec:7.1f}s for {len(ids)} studies = {per:.3f} s/study "
              f"-> ~{per * a.test_studies / 60:.1f} min for {a.test_studies} test studies", flush=True)

    ref = None   # agreement on the gold studies, relative to the first variant
    for v in a.variants:
        preds, _ = run(v, list(gold.index), dicom_dir, series, ckpts, device, a.workers, a.threads)
        ens = sum(rank01(p) for p in preds) / len(preds)
        auc, _ = macro_auc(gold.values, ens)
        r = report["variants"][v]
        r["gold_auc_ensemble"] = round(auc, 5)
        r["gold_auc_per_model"] = [round(macro_auc(gold.values, p)[0], 4) for p in preds]
        if ref is None:
            ref = preds
        else:
            r["max_abs_diff_vs_ref"] = round(float(max(np.abs(p - q).max() for p, q in zip(preds, ref))), 5)
            r["mean_abs_diff_vs_ref"] = round(float(np.mean([np.abs(p - q).mean() for p, q in zip(preds, ref)])), 5)
            r["corr_vs_ref"] = round(float(np.mean([np.corrcoef(p.ravel(), q.ravel())[0, 1] for p, q in zip(preds, ref)])), 5)
        print(f"[agree] {v:26s} gold ens AUC {auc:.4f} " + " ".join(f"{k}={r[k]}" for k in
              ("max_abs_diff_vs_ref", "mean_abs_diff_vs_ref", "corr_vs_ref") if k in r), flush=True)

    json.dump(report, open(a.out, "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
