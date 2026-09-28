"""
Test-time inference: DICOM -> stack (same code as training cache) -> ensemble -> submission.csv.

Runs offline inside the Kaggle kernel (internet disabled, <= 9 h; efficiency prize rewards speed).

    python -m src.models.inference --data_dir /kaggle/input/rsna-knee-abnormality-detection \
        --ckpts /kaggle/input/my-weights/*.pt --out submission.csv
"""
import argparse
import glob
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from typing import List

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.constants import ID_COL, LABELS
from src.dataset.datasets import KneeStudyDataset
from src.dataset.dicom import build_study_stack, slot_specs_from_cfg
from src.models.model import build_model


def _cache_one(args):
    uid, study_dir, series, pp, cache_dir = args
    out = os.path.join(cache_dir, f"{uid}.npz")
    if not os.path.exists(out):
        d = build_study_stack(
            study_dir, series, slot_specs_from_cfg(pp["slots"]), res=pp["res"],
            crop_mm=pp["crop_mm"], span=pp["span"], pcts=pp["clip_percentiles"],
        )
        np.savez(out, **d)
    return uid


def build_cache(study_ids: List[str], dicom_dir: str, series_df: pd.DataFrame, pp: dict,
                cache_dir: str, workers: int = 4) -> None:
    """Preprocess studies into cache_dir/<uid>.npz in parallel (CPU-bound DICOM decode)."""
    os.makedirs(cache_dir, exist_ok=True)
    groups = {u: g for u, g in series_df.astype({ID_COL: str}).groupby(ID_COL)}
    jobs = [(u, os.path.join(dicom_dir, u), groups.get(u, series_df.iloc[:0]), pp, cache_dir) for u in study_ids]
    with ProcessPoolExecutor(workers) as ex:
        for _ in ex.map(_cache_one, jobs, chunksize=4):
            pass


@torch.no_grad()
def predict_ckpt(ckpt_path: str, study_ids: List[str], cache_dir: str, device, batch_size: int = 2,
                 workers: int = 4, k_eval: int = None) -> np.ndarray:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ck["cfg"])
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.eval().to(device)
    if device.type == "cuda":
        model.half()
    dc = model.data_config
    ds = KneeStudyDataset(study_ids, cache_dir, k=k_eval or cfg.windows.k_eval, img_size=cfg.model.img_size,
                          mean=dc["mean"], std=dc["std"])
    dl = DataLoader(ds, batch_size=batch_size, num_workers=workers, pin_memory=True)
    out = []
    for b in dl:
        x = b["x"].to(device, non_blocking=True)
        x = x.half() if device.type == "cuda" else x
        logits = model(x, b["slot"].to(device), b["mask"].to(device))
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)


def rank01(p: np.ndarray) -> np.ndarray:
    """Column-wise rank to [0, 1] (AUC only cares about order; makes models comparable)."""
    return (p.argsort(0).argsort(0) + 0.5) / len(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/kaggle/input/rsna-knee-abnormality-detection")
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--rank_average", action="store_true")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--out", default="submission.csv")
    a = ap.parse_args()

    ckpts = sorted(p for pat in a.ckpts for p in glob.glob(pat))
    test = pd.read_csv(os.path.join(a.data_dir, "test.csv"), dtype={ID_COL: str})
    series = pd.read_csv(os.path.join(a.data_dir, "test_series.csv"), dtype={ID_COL: str, "SeriesInstanceUID": str})
    ids = test[ID_COL].tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # One cache per distinct preprocessing contract (ckpts trained at different res/crop).
    by_pp = {}
    for c in ckpts:
        pp = torch.load(c, map_location="cpu", weights_only=False)["cfg"]["preprocess"]
        by_pp.setdefault(repr(sorted(pp.items())), (pp, []))[1].append(c)

    root = a.cache_dir or tempfile.mkdtemp(prefix="knee_cache_")
    preds = []
    for i, (pp, group) in enumerate(by_pp.values()):
        cache_dir = os.path.join(root, f"pp{i}")
        build_cache(ids, os.path.join(a.data_dir, "test_series"), series, pp, cache_dir, a.workers)
        for c in group:
            p = predict_ckpt(c, ids, cache_dir, device, workers=min(4, a.workers))
            preds.append(rank01(p) if a.rank_average else p)
            print(f"[infer] {os.path.basename(c)} done")

    w = np.asarray(a.weights or [1.0] * len(preds), dtype=float)
    final = sum(wi * p for wi, p in zip(w, preds)) / w.sum()
    sub = pd.DataFrame(final, columns=LABELS)
    sub.insert(0, ID_COL, ids)
    sub.to_csv(a.out, index=False)
    print(f"[infer] wrote {a.out} {sub.shape}")


if __name__ == "__main__":
    main()
