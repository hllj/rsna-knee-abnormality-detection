"""
Test-time inference: DICOM -> stack (same code as training cache) -> ensemble -> submission.csv.

Runs offline inside the Kaggle kernel (internet disabled, <= 9 h; efficiency prize rewards speed).

    python -m src.models.inference --data_dir /kaggle/input/rsna-knee-abnormality-detection \
        --ckpts /kaggle/input/my-weights/*.pt --out submission.csv

Two modes:
  * stream (default) — a process pool builds each study's stack in memory (selective DICOM decode,
    threaded reads) while the main process batches finished studies onto the GPU and runs every
    checkpoint on the same windows: preprocessing and GPU overlap, and the data is read once.
  * cache — the original path: write every study to <cache_dir>/<uid>.npz, then one DataLoader pass
    per checkpoint. Kept for comparison (scripts/bench_inference.py) and debugging.
"""
import argparse
import glob
import json
import os
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.constants import ID_COL, LABELS
from src.dataset.datasets import KneeStudyDataset, eval_windows
from src.dataset.dicom import build_study_stack, slot_specs_from_cfg
from src.kaggle_paths import find_competition_root
from src.models.model import build_model

# preprocessing fields that change the stack; others (num_workers, limit, compress) do not
PP_KEYS = ("slots", "res", "crop_mm", "span", "clip_percentiles", "canonical")


def pp_key(pp: dict) -> str:
    return json.dumps({k: pp.get(k) for k in PP_KEYS}, sort_keys=True, default=str)


def _stack(uid, study_dir, series, pp, opts) -> Dict[str, np.ndarray]:
    return build_study_stack(
        study_dir, series, slot_specs_from_cfg(pp["slots"]), res=pp["res"], crop_mm=pp["crop_mm"],
        span=pp["span"], pcts=pp["clip_percentiles"], canonical=pp.get("canonical", True),
        selective=opts.get("selective", False), pct_stride=opts.get("pct_stride", 0), threads=opts.get("threads", 8),
    )


# ------------------------------------------------------------------------------------------------ cache mode
def _cache_one(args):
    uid, study_dir, series, pp, cache_dir = args[:5]
    opts = args[5] if len(args) > 5 else {}
    out = os.path.join(cache_dir, f"{uid}.npz")
    if not os.path.exists(out):
        (np.savez_compressed if pp.get("compress") else np.savez)(out, **_stack(uid, study_dir, series, pp, opts))
    return uid


def _jobs(study_ids, dicom_dir, series_df, pp, *extra):
    groups = {u: g for u, g in series_df.astype({ID_COL: str}).groupby(ID_COL)}
    return [(u, os.path.join(dicom_dir, u), groups.get(u, series_df.iloc[:0]), pp, *extra) for u in study_ids]


def build_cache(study_ids: List[str], dicom_dir: str, series_df: pd.DataFrame, pp: dict,
                cache_dir: str, workers: int = 4, opts: Optional[dict] = None) -> None:
    """Preprocess studies into cache_dir/<uid>.npz in parallel (CPU-bound DICOM decode)."""
    os.makedirs(cache_dir, exist_ok=True)
    with ProcessPoolExecutor(workers) as ex:
        for _ in ex.map(_cache_one, _jobs(study_ids, dicom_dir, series_df, pp, cache_dir, opts or {}), chunksize=4):
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


# ----------------------------------------------------------------------------------------------- stream mode
def _stack_job(args):
    uid, study_dir, series, pp, opts = args
    t = time.time()
    try:
        d = _stack(uid, study_dir, series, pp, opts)
    except Exception:  # never lose a study: an empty (masked) stack predicts the model's prior
        n = sum(int(v["n"]) for v in pp["slots"].values())
        d = {"image": np.zeros((n, pp["res"], pp["res"]), np.uint8), "slot": np.zeros(n, np.int8),
             "valid": np.zeros(n, np.uint8), "side": np.int8(0)}
    return uid, d, time.time() - t


def stream_stacks(study_ids, dicom_dir, series_df, pp, workers: int, opts: dict,
                  max_inflight: int = 32) -> Iterator[Tuple[str, dict, float]]:
    """Yield (uid, stack, seconds) as soon as each study is built; at most `max_inflight` in memory."""
    jobs = iter(_jobs(study_ids, dicom_dir, series_df, pp, opts))
    with ProcessPoolExecutor(workers) as ex:
        pending = set()
        for job in jobs:
            pending.add(ex.submit(_stack_job, job))
            if len(pending) >= max_inflight:
                break
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in done:
                yield f.result()
                job = next(jobs, None)
                if job is not None:
                    pending.add(ex.submit(_stack_job, job))


class Member:
    """One checkpoint on the device plus its input contract."""

    def __init__(self, path: str, device):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.path, self.cfg = path, OmegaConf.create(ck["cfg"])
        self.model = build_model(self.cfg, pretrained=False)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval().to(device)
        if device.type == "cuda":
            self.model.half()
        dc = self.model.data_config
        self.k, self.img = int(self.cfg.windows.k_eval), int(self.cfg.model.img_size)
        self.mean, self.std = tuple(dc["mean"]), tuple(dc["std"])
        self.pp = OmegaConf.to_container(self.cfg.preprocess, resolve=True)

    @property
    def contract(self):
        return self.k, self.img, self.mean, self.std


@torch.no_grad()
def predict_batch(groups: Dict[tuple, List[Member]], stacks: List[dict], device) -> Dict[str, np.ndarray]:
    """Run every member on one batch of studies; windows are built once per input contract."""
    out = {}
    for (k, img, mean, std), members in groups.items():
        wins, slots = zip(*(eval_windows(d, k) for d in stacks))
        x = torch.from_numpy(np.stack(wins)).to(device, non_blocking=True)  # B,K,3,H,W uint8
        b = x.shape[0]
        x = x.flatten(0, 1).float().div_(255.0)
        if x.shape[-1] != img:
            x = F.interpolate(x, size=(img, img), mode="bilinear", align_corners=False)
        m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        s = torch.tensor(std, device=device).view(1, 3, 1, 1)
        x = ((x - m) / s).view(b, k, 3, img, img)
        if device.type == "cuda":
            x = x.half()
        slot = torch.from_numpy(np.stack(slots)).to(device)
        mask = torch.ones(b, k, dtype=torch.bool, device=device)
        for mem in members:
            out[mem.path] = torch.sigmoid(mem.model(x, slot, mask).float()).cpu().numpy()
    return out


def run_stream(ids: List[str], dicom_dir: str, series: pd.DataFrame, ckpts: List[str], device, workers: int,
               opts: dict, batch: int = 8, log_every: int = 100) -> Dict[str, np.ndarray]:
    """All checkpoints' predictions (ckpt -> (N, 12), rows in `ids` order)."""
    members = [Member(c, device) for c in ckpts]
    by_pp: Dict[str, List[Member]] = {}
    for m in members:
        by_pp.setdefault(pp_key(m.pp), []).append(m)
    row = {u: i for i, u in enumerate(ids)}
    preds = {m.path: np.zeros((len(ids), len(LABELS)), np.float32) for m in members}
    t0, stack_s, gpu_s, n = time.time(), 0.0, 0.0, 0
    for mems in by_pp.values():
        groups: Dict[tuple, List[Member]] = {}
        for m in mems:
            groups.setdefault(m.contract, []).append(m)
        buf: List[Tuple[str, dict]] = []

        def flush():
            nonlocal gpu_s
            tg = time.time()
            res = predict_batch(groups, [d for _, d in buf], device)
            for path, p in res.items():
                for j, (u, _) in enumerate(buf):
                    preds[path][row[u]] = p[j]
            gpu_s += time.time() - tg
            buf.clear()

        for uid, d, sec in stream_stacks(ids, dicom_dir, series, mems[0].pp, workers, opts):
            buf.append((uid, d))
            stack_s += sec
            n += 1
            if len(buf) >= batch:
                flush()
            if n % log_every == 0:
                print(f"[stream] {n}/{len(ids) * len(by_pp)} studies | {time.time() - t0:.0f}s wall | "
                      f"{stack_s / n:.2f} s/study CPU | GPU {gpu_s:.0f}s", flush=True)
        if buf:
            flush()
    print(f"[stream] done: {n} studies, {len(members)} models, {time.time() - t0:.0f}s wall, "
          f"{stack_s / max(n, 1):.2f} s/study CPU (x{workers} workers), GPU {gpu_s:.0f}s", flush=True)
    return preds


def rank01(p: np.ndarray) -> np.ndarray:
    """Column-wise rank to [0, 1] (AUC only cares about order; makes models comparable)."""
    return (p.argsort(0).argsort(0) + 0.5) / len(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None, help="competition dir (auto-detected on Kaggle)")
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--rank_average", action="store_true")
    ap.add_argument("--mode", choices=["stream", "cache"], default="stream")
    ap.add_argument("--selective", type=int, default=1, help="decode only sampled slices (1) or every slice (0)")
    ap.add_argument("--pct_stride", type=int, default=0, help="extra slices for the intensity window (0 = none)")
    ap.add_argument("--threads", type=int, default=8, help="DICOM read threads per worker process")
    ap.add_argument("--batch", type=int, default=8, help="studies per GPU batch (stream mode)")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--out", default="submission.csv")
    a = ap.parse_args()

    ckpts = sorted({p for pat in a.ckpts for p in glob.glob(pat, recursive=True)})
    assert ckpts, f"no checkpoints match {a.ckpts}"
    a.data_dir = a.data_dir or find_competition_root()
    assert a.data_dir, "competition data not found; pass --data_dir"
    test = pd.read_csv(os.path.join(a.data_dir, "test.csv"), dtype={ID_COL: str})
    series = pd.read_csv(os.path.join(a.data_dir, "test_series.csv"), dtype={ID_COL: str, "SeriesInstanceUID": str})
    ids = test[ID_COL].tolist()
    dicom_dir = os.path.join(a.data_dir, "test_series")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opts = {"selective": bool(a.selective), "pct_stride": a.pct_stride, "threads": a.threads}

    if a.mode == "stream":
        by_ckpt = run_stream(ids, dicom_dir, series, ckpts, device, a.workers, opts, batch=a.batch)
        preds = [by_ckpt[c] for c in ckpts]
    else:
        by_pp = {}
        for c in ckpts:
            pp = torch.load(c, map_location="cpu", weights_only=False)["cfg"]["preprocess"]
            by_pp.setdefault(pp_key(pp), (pp, []))[1].append(c)
        root = a.cache_dir or tempfile.mkdtemp(prefix="knee_cache_")
        preds = []
        for i, (pp, group) in enumerate(by_pp.values()):
            cache_dir = os.path.join(root, f"pp{i}")
            build_cache(ids, dicom_dir, series, pp, cache_dir, a.workers, opts)
            preds += [predict_ckpt(c, ids, cache_dir, device, workers=min(4, a.workers)) for c in group]
    for c in ckpts:
        print(f"[infer] {os.path.basename(c)} done")

    preds = [rank01(p) if a.rank_average else p for p in preds]
    w = np.asarray(a.weights or [1.0] * len(preds), dtype=float)
    final = sum(wi * p for wi, p in zip(w, preds)) / w.sum()
    sub = pd.DataFrame(final, columns=LABELS)
    sub.insert(0, ID_COL, ids)
    sub.to_csv(a.out, index=False)
    print(f"[infer] wrote {a.out} {sub.shape}")


if __name__ == "__main__":
    main()
