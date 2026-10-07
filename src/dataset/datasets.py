"""
Study-level dataset: each item is a bag of K 2.5D windows (3 physically adjacent slices stacked as RGB) plus the
slot id, a validity mask and the slice position (0..1 inside its series) of each window.

Two cache layouts (`preprocess.layout`):
  * study  — fixed slots per study (ours, Raptor). Windows stay inside one slot unless `windows.cross_slot`
             (Raptor runs windows over the concatenated stack and masks those centred on an empty slot).
  * series — every series with its native slices (goodpjw's reader). Per slot: one candidate series (random in
             training when `windows.random_series`, else the largest), k centres in equal bins over `windows.span`.

`make_windows` + `to_model_input` are shared by training, validation and stream inference, so the model sees
identical inputs everywhere.
"""
import math
import os
import random
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.dataset.augmentations import StudyAugment

N_SERIES_SLOTS = 6  # series layout: plane x fat-suppression


# ------------------------------------------------------------------------------------------- study layout
def window_centers(slot: np.ndarray, valid: np.ndarray, cross_slot: bool = False) -> np.ndarray:
    """Indices c usable as window centres: c-1, c, c+1 valid and in the same slot (or every position)."""
    if cross_slot:
        return np.arange(1, len(slot) - 1)
    c = np.arange(1, len(slot) - 1)
    ok = (
        (slot[c - 1] == slot[c]) & (slot[c + 1] == slot[c])
        & (valid[c - 1] > 0) & (valid[c] > 0) & (valid[c + 1] > 0)
    )
    return c[ok]


def select_centers(centers: np.ndarray, k: int, train: bool, sampling: str = "uniform") -> np.ndarray:
    """k centres: random (train; `stratified` = one per equal bin) or evenly spaced (eval); k <= 0 = all."""
    if len(centers) == 0:
        return np.ones(max(k, 1), dtype=int)
    if k <= 0:
        return centers
    if train:
        if sampling == "stratified":
            edges = np.linspace(0, len(centers), k + 1)
            idx = np.floor(edges[:-1] + np.random.rand(k) * (edges[1:] - edges[:-1])).astype(int)
            return centers[np.clip(idx, 0, len(centers) - 1)]
        return np.random.choice(centers, k, replace=len(centers) < k)
    return centers[np.linspace(0, len(centers) - 1, k).round().astype(int)]


def slot_positions(slot: np.ndarray) -> np.ndarray:
    """Relative position (0..1) of every slice inside its slot."""
    pos = np.zeros(len(slot), np.float32)
    for s in np.unique(slot):
        idx = np.where(slot == s)[0]
        pos[idx] = np.arange(len(idx)) / max(len(idx) - 1, 1)
    return pos


def study_windows(d: Dict[str, np.ndarray], k: int, train: bool, w: Optional[dict] = None):
    w = w or {}
    slot, valid = d["slot"], d["valid"]
    cross = bool(w.get("cross_slot", False))
    cs = select_centers(window_centers(slot, valid, cross), k, train, w.get("sampling", "uniform"))
    wins = np.stack([d["image"][c - 1: c + 2] for c in cs])
    mask = (valid[cs] > 0) if cross else np.ones(len(cs), bool)
    return wins, slot[cs].astype(np.int64), mask, slot_positions(slot)[cs]


# ------------------------------------------------------------------------------------------ series layout
def reader_centres(n: int, k: int, train: bool, lo: float = 0.04, hi: float = 0.96) -> np.ndarray:
    """k window centres spread over [lo, hi] of an n-slice series (one random centre per bin in training)."""
    a, b = max(1.0, lo * (n - 1)), min(n - 2.0, hi * (n - 1))
    if b < a:
        a = b = (n - 1) / 2
    if train:
        edges = np.linspace(a, b, k + 1)
        c = edges[:-1] + np.random.rand(k) * (edges[1:] - edges[:-1])
    else:
        c = np.linspace(a, b, k)
    return np.clip(np.round(c).astype(int), 1, max(1, n - 2))


def series_windows(d: Dict[str, np.ndarray], k: int, train: bool, w: Optional[dict] = None):
    """(S*k, 3, H, W) uint8 windows, slot ids, mask (False for absent slots), positions."""
    w = w or {}
    img, slot, cand, present = d["image"], d["slot"], d["cand"], d["valid"] > 0
    lo, hi = w.get("span", (0.04, 0.96))
    h, wd = img.shape[-2:]
    wins = np.zeros((N_SERIES_SLOTS, k, 3, h, wd), np.uint8)
    mask = np.zeros((N_SERIES_SLOTS, k), bool)
    pos = np.zeros((N_SERIES_SLOTS, k), np.float32)
    for s in range(N_SERIES_SLOTS):
        cands = sorted(set(cand[(slot == s) & present].tolist()))
        if not cands:
            continue
        ci = cands[np.random.randint(len(cands))] if (train and w.get("random_series")) else cands[0]
        idx = np.where((slot == s) & (cand == ci) & present)[0]
        n = len(idx)
        c = reader_centres(n, k, train, lo, hi)
        for j, cj in enumerate(c):
            wins[s, j] = img[idx[np.clip([cj - 1, cj, cj + 1], 0, n - 1)]]
        mask[s], pos[s] = True, c / max(n - 1, 1)
    slots = np.repeat(np.arange(N_SERIES_SLOTS), k).astype(np.int64)
    return wins.reshape(N_SERIES_SLOTS * k, 3, h, wd), slots, mask.reshape(-1), pos.reshape(-1)


def make_windows(d: Dict[str, np.ndarray], k: int, train: bool, w: Optional[dict] = None):
    """Windows of one cached study for either layout: (wins uint8 (K,3,H,W), slot (K,), mask (K,), pos (K,))."""
    fn = series_windows if "cand" in d else study_windows
    wins, slots, mask, pos = fn(d, k, train, w)
    if not mask.any():  # empty study: attend to everything (predicts the prior) instead of NaN
        mask = np.ones_like(mask)
    return wins, slots, mask, pos.astype(np.float32)


def eval_windows(d: Dict[str, np.ndarray], k: int, w: Optional[dict] = None):
    """Deterministic evaluation windows (used by the streaming inference)."""
    return make_windows(d, k, False, w)


# --------------------------------------------------------------------------------------------- pixel side
def reader_views(wins: np.ndarray, res: int, group: int, fov: float = 0.92, train: bool = False,
                 aug: bool = False) -> np.ndarray:
    """goodpjw's make_views on CPU: crop a centred `fov` fraction, resize to res; in training a rotation
    (+-12 deg), scale (+-12%), shift (+-6% of the half-width) and gamma / gain jitter shared within each series
    (`group` consecutive windows). uint8 (N,3,H,W) -> float32 (N,3,res,res) in [0, ~1.1]."""
    n, c, h, wd = wins.shape
    out = np.zeros((n, c, res, res), np.float32)
    for g0 in range(0, n, group):
        if aug and train:
            ang = (random.random() - 0.5) * 2 * math.radians(12)
            sc = fov * (1 + (random.random() - 0.5) * 0.24)
            tx, ty = (random.random() - 0.5) * 0.12, (random.random() - 0.5) * 0.12
            gamma, gain = math.exp((random.random() - 0.5) * 0.5), 1 + (random.random() - 0.5) * 0.2
        else:
            ang, sc, tx, ty, gamma, gain = 0.0, fov, 0.0, 0.0, None, None
        a11, a12, a21, a22 = math.cos(ang) * sc, -math.sin(ang) * sc, math.sin(ang) * sc, math.cos(ang) * sc
        c0 = 1.0 / res - 1.0  # normalised output coordinate of pixel 0 (align_corners=False)
        m = np.array([
            [wd / res * a11, wd / res * a12, wd / 2 * ((a11 + a12) * c0 + tx + 1) - 0.5],
            [h / res * a21, h / res * a22, h / 2 * ((a21 + a22) * c0 + ty + 1) - 0.5],
        ], np.float64)
        blk = wins[g0:g0 + group]
        flat = blk.reshape(-1, h, wd).transpose(1, 2, 0)  # H, W, group*3
        warped = cv2.warpAffine(np.ascontiguousarray(flat), m, (res, res),
                                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
        y = warped.reshape(res, res, -1).transpose(2, 0, 1).reshape(len(blk), c, res, res).astype(np.float32) / 255.0
        if gamma is not None:
            y = np.power(np.clip(y, 1e-4, None), gamma) * gain
        out[g0:g0 + group] = y
    return out


def to_model_input(wins: np.ndarray, img_size: int, train: bool, w: Optional[dict] = None,
                   aug_cfg: Optional[dict] = None, study_aug: Optional[StudyAugment] = None) -> torch.Tensor:
    """uint8 windows -> float tensor (K,3,img,img) in [0, ~1] (normalisation is applied by the caller)."""
    w, preset = w or {}, (aug_cfg or {}).get("preset", "ours")
    if w.get("fov") is not None:  # series layout: crop + resize (+ per-series augmentation) in one warp
        group = int(w.get("_group", len(wins)))
        x = reader_views(wins, img_size, group, float(w["fov"]), train, aug=train and preset == "reader_gpu")
        return torch.from_numpy(x)
    if train and preset == "gain_only":
        x = wins.astype(np.float32) / 255.0 * (1.0 + random.uniform(-0.1, 0.1))
    elif train and study_aug is not None:
        x = study_aug(wins)
    else:
        x = wins.astype(np.float32) / 255.0
    x = torch.from_numpy(np.ascontiguousarray(x))
    if x.shape[-1] != img_size:
        x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return x


class KneeStudyDataset(Dataset):
    """
    Args:
        study_ids: StudyInstanceUIDs
        cache_dir: directory of <uid>.npz produced by scripts/preprocess.py
        targets: uid -> float32[12] soft targets (None at inference)
        k: windows per study (series layout: per slot); 0 = every position (study layout, eval)
        img_size: model input size
        windows: cfg.windows (sampling, cross_slot, span, random_series, fov)
        mean/std: backbone normalisation constants (each checkpoint has its own!)
    """

    def __init__(
        self,
        study_ids: List[str],
        cache_dir: str,
        targets: Optional[Dict[str, np.ndarray]] = None,
        k: int = 16,
        img_size: int = 224,
        train: bool = False,
        aug_cfg: Optional[dict] = None,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        window_dropout: float = 0.0,
        windows: Optional[dict] = None,
    ):
        self.ids = list(study_ids)
        self.cache_dir = cache_dir
        self.targets = targets
        self.k, self.img_size, self.train = k, img_size, train
        self.aug_cfg = aug_cfg
        preset = (aug_cfg or {}).get("preset", "ours")
        self.aug = StudyAugment(aug_cfg) if (train and aug_cfg is not None and preset == "ours") else None
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.window_dropout = window_dropout if train else 0.0
        self.w = dict(windows or {})

    def __len__(self) -> int:
        return len(self.ids)

    def load(self, uid: str) -> Dict[str, np.ndarray]:
        with np.load(os.path.join(self.cache_dir, f"{uid}.npz")) as z:
            return {k: z[k] for k in z.files}

    def __getitem__(self, i: int):
        uid = self.ids[i]
        d = self.load(uid)
        wins, slots, mask, pos = make_windows(d, self.k, self.train, self.w)
        w = dict(self.w, _group=self.k) if "cand" in d else {k: v for k, v in self.w.items() if k != "fov"}
        x = to_model_input(wins, self.img_size, self.train, w, self.aug_cfg, self.aug)
        x = (x - self.mean) / self.std
        mask = torch.from_numpy(mask)
        if self.window_dropout > 0:
            keep = torch.rand(len(mask)) >= self.window_dropout
            if (mask & keep).any():
                mask = mask & keep
        out = {"x": x, "slot": torch.from_numpy(slots), "mask": mask, "pos": torch.from_numpy(pos), "uid": uid}
        if self.targets is not None:
            out["y"] = torch.from_numpy(np.asarray(self.targets[uid], dtype=np.float32))
        return out
