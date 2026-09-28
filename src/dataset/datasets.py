"""
Study-level dataset: each item is a bag of K 2.5D windows (3 physically adjacent slices of the
same slot stacked as RGB) plus the slot id of each window.
"""
import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.dataset.augmentations import StudyAugment


def window_centers(slot: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Indices c where c-1, c, c+1 are valid and belong to the same slot."""
    c = np.arange(1, len(slot) - 1)
    ok = (
        (slot[c - 1] == slot[c]) & (slot[c + 1] == slot[c])
        & (valid[c - 1] > 0) & (valid[c] > 0) & (valid[c + 1] > 0)
    )
    return c[ok]


def select_centers(centers: np.ndarray, k: int, train: bool) -> np.ndarray:
    """Random (train) or evenly spaced (eval) choice of k window centres."""
    if len(centers) == 0:
        return np.ones(k, dtype=int)
    if train:
        return np.random.choice(centers, k, replace=len(centers) < k)
    return centers[np.linspace(0, len(centers) - 1, k).round().astype(int)]


def eval_windows(d: Dict[str, np.ndarray], k: int):
    """Deterministic evaluation windows of one cached study: uint8 (K, 3, H, W) and slot ids (K,).
    Same selection as KneeStudyDataset(train=False); used by the streaming inference."""
    cs = select_centers(window_centers(d["slot"], d["valid"]), k, train=False)
    return np.stack([d["image"][c - 1: c + 2] for c in cs]), d["slot"][cs].astype(np.int64)


class KneeStudyDataset(Dataset):
    """
    Args:
        study_ids: StudyInstanceUIDs
        cache_dir: directory of <uid>.npz produced by scripts/preprocess.py
        targets: uid -> float32[12] soft targets (None at inference)
        k: windows per study
        img_size: model input size (cache is resized on the fly if different)
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
    ):
        self.ids = list(study_ids)
        self.cache_dir = cache_dir
        self.targets = targets
        self.k, self.img_size, self.train = k, img_size, train
        self.aug = StudyAugment(aug_cfg) if (train and aug_cfg is not None) else None
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.window_dropout = window_dropout if train else 0.0

    def __len__(self) -> int:
        return len(self.ids)

    def load(self, uid: str) -> Dict[str, np.ndarray]:
        with np.load(os.path.join(self.cache_dir, f"{uid}.npz")) as z:
            return {k: z[k] for k in z.files}

    def __getitem__(self, i: int):
        uid = self.ids[i]
        d = self.load(uid)
        img, slot, valid = d["image"], d["slot"], d["valid"]
        cs = select_centers(window_centers(slot, valid), self.k, self.train)
        wins = np.stack([img[c - 1 : c + 2] for c in cs])  # (K, 3, H, W) uint8
        x = self.aug(wins) if self.aug is not None else wins.astype(np.float32) / 255.0
        x = torch.from_numpy(np.ascontiguousarray(x))
        if x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std

        mask = torch.ones(self.k, dtype=torch.bool)
        if self.window_dropout > 0:
            mask = torch.rand(self.k) >= self.window_dropout
            if not mask.any():
                mask[random.randrange(self.k)] = True
        out = {"x": x, "slot": torch.from_numpy(slot[cs].astype(np.int64)), "mask": mask, "uid": uid}
        if self.targets is not None:
            out["y"] = torch.from_numpy(np.asarray(self.targets[uid], dtype=np.float32))
        return out
