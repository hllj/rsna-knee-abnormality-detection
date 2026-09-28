"""
Study-consistent augmentations for bags of 2.5D windows.

One random transform is drawn per study and applied to every window, so windows stay
anatomically registered. No horizontal/vertical flips: medial vs lateral compartment labels
depend on orientation.
"""
import random
from typing import Optional

import cv2
import numpy as np


class StudyAugment:
    """Random affine + intensity jitter applied identically to all windows of a study."""

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        ssr = cfg.get("shift_scale_rotate", {}) or {}
        self.shift = float(ssr.get("shift", 0.06))
        self.scale = float(ssr.get("scale", 0.1))
        self.rotate = float(ssr.get("rotate", 10))
        self.bc = float(cfg.get("brightness_contrast", 0.2))
        self.gamma = tuple(cfg.get("gamma", (0.8, 1.2)))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """x: uint8 (K, 3, H, W) -> float32 (K, 3, H, W) in [0, 1]."""
        k, c, h, w = x.shape
        ang = random.uniform(-self.rotate, self.rotate)
        sc = 1.0 + random.uniform(-self.scale, self.scale)
        m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
        m[:, 2] += (random.uniform(-self.shift, self.shift) * w, random.uniform(-self.shift, self.shift) * h)
        flat = x.reshape(k * c, h, w).transpose(1, 2, 0)  # H, W, K*C for one warp call
        out = np.empty_like(flat)
        for i in range(0, flat.shape[-1], 512):  # cv2 caps channels at 512
            out[..., i : i + 512] = cv2.warpAffine(
                np.ascontiguousarray(flat[..., i : i + 512]), m, (w, h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            ).reshape(h, w, -1)
        y = out.transpose(2, 0, 1).reshape(k, c, h, w).astype(np.float32) / 255.0
        y = np.power(y, random.uniform(*self.gamma))
        alpha = 1.0 + random.uniform(-self.bc, self.bc)
        beta = random.uniform(-self.bc, self.bc) * 0.5
        return np.clip(y * alpha + beta, 0.0, 1.0)
