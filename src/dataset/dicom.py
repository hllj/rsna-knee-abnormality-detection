"""
DICOM -> fixed-size uint8 slice stack per study.

Used identically by scripts/preprocess.py (train cache) and the Kaggle inference kernel (test),
so train/test preprocessing can never drift apart.

Data traps handled here (from the competition forum):
  * File names are random SOP UIDs: sort slices by ImagePositionPatient projected on the slice
    normal, falling back to SliceLocation -> InstanceNumber. InstanceNumber alone is reversed in
    about a third of series.
  * Mixed transfer syntaxes (JPEG Lossless / JPEG 2000): needs pylibjpeg + openjpeg (or gdcm).
  * Slices inside one series can have different shapes: resized to the first slice's shape.
  * Laterality is missing for ~half the studies: do NOT mirror images; medial/lateral matters.
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import pydicom


@dataclass
class SlotSpec:
    """One input slot: a plane / fluid-sensitivity combination and how many slices to take."""

    name: str
    plane: str
    fluid: int
    n: int


def slot_specs_from_cfg(slots_cfg) -> List[SlotSpec]:
    """Build SlotSpecs from the `preprocess.slots` config mapping (order is preserved)."""
    return [SlotSpec(k, v["plane"], int(v["fluid"]), int(v["n"])) for k, v in slots_cfg.items()]


def _slice_sort_key(ds) -> Tuple[int, float]:
    try:
        ipp = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
        iop = np.asarray(ds.ImageOrientationPatient, dtype=np.float64)
        normal = np.cross(iop[:3], iop[3:])
        return 0, float(ipp @ normal)
    except (AttributeError, ValueError, TypeError):
        pass
    for tag in ("SliceLocation", "InstanceNumber"):
        v = getattr(ds, tag, None)
        if v is not None:
            return 1, float(v)
    return 2, 0.0


def load_series(series_dir: str) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Read every slice of one series, sorted in physical order.

    Returns:
        vol: float32 (N, H, W) in modality units
        spacing: (row_mm, col_mm) pixel spacing of the first slice (1.0 if absent)
    """
    files = [os.path.join(series_dir, f) for f in os.listdir(series_dir) if f.endswith(".dcm")]
    items = []
    for f in files:
        try:
            ds = pydicom.dcmread(f)
            arr = ds.pixel_array.astype(np.float32)
        except Exception:  # corrupt / undecodable slice: skip it
            continue
        if arr.ndim == 3:  # rare multi-frame / RGB
            arr = arr.mean(-1) if arr.shape[-1] in (3, 4) else arr[0]
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        inter = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        items.append((_slice_sort_key(ds), arr * slope + inter, ds))
    if not items:
        raise ValueError(f"no readable slices in {series_dir}")
    items.sort(key=lambda t: t[0])
    h, w = items[0][1].shape
    vol = np.stack(
        [a if a.shape == (h, w) else cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA) for _, a, _ in items]
    )
    ps = getattr(items[0][2], "PixelSpacing", None)
    spacing = (float(ps[0]), float(ps[1])) if ps is not None else (1.0, 1.0)
    return vol, spacing


def to_uint8(vol: np.ndarray, pcts: Sequence[float] = (1, 99)) -> np.ndarray:
    """Per-series percentile window to uint8 (percentiles from a strided subsample: sorting is slow)."""
    sub = vol[:, ::4, ::4].ravel()
    lo, hi = np.percentile(sub, pcts)
    if hi <= lo:
        hi = lo + 1.0
    return (np.clip((vol - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def center_crop_mm(img: np.ndarray, spacing: Tuple[float, float], crop_mm: float) -> np.ndarray:
    """Centre crop to a fixed physical field of view (zero-pads when the FOV is smaller)."""
    h, w = img.shape[-2:]
    ch = int(round(crop_mm / max(spacing[0], 1e-3)))
    cw = int(round(crop_mm / max(spacing[1], 1e-3)))
    out = np.zeros(img.shape[:-2] + (ch, cw), dtype=img.dtype)
    sy, sx = max(0, (h - ch) // 2), max(0, (w - cw) // 2)
    dy, dx = max(0, (ch - h) // 2), max(0, (cw - w) // 2)
    hh, ww = min(h, ch), min(w, cw)
    out[..., dy : dy + hh, dx : dx + ww] = img[..., sy : sy + hh, sx : sx + ww]
    return out


def sample_indices(n_avail: int, n: int, span: Sequence[float] = (0.04, 0.96)) -> np.ndarray:
    """n evenly spaced slice indices inside the central `span` fraction of the series."""
    lo = int(np.floor(span[0] * (n_avail - 1)))
    hi = int(np.ceil(span[1] * (n_avail - 1)))
    return np.linspace(lo, max(lo, hi), n).round().astype(int)


def choose_series(series: pd.DataFrame, spec: SlotSpec, used: set) -> Optional[str]:
    """Pick the series for a slot: exact plane+fluid match first, then any same-plane series."""
    same_plane = series[series["Anatomical_Plane"] == spec.plane]
    exact = same_plane[same_plane["Fluid_Sensitive"] == spec.fluid]
    for cand in (exact, same_plane):
        cand = cand[~cand["SeriesInstanceUID"].isin(used)] if len(cand) > 1 else cand
        if len(cand):
            return str(cand["SeriesInstanceUID"].iloc[0])
    return None


def build_study_stack(
    study_dir: str,
    series: pd.DataFrame,
    slots: List[SlotSpec],
    res: int = 256,
    crop_mm: float = 150.0,
    span: Sequence[float] = (0.04, 0.96),
    pcts: Sequence[float] = (1, 99),
) -> Dict[str, np.ndarray]:
    """
    Build the cached representation of one study.

    Args:
        study_dir: <root>/<StudyInstanceUID>
        series: rows of train_series.csv / test_series.csv for this study

    Returns:
        dict with
          image:  uint8 (D, res, res), D = sum(slot.n)
          slot:   int8  (D,) slot index of each slice
          valid:  uint8 (D,) 0 where the slot had no series
    """
    total = sum(s.n for s in slots)
    image = np.zeros((total, res, res), dtype=np.uint8)
    slot_ids = np.zeros(total, dtype=np.int8)
    valid = np.zeros(total, dtype=np.uint8)
    cache: Dict[str, Tuple[np.ndarray, Tuple[float, float]]] = {}
    used: set = set()
    pos = 0
    for si, spec in enumerate(slots):
        slot_ids[pos : pos + spec.n] = si
        uid = choose_series(series, spec, used)
        if uid is not None:
            try:
                if uid not in cache:
                    cache[uid] = load_series(os.path.join(study_dir, uid))
                vol, spacing = cache[uid]
                used.add(uid)
                idx = sample_indices(len(vol), spec.n, span)
                sl = to_uint8(vol, pcts)[idx]
                sl = center_crop_mm(sl, spacing, crop_mm)
                image[pos : pos + spec.n] = np.stack(
                    [cv2.resize(s, (res, res), interpolation=cv2.INTER_AREA) for s in sl]
                )
                valid[pos : pos + spec.n] = 1
            except Exception:
                pass  # leave the slot empty (masked)
        pos += spec.n
    return {"image": image, "slot": slot_ids, "valid": valid}
