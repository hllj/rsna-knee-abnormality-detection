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
  * Laterality tags are mostly empty: the knee side is recovered from geometry and every
    study is canonicalised to a right knee (never random flips: medial/lateral matters).
  * Median FOV is 160 mm: a crop <= 140 mm is needed for the physical crop to actually apply.

Two readers produce the same stack layout:
  * load_series() — decodes every slice (used to build the training cache).
  * selective path (build_study_stack(selective=True), used at test time) — reads all headers
    (threaded, no pixels) to sort and orient, then decodes only the slices that are sampled into
    the stack (~72 of ~165 per study). Uncompressed files skip pydicom's pixel handlers.
    Percentile windows are computed on the decoded slices (plus every `pct_stride`-th slice of the
    series when > 0); pct_stride=1 reproduces the training cache exactly.
"""
import os
from concurrent.futures import ThreadPoolExecutor
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
    fluid: int  # 1 / 0, or -1 = any series of the plane
    n: int
    fat: Optional[int] = None  # match Fat_Suppression instead of Fluid_Sensitive (reader-style slots)


def slot_specs_from_cfg(slots_cfg) -> List[SlotSpec]:
    """Build SlotSpecs from the `preprocess.slots` config mapping (order is preserved)."""
    return [SlotSpec(k, v["plane"], int(v.get("fluid", -1)), int(v.get("n", 0)),
                     None if v.get("fat") is None else int(v["fat"])) for k, v in slots_cfg.items()]


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


def load_series(series_dir: str) -> Tuple[np.ndarray, Tuple[float, float], Dict]:
    """
    Read every slice of one series, sorted in physical order.

    Returns:
        vol: float32 (N, H, W) in modality units
        spacing: (row_mm, col_mm) pixel spacing of the first slice (1.0 if absent)
        geo: per-slice centres / row direction / laterality tag (see _geometry)
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
    return vol, spacing, _geometry([d for _, _, d in items], spacing, (h, w))


UNCOMPRESSED_TS = {"1.2.840.10008.1.2", "1.2.840.10008.1.2.1"}  # implicit / explicit VR little endian


def _read_header(path: str):
    try:
        return pydicom.dcmread(path, stop_before_pixels=True)
    except Exception:
        return None


def decode_slice(path: str) -> np.ndarray:
    """Pixels of one slice as float32 modality values; raw-buffer fast path for plain uncompressed files."""
    ds = pydicom.dcmread(path)
    ts = str(getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", ""))
    fast = (ts in UNCOMPRESSED_TS and int(getattr(ds, "SamplesPerPixel", 1)) == 1
            and int(getattr(ds, "NumberOfFrames", 1) or 1) == 1 and int(getattr(ds, "PixelRepresentation", 0)) == 0
            and int(ds.BitsAllocated) in (8, 16) and int(getattr(ds, "BitsStored", ds.BitsAllocated)) == int(ds.BitsAllocated))
    if fast:
        dt = np.uint8 if int(ds.BitsAllocated) == 8 else np.dtype("<u2")
        arr = np.frombuffer(ds.PixelData, dtype=dt, count=int(ds.Rows) * int(ds.Columns))
        arr = arr.reshape(int(ds.Rows), int(ds.Columns)).astype(np.float32)
    else:
        arr = ds.pixel_array.astype(np.float32)
        if arr.ndim == 3:  # rare multi-frame / RGB
            arr = arr.mean(-1) if arr.shape[-1] in (3, 4) else arr[0]
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    inter = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    return arr * slope + inter


@dataclass
class SeriesHandle:
    """A series whose headers are read and sorted but whose pixels are decoded on demand."""

    paths: List[str]
    spacing: Tuple[float, float]
    shape: Tuple[int, int]
    geo: Dict


def open_series(series_dir: str, pool: ThreadPoolExecutor) -> SeriesHandle:
    files = [os.path.join(series_dir, f) for f in os.listdir(series_dir) if f.endswith(".dcm")]
    hdrs = list(pool.map(_read_header, files))
    items = sorted(((_slice_sort_key(h), f, h) for f, h in zip(files, hdrs) if h is not None), key=lambda t: t[0])
    if not items:
        raise ValueError(f"no readable headers in {series_dir}")
    first = items[0][2]
    ps = getattr(first, "PixelSpacing", None)
    spacing = (float(ps[0]), float(ps[1])) if ps is not None else (1.0, 1.0)
    shape = (int(first.Rows), int(first.Columns))
    return SeriesHandle([f for _, f, _ in items], spacing, shape, _geometry([h for _, _, h in items], spacing, shape))


def decode_slices(h: SeriesHandle, positions: Sequence[int], pool: ThreadPoolExecutor) -> Dict[int, np.ndarray]:
    """Decode the given sorted positions (threaded), resized to the series' first-slice shape."""
    pos = sorted(set(int(p) for p in positions))
    arrs = pool.map(decode_slice, [h.paths[p] for p in pos])
    out = {}
    for p, a in zip(pos, arrs):
        out[p] = a if a.shape == h.shape else cv2.resize(a, (h.shape[1], h.shape[0]), interpolation=cv2.INTER_AREA)
    return out


def _geometry(dss, spacing, shape) -> Dict:
    """Per-slice image-centre positions (patient LPS mm), row direction and laterality tag."""
    geo: Dict = {"centres": None, "row_cos": None, "tag": ""}
    for tag in ("Laterality", "ImageLaterality"):
        v = str(getattr(dss[0], tag, "") or "").strip().upper()
        if v in ("L", "R"):
            geo["tag"] = v
            break
    try:
        iop = np.asarray(dss[0].ImageOrientationPatient, dtype=np.float64)
        row_cos, col_cos = iop[:3], iop[3:]
        h, w = shape
        # IPP is the first voxel (a corner): walk half the image to reach the centre
        half = row_cos * spacing[1] * (w / 2) + col_cos * spacing[0] * (h / 2)
        geo["centres"] = np.stack([np.asarray(d.ImagePositionPatient, dtype=np.float64) + half for d in dss])
        geo["row_cos"], geo["col_cos"] = row_cos, col_cos
    except (AttributeError, ValueError, TypeError):
        pass
    return geo


def knee_side(geos: List[Dict], dead_zone_mm: float = 25.0) -> int:
    """+1 right knee, -1 left knee, 0 unknown. DICOM LPS: +x is the patient's left.

    Geometry first (centre x of the imaged volume); the Laterality tag is empty in ~72% of
    studies, so it is only used inside the dead zone around the scanner midline.
    """
    xs = [g["centres"][:, 0].mean() for g in geos if g.get("centres") is not None]
    if xs:
        x = float(np.median(xs))
        if abs(x) > dead_zone_mm:
            return -1 if x > 0 else 1
    tags = [g["tag"] for g in geos if g.get("tag")]
    if tags:
        return 1 if max(set(tags), key=tags.count) == "R" else -1
    return 0


def canonical_order(geo: Dict, n: int, plane: str, side: int) -> Tuple[np.ndarray, bool]:
    """Slice order and horizontal flip that make a series look like a right knee.

    * slices are ordered along the dominant patient axis (ascending), and sagittal stacks
      additionally run lateral -> medial, so slice index means the same anatomy everywhere
    * coronal / axial images are mirrored so the lateral side is always on the image left
    """
    order = np.arange(n)
    c = geo.get("centres")
    if c is not None and len(c) > 1:
        axis = int(np.argmax(np.abs(c[-1] - c[0])))
        order = np.argsort(c[:, axis])
        if plane == "Sagittal" and side == -1:  # left knee: lateral is +x, so descend
            order = order[::-1]
    rc = geo.get("row_cos")
    # right knee: lateral = -x; keep columns increasing toward +x (lateral on the left)
    flip = bool(plane != "Sagittal" and side != 0 and rc is not None and abs(rc[0]) > 0.5 and np.sign(rc[0]) != side)
    return order, flip


def canonicalise(vol: np.ndarray, geo: Dict, plane: str, side: int) -> np.ndarray:
    """Re-orient a whole series as if every knee were a right knee (see canonical_order)."""
    order, flip = canonical_order(geo, len(vol), plane, side)
    vol = vol[order]
    if flip:
        vol = vol[:, :, ::-1]
    return np.ascontiguousarray(vol)


def percentile_window(vol: np.ndarray, pcts: Sequence[float] = (1, 99)) -> Tuple[float, float]:
    """(lo, hi) intensity window from a strided subsample of the slices (sorting is slow)."""
    lo, hi = np.percentile(vol[:, ::4, ::4].ravel(), pcts)
    return float(lo), float(hi if hi > lo else lo + 1.0)


def apply_window(vol: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (np.clip((vol - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def to_uint8(vol: np.ndarray, pcts: Sequence[float] = (1, 99)) -> np.ndarray:
    """Per-series percentile window to uint8."""
    return apply_window(vol, *percentile_window(vol, pcts))


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


def choose_series(series: pd.DataFrame, spec: SlotSpec, used: set,
                  counts: Optional[Dict[str, int]] = None) -> Optional[str]:
    """Pick the series for a slot: exact plane+fluid (or fat-suppression) match first, then any same-plane
    series. Among candidates: CSV order, or the most slices when `counts` (series uid -> files) is given."""
    same_plane = series[series["Anatomical_Plane"] == spec.plane]
    if spec.fat is not None:
        exact = same_plane[same_plane["Fat_Suppression"] == spec.fat]
    elif spec.fluid >= 0:
        exact = same_plane[same_plane["Fluid_Sensitive"] == spec.fluid]
    else:
        exact = same_plane
    for cand in (exact, same_plane):
        cand = cand[~cand["SeriesInstanceUID"].isin(used)] if len(cand) > 1 else cand
        if len(cand):
            if counts:
                n = cand["SeriesInstanceUID"].astype(str).map(lambda u: counts.get(u, 0))
                cand = cand.iloc[np.argsort(-n.to_numpy(), kind="stable")]
            return str(cand["SeriesInstanceUID"].iloc[0])
    return None


def series_file_counts(study_dir: str, uids) -> Dict[str, int]:
    """Number of .dcm files per series directory (cheap proxy for slice count)."""
    out = {}
    for u in uids:
        try:
            out[str(u)] = sum(f.endswith(".dcm") for f in os.listdir(os.path.join(study_dir, str(u))))
        except OSError:
            out[str(u)] = 0
    return out


def build_study_stack(
    study_dir: str,
    series: pd.DataFrame,
    slots: List[SlotSpec],
    res: int = 256,
    crop_mm: float = 140.0,
    span: Sequence[float] = (0.04, 0.96),
    pcts: Sequence[float] = (1, 99),
    canonical: bool = True,
    selective: bool = False,
    pct_stride: int = 0,
    threads: int = 8,
    series_pick: str = "first",
) -> Dict[str, np.ndarray]:
    """
    Build the cached representation of one study.

    Args:
        study_dir: <root>/<StudyInstanceUID>
        series: rows of train_series.csv / test_series.csv for this study
        canonical: mirror left knees to right and order slices anatomically
        selective: decode only the sampled slices (test-time speed-up); False = decode every slice
        pct_stride: selective only — also decode every k-th slice for the percentile window (0 = none)
        series_pick: "first" (CSV order) or "most_slices" among the candidate series of a slot

    Returns:
        dict with
          image:  uint8 (D, res, res), D = sum(slot.n)
          slot:   int8  (D,) slot index of each slice
          valid:  uint8 (D,) 0 where the slot had no series
          side:   int8  ()   +1 right / -1 left / 0 unknown knee
    """
    total = sum(s.n for s in slots)
    image = np.zeros((total, res, res), dtype=np.uint8)
    slot_ids = np.zeros(total, dtype=np.int8)
    valid = np.zeros(total, dtype=np.uint8)

    chosen: List[Optional[str]] = []
    used: set = set()
    counts = series_file_counts(study_dir, series["SeriesInstanceUID"]) if series_pick == "most_slices" else None
    for spec in slots:
        uid = choose_series(series, spec, used, counts)
        chosen.append(uid)
        if uid is not None:
            used.add(uid)
    if selective:
        try:
            return _build_selective(study_dir, slots, chosen, res, crop_mm, span, pcts, canonical, pct_stride,
                                    threads, image, slot_ids, valid)
        except Exception:
            pass  # any surprise (corrupt slice, odd header): fall back to the full decode below
        image[:] = 0
        valid[:] = 0
    loaded: Dict[str, Tuple] = {}
    for uid in {u for u in chosen if u is not None}:
        try:
            loaded[uid] = load_series(os.path.join(study_dir, uid))
        except Exception:
            pass  # unreadable series: its slot stays empty (masked)
    side = knee_side([g for _, _, g in loaded.values()]) if canonical else 0

    pos = 0
    for si, (spec, uid) in enumerate(zip(slots, chosen)):
        slot_ids[pos : pos + spec.n] = si
        if uid in loaded:
            vol, spacing, geo = loaded[uid]
            if canonical:
                vol = canonicalise(vol, geo, spec.plane, side)
            idx = sample_indices(len(vol), spec.n, span)
            sl = center_crop_mm(to_uint8(vol, pcts)[idx], spacing, crop_mm)
            image[pos : pos + spec.n] = np.stack(
                [cv2.resize(s, (res, res), interpolation=cv2.INTER_AREA) for s in sl]
            )
            valid[pos : pos + spec.n] = 1
        pos += spec.n
    return {"image": image, "slot": slot_ids, "valid": valid, "side": np.int8(side)}


def _build_selective(study_dir, slots, chosen, res, crop_mm, span, pcts, canonical, pct_stride, threads,
                     image, slot_ids, valid) -> Dict[str, np.ndarray]:
    with ThreadPoolExecutor(threads) as pool:
        handles: Dict[str, SeriesHandle] = {}
        for uid in dict.fromkeys(u for u in chosen if u is not None):
            try:
                handles[uid] = open_series(os.path.join(study_dir, uid), pool)
            except Exception:
                pass
        side = knee_side([h.geo for h in handles.values()]) if canonical else 0

        # which sorted positions each slot needs, and the extra ones for percentile windows
        plan = []
        need: Dict[str, set] = {u: set() for u in handles}
        for spec, uid in zip(slots, chosen):
            if uid not in handles:
                plan.append(None)
                continue
            h = handles[uid]
            order, flip = canonical_order(h.geo, len(h.paths), spec.plane, side) if canonical else (np.arange(len(h.paths)), False)
            take = order[sample_indices(len(order), spec.n, span)]
            plan.append((uid, take, flip))
            need[uid].update(int(t) for t in take)
        pct_pos = {u: set(range(0, len(h.paths), pct_stride)) if pct_stride > 0 else set() for u, h in handles.items()}
        decoded = {u: decode_slices(handles[u], need[u] | pct_pos[u], pool) for u in handles if need[u]}

    pos = 0
    windows: Dict[str, Tuple[float, float]] = {}
    for si, (spec, p) in enumerate(zip(slots, plan)):
        slot_ids[pos: pos + spec.n] = si
        if p is not None:
            uid, take, flip = p
            d = decoded[uid]
            if uid not in windows:  # one window per series, as in the training cache
                keys = sorted(need[uid] | pct_pos[uid])
                windows[uid] = percentile_window(np.stack([d[k] for k in keys]), pcts)
            vol = np.stack([d[int(t)] for t in take])
            if flip:
                vol = vol[:, :, ::-1]
            sl = center_crop_mm(apply_window(vol, *windows[uid]), handles[uid].spacing, crop_mm)
            image[pos: pos + spec.n] = np.stack([cv2.resize(np.ascontiguousarray(x), (res, res),
                                                            interpolation=cv2.INTER_AREA) for x in sl])
            valid[pos: pos + spec.n] = 1
        pos += spec.n
    return {"image": image, "slot": slot_ids, "valid": valid, "side": np.int8(side)}


# ------------------------------------------------------------------------------------------ series layout
# Reader-style cache (goodpjw2008/rsna-knee-2-5d-convnext-reader, Apache-2.0): every series of a study is kept
# with its native slices (<= max_slices), reoriented to a fixed patient frame, resampled to a fixed mm/px.
PLANE_NAMES = ("Sagittal", "Coronal", "Axial")
# patient axis (LPS) -> (slice-order sign, (image-column axis, sign), (image-row axis, sign))
# Sagittal: slices right -> left, columns anterior -> posterior, rows superior -> inferior
# Coronal:  slices anterior -> posterior, columns patient right -> left, rows superior -> inferior
# Axial:    slices superior -> inferior, columns patient right -> left, rows anterior -> posterior
CANON_FRAME = {0: (+1, (1, +1), (2, -1)), 1: (+1, (0, +1), (2, -1)), 2: (-1, (0, +1), (1, +1))}


def canonical_inplane(vol: np.ndarray, spacing, row_cos, col_cos, axis: int):
    """Transpose / flip a (N, H, W) series into the fixed in-plane frame of its plane. Returns (vol, spacing)."""
    _, (cax, csgn), (rax, rsgn) = CANON_FRAME[axis]
    r, c = np.asarray(row_cos, float), np.asarray(col_cos, float)
    spacing = tuple(spacing)
    if abs(r[cax]) < abs(r[rax]):  # image rows and columns are swapped relative to the canonical frame
        vol, r, c, spacing = vol.transpose(0, 2, 1), c, r, spacing[::-1]
    if r[cax] * csgn < 0:
        vol = vol[:, :, ::-1]
    if c[rax] * rsgn < 0:
        vol = vol[:, ::-1, :]
    return np.ascontiguousarray(vol), spacing


def resample_frame(img: np.ndarray, spacing, mm_per_px: float, size: int) -> np.ndarray:
    """Centre crop to size*mm_per_px mm, resample to mm_per_px, zero-pad to (size, size)."""
    out = np.zeros((size, size), np.float32)
    h, w = img.shape
    ch = min(h, int(round(size * mm_per_px / spacing[0])))
    cw = min(w, int(round(size * mm_per_px / spacing[1])))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    crop = np.ascontiguousarray(img[y0:y0 + ch, x0:x0 + cw])
    nh = max(1, min(size, int(round(ch * spacing[0] / mm_per_px))))
    nw = max(1, min(size, int(round(cw * spacing[1] / mm_per_px))))
    res = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA if nh < ch else cv2.INTER_LINEAR)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = res
    return out


def build_series_stack(study_dir: str, series: pd.DataFrame, mm_per_px: float = 0.55, size: int = 280,
                       max_slices: int = 64, pcts: Sequence[float] = (1, 99.5), max_cands: int = 2,
                       min_slices: int = 3) -> Dict[str, np.ndarray]:
    """
    Reader-style cached representation of one study: all series, native slices.

    Slot = plane + 3 * (1 - Fat_Suppression) (fat-suppressed planes first); per slot the candidate series are
    sorted by slice count and the largest `max_cands` kept (cand 0 = largest).

    Returns dict with image uint8 (D, size, size), slot int8 (D,), cand int8 (D,), pos float16 (D,) = index /
    (n - 1) inside its series, valid uint8 (D,), side int8 () (knee side, informational).
    """
    vols, geos = [], []
    for row in series.itertuples(index=False):
        uid = str(row.SeriesInstanceUID)
        try:
            vol, spacing, geo = load_series(os.path.join(study_dir, uid))
        except Exception:
            continue
        rc, cc = geo.get("row_cos"), geo.get("col_cos")
        axis = int(np.abs(np.cross(rc, cc)).argmax()) if rc is not None and cc is not None else \
            PLANE_NAMES.index(row.Anatomical_Plane) if row.Anatomical_Plane in PLANE_NAMES else 0
        cen = geo.get("centres")
        if cen is not None and len(cen) == len(vol) and len(vol) > 1:
            vol = vol[np.argsort(CANON_FRAME[axis][0] * cen[:, axis], kind="stable")]
        if len(vol) > max_slices:
            vol = vol[np.linspace(0, len(vol) - 1, max_slices).round().astype(int)]
        if rc is not None and cc is not None:
            vol, spacing = canonical_inplane(vol, spacing, rc, cc, axis)
        res = np.stack([resample_frame(x, spacing, mm_per_px, size) for x in vol])
        lo, hi = np.percentile(res[:, ::4, ::4], pcts)
        u8 = (np.clip((res - lo) / max(hi - lo, 1e-6), 0, 1) * 255).round().astype(np.uint8)
        plane = row.Anatomical_Plane if row.Anatomical_Plane in PLANE_NAMES else PLANE_NAMES[axis]
        slot = PLANE_NAMES.index(plane) + 3 * (1 - int(row.Fat_Suppression))
        if len(u8) >= min_slices:
            vols.append((slot, len(u8), u8))
            geos.append(geo)
    by_slot: Dict[int, list] = {}
    for slot, n, u8 in vols:
        by_slot.setdefault(slot, []).append((n, u8))
    image, slot_ids, cand_ids, pos = [], [], [], []
    for slot in sorted(by_slot):
        for ci, (n, u8) in enumerate(sorted(by_slot[slot], key=lambda t: -t[0])[:max_cands]):
            image.append(u8)
            slot_ids.append(np.full(n, slot, np.int8))
            cand_ids.append(np.full(n, ci, np.int8))
            pos.append((np.arange(n) / max(n - 1, 1)).astype(np.float16))
    side = np.int8(knee_side(geos)) if geos else np.int8(0)
    if not image:
        return {"image": np.zeros((1, size, size), np.uint8), "slot": np.zeros(1, np.int8),
                "cand": np.zeros(1, np.int8), "pos": np.zeros(1, np.float16), "valid": np.zeros(1, np.uint8),
                "side": side}
    image = np.concatenate(image)
    return {"image": image, "slot": np.concatenate(slot_ids), "cand": np.concatenate(cand_ids),
            "pos": np.concatenate(pos), "valid": np.ones(len(image), np.uint8), "side": side}
