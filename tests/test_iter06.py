"""ITER-06: Raptor / reader pipeline options (cache side)."""
import os

import numpy as np
import pandas as pd

from src.config import get_config
from src.constants import ID_COL
from src.dataset.dicom import (SlotSpec, build_series_stack, build_study_stack, canonical_inplane, choose_series,
                               slot_specs_from_cfg)
from src.models.inference import _stack, empty_stack


def _series(rows):
    return pd.DataFrame(rows, columns=["SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"])


def test_choose_series_any_fat_and_most_slices():
    s = _series([("a", "Coronal", 1, 1), ("b", "Coronal", 0, 0), ("c", "Coronal", 0, 1)])
    assert choose_series(s, SlotSpec("x", "Coronal", -1, 5), {"a"}) == "b"          # any fluid, skips used
    assert choose_series(s, SlotSpec("x", "Coronal", -1, 5, fat=1), {"a"}) == "c"   # fat-suppression match
    assert choose_series(s, SlotSpec("x", "Coronal", 0, 5), set()) == "b"           # CSV order
    assert choose_series(s, SlotSpec("x", "Coronal", 0, 5), set(), {"b": 10, "c": 30}) == "c"  # most slices
    assert choose_series(s, SlotSpec("x", "Axial", 1, 5), set()) is None


def test_slot_specs_defaults():
    specs = slot_specs_from_cfg({"a": {"plane": "Axial", "fat": 0}, "b": {"plane": "Sagittal", "fluid": 1, "n": 3}})
    assert (specs[0].fluid, specs[0].n, specs[0].fat) == (-1, 0, 0)
    assert (specs[1].fluid, specs[1].n, specs[1].fat) == (1, 3, None)


def test_canonical_inplane_flips_and_transposes():
    vol = np.zeros((1, 4, 6), np.float32)
    vol[0, 0, 0] = 1  # marker in the first row / first column
    # coronal: columns must run toward +x (patient left), rows toward -z (inferior)
    out, sp = canonical_inplane(vol, (0.5, 0.7), [-1, 0, 0], [0, 0, -1], axis=1)
    assert out[0, 0, -1] == 1 and sp == (0.5, 0.7)                                  # column flip only
    out, sp = canonical_inplane(vol, (0.5, 0.7), [0, 0, -1], [1, 0, 0], axis=1)
    assert out.shape == (1, 6, 4) and sp == (0.7, 0.5) and out[0, 0, 0] == 1         # transposed frame


def test_series_stack_on_synthetic_study(fake_comp):
    raw = fake_comp / "raw"
    series = pd.read_csv(raw / "train_series.csv", dtype={ID_COL: str, "SeriesInstanceUID": str})
    uid = series[ID_COL].iloc[0]
    d = build_series_stack(str(raw / "train_series" / uid), series[series[ID_COL] == uid], mm_per_px=0.55, size=64)
    assert d["image"].shape[1:] == (64, 64) and d["image"].dtype == np.uint8
    assert sorted(set(d["slot"].tolist())) == [0, 1, 2, 3, 4, 5] and d["valid"].all()
    assert (d["cand"] == 0).all() and d["pos"].min() == 0 and d["pos"].max() == 1
    # fixture intensity = 100 + 50 * z; canonical slice order: sagittal right->left (z down), coronal anterior->
    # posterior (z up), axial superior->inferior (z down)
    centre = d["image"][:, 32, 32].astype(int)
    for slot, sign in ((0, -1), (1, +1), (2, -1)):
        v = centre[d["slot"] == slot]
        assert np.all(np.sign(np.diff(v)) * sign >= 0) and v[0] != v[-1], (slot, v)


def test_stack_dispatch_and_empty(fake_comp):
    raw = fake_comp / "raw"
    series = pd.read_csv(raw / "train_series.csv", dtype={ID_COL: str, "SeriesInstanceUID": str})
    uid = series[ID_COL].iloc[0]
    sdir, rows = str(raw / "train_series" / uid), series[series[ID_COL] == uid]
    pp = dict(layout="series", res=48, mm_per_px=0.7, max_slices=8, clip_percentiles=[1, 99.5], max_cands=2)
    d = _stack(uid, sdir, rows, pp, {})
    assert d["image"].shape[1:] == (48, 48) and d["image"].shape[0] == 6 * 8  # 12 native slices -> linspace 8
    assert empty_stack(pp)["valid"].sum() == 0
    cfg = get_config(["exp=raptor_repro"])
    spp = dict(cfg.preprocess)
    spp.update(res=32, slots={k: dict(v) for k, v in cfg.preprocess.slots.items()})
    d2 = _stack(uid, sdir, rows, spp, {})
    assert d2["image"].shape == (80, 32, 32) and d2["valid"].all()
    d3 = build_study_stack(sdir, rows, slot_specs_from_cfg(spp["slots"]), res=32, series_pick="most_slices")
    assert d3["image"].shape == (80, 32, 32)


def test_exp_slots_replace_defaults():
    assert list(get_config(["exp=raptor_repro"]).preprocess.slots) == ["sag_fluid", "sag_2nd", "cor_fluid",
                                                                      "cor_2nd", "axial"]
    assert len(get_config(["exp=reader_repro"]).preprocess.slots) == 6
    assert len(get_config(["exp=convnext_288_dense"]).preprocess.slots) == 6
