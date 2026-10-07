"""ITER-06: Raptor / reader pipeline options (training + inference side)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from src.constants import ID_COL, LABELS
from src.dataset.datasets import KneeStudyDataset, make_windows, reader_views, select_centers
from src.dataset.dicom import build_series_stack
from src.models.model import KneeMILModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("head", ["tanh_mil", "transformer_mil"])
def test_new_heads_mask_and_pos(head):
    m = KneeMILModel("resnet18", pretrained=False, head=head, num_slots=6).eval()
    x = torch.randn(2, 6, 3, 32, 32)
    slot = torch.randint(0, 6, (2, 6))
    pos = torch.rand(2, 6)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1] * 6], dtype=torch.bool)
    out = m(x, slot, mask, pos)
    assert out.shape == (2, 12) and torch.isfinite(out).all()
    x2 = x.clone()
    x2[0, 4:] = 50.0  # masked windows must not matter
    assert torch.allclose(m(x, slot, mask, pos)[0], m(x2, slot, mask, pos)[0], atol=1e-4)
    assert torch.isfinite(m(x, slot, torch.zeros_like(mask), pos)).all()  # all-masked study stays finite


def test_study_windows_cross_slot_all_positions():
    d = {"image": np.arange(10, dtype=np.uint8)[:, None, None].repeat(4, 1).repeat(4, 2),
         "slot": np.array([0] * 4 + [1] * 3 + [2] * 3, np.int8), "valid": np.array([1] * 7 + [0] * 3, np.uint8)}
    wins, slots, mask, pos = make_windows(d, 0, False, {"cross_slot": True})
    assert len(wins) == 8 and mask.tolist() == [True] * 6 + [False] * 2  # every centre 1..8; centred-on-empty masked
    wins, slots, mask, pos = make_windows(d, 3, False, {})
    assert mask.all() and set(slots.tolist()) <= {0, 1} and pos.max() <= 1
    c = select_centers(np.arange(1, 41), 4, True, "stratified")
    assert all(lo <= v < lo + 10 for v, lo in zip(sorted(c), (1, 11, 21, 31)))  # one per bin


def test_reader_views_identity_and_aug():
    rng = np.random.default_rng(0)
    wins = rng.integers(0, 255, (4, 3, 32, 32), dtype=np.uint8)
    x = reader_views(wins, 32, group=2, fov=1.0)
    np.testing.assert_allclose(x, wins / 255.0, atol=1e-3)  # fov 1, no aug = identity
    y = reader_views(wins, 16, group=2, fov=0.5)  # centre half at the same pixel scale
    np.testing.assert_allclose(y[0, 0, 4:12, 4:12], wins[0, 0, 12:20, 12:20] / 255.0, atol=1e-3)
    z = reader_views(wins, 24, group=2, fov=0.92, train=True, aug=True)
    assert z.shape == (4, 3, 24, 24) and np.isfinite(z).all()


def test_series_dataset(fake_comp, tmp_path):
    raw = fake_comp / "raw"
    series = pd.read_csv(raw / "train_series.csv", dtype={ID_COL: str, "SeriesInstanceUID": str})
    uid = series[ID_COL].iloc[0]
    d = build_series_stack(str(raw / "train_series" / uid), series[series[ID_COL] == uid], mm_per_px=0.7, size=48)
    np.savez(tmp_path / f"{uid}.npz", **d)
    w = {"span": [0.04, 0.96], "random_series": True, "fov": 0.92, "sampling": "stratified"}
    for train in (True, False):
        ds = KneeStudyDataset([uid], str(tmp_path), {uid: np.zeros(12, np.float32)}, k=3, img_size=40,
                              train=train, aug_cfg={"preset": "reader_gpu"}, windows=w)
        b = ds[0]
        assert b["x"].shape == (18, 3, 40, 40) and b["mask"].all() and b["pos"].shape == (18,)
        assert b["slot"].tolist() == sum([[s] * 3 for s in range(6)], [])


def _e2e(fake_comp, tmp_path, monkeypatch, exp, extra):
    monkeypatch.chdir(tmp_path)
    ov = [f"exp={exp}", f"paths.data_dir={fake_comp}", "labels.sources=[fake_labels.csv]", "data.num_folds=2",
          "preprocess.num_workers=2", "model.backbone=resnet18", "model.pretrained=false",
          "model.grad_checkpointing=false", "training.batch_size=2", "training.num_epochs=2",
          "training.precision=32", "resources.num_workers=0", "resources.devices=1", f"exp_name=smoke_{exp}",
          f"data.folds_file={tmp_path}/folds.csv", *extra]
    monkeypatch.setattr(sys, "argv", ["preprocess"] + ov)
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import preprocess

    preprocess.main()
    from src.models import train_pl

    monkeypatch.setattr(sys, "argv", ["train"] + ov + ["fold=0"])
    train_pl.main()
    assert os.path.exists(f"models/smoke_{exp}/fold0.pt")
    assert os.path.exists(f"models/oof/smoke_{exp}_gold_fold0.csv")
    assert os.path.exists(f"models/oof/smoke_{exp}_report_targets.csv")  # every run keeps its own targets
    os.makedirs("models/oof", exist_ok=True)
    pd.DataFrame({ID_COL: ["ruler"]}).to_csv("models/oof/report_targets.csv", index=False)
    monkeypatch.setattr(sys, "argv", ["train"] + ov + ["fold=0", "training.num_epochs=1"])
    train_pl.main()
    assert len(pd.read_csv("models/oof/report_targets.csv")) == 1  # an existing ruler is never overwritten
    from src.models import inference

    monkeypatch.setattr(sys, "argv", ["infer", "--data_dir", str(fake_comp / "raw"), "--ckpts",
                                      f"models/smoke_{exp}/*.pt", "--workers", "2", "--out", "sub.csv"])
    inference.main()
    sub = pd.read_csv("sub.csv")
    assert len(sub) == 3 and ((sub[LABELS] >= 0) & (sub[LABELS] <= 1)).all().all()


@pytest.mark.slow
def test_e2e_raptor_repro(fake_comp, tmp_path, monkeypatch):
    _e2e(fake_comp, tmp_path, monkeypatch, "raptor_repro", [
        "preprocess.res=48", "preprocess.crop_mm=30", "preprocess.slots.sag_fluid.n=5",
        "preprocess.slots.sag_2nd.n=4", "preprocess.slots.cor_fluid.n=4", "preprocess.slots.cor_2nd.n=4",
        "preprocess.slots.axial.n=4", "model.img_size=56", "windows.k_train=4"])


@pytest.mark.slow
def test_e2e_reader_repro(fake_comp, tmp_path, monkeypatch):
    _e2e(fake_comp, tmp_path, monkeypatch, "reader_repro", [
        "preprocess.res=48", "preprocess.mm_per_px=0.7", "preprocess.max_slices=8", "model.img_size=40",
        "windows.k_train=2", "windows.k_eval=3"])
