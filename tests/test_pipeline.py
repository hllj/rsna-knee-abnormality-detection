"""Unit tests for preprocessing, labels, model and metric + an end-to-end smoke run."""
import glob
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from src.constants import ID_COL, LABELS
from src.dataset.datasets import window_centers
from src.dataset.dicom import center_crop_mm, load_series, sample_indices
from src.evaluation.metrics import macro_auc
from src.labeling.blend import apply_pseudo, blend_sources, fill_silence, load_source
from src.models.model import KneeMILModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_load_series_sorts_by_position(fake_comp):
    study = glob.glob(str(fake_comp / "raw" / "train_series" / "*"))[0]
    series = glob.glob(os.path.join(study, "*"))[0]
    vol, spacing, geo = load_series(series)
    # intensity encodes true z; must be monotonic despite shuffled file names / instance numbers
    means = vol.mean((1, 2))
    assert np.all(np.diff(means) > 0) or np.all(np.diff(means) < 0)
    assert spacing == (0.5, 0.5)
    assert geo["centres"].shape == (len(vol), 3)


def test_knee_side_and_canonicalise():
    from src.dataset.dicom import canonicalise, knee_side

    right = {"centres": np.array([[-80.0, 0, 0], [-60.0, 0, 0]]), "row_cos": np.array([1.0, 0, 0]), "tag": ""}
    left = {"centres": np.array([[60.0, 0, 0], [80.0, 0, 0]]), "row_cos": np.array([1.0, 0, 0]), "tag": ""}
    mid = {"centres": np.array([[5.0, 0, 0], [6.0, 0, 0]]), "row_cos": None, "tag": "L"}
    assert knee_side([right]) == 1 and knee_side([left]) == -1 and knee_side([mid]) == -1
    vol = np.arange(2 * 1 * 3, dtype=np.float32).reshape(2, 1, 3)
    # sagittal left knee: slices descend in x (lateral -> medial)
    assert canonicalise(vol, left, "Sagittal", -1)[0, 0, 0] == vol[1, 0, 0]
    assert canonicalise(vol, right, "Sagittal", 1)[0, 0, 0] == vol[0, 0, 0]
    # coronal left knee: mirrored columns; right knee untouched
    assert canonicalise(vol, left, "Coronal", -1)[0, 0].tolist() == vol[0, 0, ::-1].tolist()
    assert canonicalise(vol, right, "Coronal", 1)[0, 0].tolist() == vol[0, 0].tolist()


def test_crop_and_sampling():
    img = np.ones((2, 100, 80), dtype=np.uint8)
    out = center_crop_mm(img, (0.5, 0.5), 60)  # 120 px FOV > image -> zero padded
    assert out.shape == (2, 120, 120) and out.sum() == img.sum()
    idx = sample_indices(30, 8, (0.1, 0.9))
    assert len(idx) == 8 and idx.min() >= 2 and idx.max() <= 27


def test_window_centers_stay_in_slot():
    slot = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    valid = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    assert window_centers(slot, valid).tolist() == [1, 2, 5]


def test_silence_and_blend(tmp_path):
    a = pd.DataFrame({ID_COL: ["u1", "u2"], **{c: [0.5, 1.0] for c in LABELS}})
    a.loc[0, "Effusion"] = 0.9
    a.to_csv(tmp_path / "a.csv", index=False)
    vals, silent = load_source(str(tmp_path / "a.csv"))
    filled = fill_silence(vals, silent, {"default": 0.1, "Synovitis": "from_effusion", "Baker's": 0.03})
    assert filled.loc["u1", "Synovitis"] == pytest.approx(0.9)
    assert filled.loc["u1", "Baker's"] == pytest.approx(0.03)
    assert filled.loc["u1", "ACL"] == pytest.approx(0.1)
    assert filled.loc["u2", "ACL"] == 1.0
    b = pd.DataFrame({ID_COL: ["u2"], **{c: [0.0] for c in LABELS}})
    b.to_csv(tmp_path / "b.csv", index=False)
    y = blend_sources([str(tmp_path / "a.csv"), str(tmp_path / "b.csv")])
    assert y.loc["u2", "ACL"] == pytest.approx(0.5) and y.loc["u1", "Effusion"] == pytest.approx(0.9)
    oof = pd.DataFrame({ID_COL: ["u2"], **{c: [0.0] for c in LABELS}})
    oof.to_csv(tmp_path / "oof.csv", index=False)
    p = apply_pseudo(y, [str(tmp_path / "oof.csv")], 0.5, {"ACL": 1.0})
    assert p.loc["u2", "ACL"] == 0.0 and p.loc["u2", "MCL"] == pytest.approx(0.25)
    assert p.loc["u1", "MCL"] == y.loc["u1", "MCL"]  # no teacher prediction -> unchanged


@pytest.mark.parametrize("head", ["attn_mil", "mean", "max"])
def test_model_forward_with_mask(head):
    m = KneeMILModel("resnet18", pretrained=False, head=head, num_slots=6)
    x = torch.randn(2, 5, 3, 64, 64)
    slot = torch.randint(0, 6, (2, 5))
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    out = m(x, slot, mask)
    assert out.shape == (2, 12) and torch.isfinite(out).all()
    # masked windows must not influence the prediction
    x2 = x.clone()
    x2[0, 3:] = 100.0
    m.eval()
    assert torch.allclose(m(x, slot, mask)[0], m(x2, slot, mask)[0], atol=1e-5)


def test_macro_auc_skips_single_class():
    y = np.zeros((4, 12))
    y[:2, 0] = 1
    p = np.zeros((4, 12))
    p[:2, 0] = 0.9
    m, per = macro_auc(y, p)
    assert m == 1.0 and list(per) == ["ACL"]


@pytest.mark.slow
def test_end_to_end(fake_comp, tmp_path, monkeypatch):
    """preprocess -> train 2 epochs (fold 0) -> OOF/gold files -> inference -> submission.csv"""
    monkeypatch.chdir(tmp_path)
    ov = [
        f"paths.data_dir={fake_comp}", "labels.sources=[fake_labels.csv]", "data.num_folds=2",
        "preprocess.res=64", "preprocess.num_workers=2", "preprocess.crop_mm=40",
        "preprocess.slots.sag_fs.n=6", "preprocess.slots.sag_t1.n=4", "preprocess.slots.cor_fs.n=4",
        "preprocess.slots.cor_t1.n=4", "preprocess.slots.ax_fs.n=4", "preprocess.slots.ax_t1.n=4",
        "model.backbone=resnet18", "model.pretrained=false", "model.img_size=64",
        "windows.k_train=4", "windows.k_eval=6", "training.batch_size=2", "training.num_epochs=2",
        "training.precision=32", "resources.num_workers=0", "resources.devices=1", "exp_name=smoke",
    ]
    monkeypatch.setattr(sys, "argv", ["preprocess"] + ov)
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import preprocess

    preprocess.main()
    assert len(glob.glob(str(fake_comp / "processed" / "cache_64" / "*.npz"))) == 14

    from src.models import train_pl

    monkeypatch.setattr(sys, "argv", ["train"] + ov + ["fold=0"])
    train_pl.main()
    assert os.path.exists("models/smoke/fold0.pt")
    assert os.path.exists("models/oof/smoke_oof_fold0.csv") and os.path.exists("models/oof/smoke_gold_fold0.csv")

    from src.models import inference

    monkeypatch.setattr(sys, "argv", [
        "infer", "--data_dir", str(fake_comp / "raw"), "--ckpts", "models/smoke/*.pt",
        "--workers", "2", "--out", "submission.csv",
    ])
    inference.main()
    sub = pd.read_csv("submission.csv")
    assert list(sub.columns) == [ID_COL] + LABELS and len(sub) == 3
    assert ((sub[LABELS] >= 0) & (sub[LABELS] <= 1)).all().all()
