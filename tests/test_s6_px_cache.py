"""S6 px cache: build_px_shard.py's shard format -> unpack_px.py -> train_s6f0_all.Studies (--px mode) round trip."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

S6 = os.path.join(os.path.dirname(__file__), "..", "submission", "external", "s6_train")
sys.path.insert(0, S6)
cv2 = pytest.importorskip("cv2")

S = 64


def fake_study(rng, n_series):
    """px rows + meta + blocks shaped like px_cache.StudySlices.build (views [[view_id, start within study]])."""
    px, meta, blocks = [], [], []
    for sidx in range(n_series):
        plane = sidx % 3
        n, views = int(rng.integers(6, 12)), ([0, 1, 4, 5] if plane != 1 else [0, 2, 3])
        block = {"series": "s%d" % sidx, "plane": ["Sagittal", "Coronal", "Axial"][plane], "n_slices": n, "views": []}
        for v in views:
            block["views"].append([v, len(px)])
            for k in range(n):
                img = np.zeros((S, S), np.uint8)
                cv2.circle(img, (S // 2, S // 2), 8 + k, int(60 + 15 * v), -1)
                px.append(img)
                meta.append([sidx, sidx, plane, v, 2 * k / max(n - 1, 1) - 1, np.log1p(3.5), 1.0, 1, 0, k, n, sidx])
        blocks.append(block)
    return np.stack(px), np.array(meta, np.float32), blocks


def write_shard(d, studies, quality):
    from build_px_shard import encode
    os.makedirs(os.path.join(d, "rows"))
    with open(os.path.join(d, "manifest.jsonl"), "w") as man:
        for uid, (px, meta, blocks) in studies.items():
            key = uid.replace(".", "")[:24]
            arrs = {"meta": meta, "blocks": np.array(json.dumps({"blocks": blocks, "removed": 0}))}
            if quality:
                jpg, off = encode(px, quality)
                np.savez(os.path.join(d, "rows", key + ".npz"), jpg=jpg, off=off, **arrs)
            else:
                np.savez_compressed(os.path.join(d, "rows", key + ".npz"), px=px, **arrs)
            man.write(json.dumps({"uid": uid, "key": key, "error": None, "rows": len(px)}) + "\n")


@pytest.mark.parametrize("quality", [0, 95])
def test_shards_unpack_into_trainer_px_cache(tmp_path, quality):
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    studies = {"1.2.%d" % i: fake_study(rng, 3 + i) for i in range(4)}
    uids = list(studies)
    write_shard(str(tmp_path / "a"), {u: studies[u] for u in uids[::2]}, quality)
    write_shard(str(tmp_path / "b"), {u: studies[u] for u in uids[1::2]}, quality)
    order = tmp_path / "order.json"
    order.write_text(json.dumps(uids + ["1.2.missing"]))
    subprocess.run([sys.executable, os.path.join(S6, "unpack_px.py"), "--shards", str(tmp_path / "a"), str(tmp_path / "b"),
                    "--out", str(tmp_path / "px"), "--order", str(order), "--size", str(S), "--workers", "2"], check=True)
    index = json.load(open(tmp_path / "px" / "index.json"))
    assert index["order"] == uids and "1.2.missing" in index["errors"]
    px = np.load(tmp_path / "px" / "px.u8.npy")
    meta = np.load(tmp_path / "px" / "meta.f32.npy")
    for u in uids:
        start, n, n_series = index["done"][u]
        ref_px, ref_meta, ref_blocks = studies[u]
        assert n == len(ref_px) and n_series == len(ref_blocks)
        np.testing.assert_array_equal(meta[start:start + n], ref_meta)
        err = np.abs(px[start:start + n].astype(int) - ref_px.astype(int))
        assert err.max() == 0 if quality == 0 else err.mean() < 2.0
        for b, rb in zip(index["blocks"][u], ref_blocks):     # view starts shifted to global rows
            assert [st - start for _, st in b["views"]] == [st for _, st in rb["views"]]

    import train_s6f0_all as T                                  # the trainer's own --px dataset reads it
    ds = T.Studies(str(tmp_path / "px"), uids, 16, "train", 42, eval_stride=3, phys_mm=3.5)
    for i in range(len(uids)):
        img, m, _ = (np.asarray(x) for x in ds[i])
        assert img.dtype == np.uint8 and img.ndim == 4 and img.shape[1:] == (3, S, S) and 0 < len(img) <= 16
        assert m.shape == (len(img), 6) and np.isfinite(m).all()
    ev = T.Studies(str(tmp_path / "px"), uids, 16, "eval", 42, eval_stride=3, phys_mm=3.5)
    assert len(ev[0][0]) > 0
