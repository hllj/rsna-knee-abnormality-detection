"""S6F0 on-the-fly replacement for compact_train.Studies (no persistent pixel cache).

Per study, in a per-study tmpfs work dir (removed in `finally`):
  Heliosli's lean native.prepare (probe_infer.install_native(True) -- the same call compact_infer.CStudies._setup makes)
  -> px_cache.StudySlices(size, views, anchors=<precomputed S6F0/anchors>).build(uid)    (no geometry bank, no VNet at train time)
  -> the window logic of compact_train.Studies.__getitem__ (compact_train.py:127-154), verbatim, file index always 0.
Returns exactly what compact_train.Studies returns: (uint8 [W,K,S,S], float32 [W,6], i).  Equality with compact_train.Studies on a
px cache of the same rows, and with compact_infer.CStudies (anchors recomputed inline) at eval stride 1, is checked by
smoke/smoke_a_dataset.py (and was shown by probe_datapath/probe_parity.py).

Failure policy (train time): a study that raises returns None (logged to <failures> as one JSON line); collate_fly drops it and
reports the count; the trainer aborts the epoch above --max-fail-frac.  A missing anchor file is a failure, never the silent
field-centre fallback of px_cache.py:200-207."""
import contextlib, hashlib, io, json, os, shutil, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s6common as SC
SC.install_paths()


class FlyStudies(torch.utils.data.Dataset):
    def __init__(self, uids, n_win, mode, seed, eval_stride=3, train_sampling="uniform", cover_stride=3, slices=3, slice_jitter=0.0,
                 phys_mm=0.0, size=320, views="v4", anchors=SC.ANCHORS, tmp="/tmp/s6f0_work", data=SC.DATA, split=SC.SPLIT,
                 failures=None, safe=True):
        self.uids, self.n_win, self.mode, self.seed = list(uids), int(n_win), mode, int(seed)
        self.eval_stride, self.train_sampling, self.cover_stride = int(eval_stride), train_sampling, int(cover_stride)
        self.slices, self.slice_jitter, self.phys_mm = int(slices), float(slice_jitter), float(phys_mm)
        assert self.slices % 2 == 1 and self.slices >= 1
        self.size, self.views, self.anchors, self.tmp = int(size), str(views), str(anchors), str(tmp)
        self.data, self.split, self.failures, self.safe = str(data), split, failures, bool(safe)
        self.epoch = 0
        self._ready = False

    def __len__(self):
        return len(self.uids)

    def _setup(self):
        SC.install_paths()
        import cv2
        cv2.setNumThreads(1)
        import probe_infer as PI, px_cache, compact_train as CT
        self.PI, self.CT = PI, CT
        self.native = SC.lean_native()
        self.builder = px_cache.StudySlices(self.tmp, [], self.size, self.views, anchors=self.anchors)
        self._ready = True

    def render(self, uid):
        """lean native.prepare + StudySlices.build -> (px uint8 [N,S,S], meta float32 [N,12], blocks with file index 0)."""
        if not self._ready:
            self._setup()
        if self.views in ("v3k", "v4") and not os.path.exists(SC.anchor_path(uid, self.anchors)):
            raise FileNotFoundError("no precomputed anchor for " + uid)
        os.makedirs(self.tmp, exist_ok=True)
        work = Path(self.tmp) / ("f%d_%s" % (os.getpid(), hashlib.sha256(uid.encode()).hexdigest()[:16]))
        try:
            self.PI.OneStudySource.current = uid
            with contextlib.redirect_stdout(io.StringIO()):
                self.native.prepare(self.data, work, split=self.split, study_ids=[uid], source_directory=True)
            self.builder.cache = str(work)
            item = self.builder.build(uid)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        blocks = [{"n_slices": b["n_slices"], "views": [(0, v, st) for v, st in b["views"]]} for b in item["blocks"]]
        return item["px"].numpy(), item["meta"].numpy(), blocks

    def item(self, i, epoch):
        uid = self.uids[i]
        px, meta_all, blocks = self.render(uid)
        # ---- compact_train.Studies.__getitem__ (compact_train.py:132-154), unchanged apart from reading the rendered arrays
        rng = np.random.default_rng(self.seed + 1000003 * epoch + i) if self.mode == "train" else None
        cand = self.CT.window_rows(blocks, lambda f, r: meta_all[r], self.mode, self.slices, self.eval_stride,
                                   self.cover_stride, self.phys_mm, rng, self.slice_jitter, self.train_sampling)
        if self.mode == "train" and len(cand) > self.n_win:
            cand = [cand[j] for j in np.sort(rng.choice(len(cand), self.n_win, replace=False))]
        if not cand:
            raise ValueError("no usable window")
        rows = np.array(cand, np.int64)
        img = np.stack([px[rows[:, 2 + d]] for d in range(self.slices)], 1)
        m = meta_all[rows[:, 1]]
        meta = np.stack([m[:, 1], m[:, 3], m[:, 4], m[:, 5], m[:, 6], m[:, 7]], 1)   # slot, view, rel_pos, log_spacing, extent, fluid
        return torch.from_numpy(np.ascontiguousarray(img)), torch.from_numpy(meta.astype(np.float32)), i

    def __getitem__(self, i):
        if isinstance(i, tuple):                         # (index, epoch) -- lets persistent workers see the epoch (unused by default)
            i, epoch = i
        else:
            epoch = self.epoch
        if not self.safe:
            return self.item(i, epoch)
        t0 = time.time()
        try:
            return self.item(i, epoch)
        except Exception as exc:
            rec = {"when": SC.amman_now().isoformat(timespec="seconds"), "uid": self.uids[i], "i": int(i), "epoch": int(epoch),
                   "mode": self.mode, "pid": os.getpid(), "s": round(time.time() - t0, 1), "error": repr(exc)[:300],
                   "tb": traceback.format_exc()[-800:]}
            print("FLY_FAIL " + json.dumps({k: rec[k] for k in ("uid", "epoch", "error")}), flush=True)
            if self.failures:
                try:
                    with open(self.failures, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec) + "\n")
                except OSError:
                    pass
            return None


def collate_fly(batch):
    """compact_train.collate over the studies that rendered; returns (img, meta, mask, idx, n_failed); img is None when every
    study of the batch failed (the trainer skips that batch)."""
    import compact_train as CT
    ok = [b for b in batch if b is not None and len(b[0])]
    n_bad = len(batch) - len(ok)
    if not ok:
        return None, None, None, None, n_bad
    img, meta, mask, idx = CT.collate(ok)
    return img, meta, mask, idx, n_bad
