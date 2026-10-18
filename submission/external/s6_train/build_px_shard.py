"""hlly34: one pass over the training DICOMs -> SKM-TEA anchors + the S6 px-cache rows of every study, as a portable shard.

Per study it runs exactly what precompute_anchors.Pre does (Heliosli's lean native.prepare -> geometry_bank.build ->
knee_anchor_bank_v2.AnchorBank.record -> px_cache.StudySlices(size, "v4", anchors).build), and keeps the rows instead of only
fingerprinting them.  Rows are what fly_data.render returns and what train_s6f0_all.py --px reads (px uint8 [N,S,S], meta [N,12]).

Output (--out):
  rows/<sha24(study)>.npz   jpg (uint8, concatenated JPEG rows) + off (int64 [N+1]) + meta (float32 [N,12]) + blocks (JSON str)
                            --quality 0 stores the rows lossless instead (px uint8 [N,S,S], np.savez_compressed)
  anchors/<sha24>.json      the anchor record (the file px_cache reads)
  manifest.jsonl            one line per study: uid, error, rows, bytes, s (+ pilot error stats)
  summary.json              totals and projections
unpack_px.py turns one or more shards into px.u8.npy / meta.f32.npy / index.json for --px.

  python build_px_shard.py --order data/pool_ok.json --shard 0 --nshards 2 --out OUT [--limit 50 --pilot] [--workers 4]"""
import argparse, contextlib, hashlib, io, json, os, shutil, sys, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s6common as SC
SC.install_paths()
import numpy as np, torch


def encode(px, quality):
    import cv2
    bufs = [cv2.imencode(".jpg", r, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])[1].ravel() for r in px]
    off = np.zeros(len(bufs) + 1, np.int64)
    off[1:] = np.cumsum([len(b) for b in bufs])
    return (np.concatenate(bufs) if bufs else np.zeros(0, np.uint8)), off


def decode(jpg, off):
    import cv2
    return np.stack([cv2.imdecode(jpg[off[i]:off[i + 1]], cv2.IMREAD_GRAYSCALE) for i in range(len(off) - 1)])


class Build(torch.utils.data.Dataset):
    def __init__(self, uids, tmp, size, quality, pilot):
        self.uids, self.tmp, self.size, self.quality, self.pilot, self.ok = uids, tmp, size, quality, pilot, False

    def __len__(self):
        return len(self.uids)

    def _setup(self):
        SC.install_paths()
        import cv2
        cv2.setNumThreads(1)
        torch.set_num_threads(1)
        import probe_infer as PI, knee_anchor_bank_v2 as KB, px_cache
        import training.geometry_bank as GB
        self.PI, self.KB, self.GB = PI, KB, GB
        self.native = SC.lean_native()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.bank = KB.AnchorBank(KB.load_model(SC.ANATOMY, dev), dev)
        self.builder = px_cache.StudySlices(self.tmp, [], self.size, "v4")
        self.ok = True

    def __getitem__(self, i):
        if not self.ok:
            self._setup()
        uid = self.uids[i]
        key = self.KB.study_key(uid)
        work = Path(self.tmp) / ("b%d_%s" % (os.getpid(), key[:16]))
        t0 = time.perf_counter()
        try:
            self.PI.OneStudySource.current = uid
            with contextlib.redirect_stdout(io.StringIO()):
                self.native.prepare(SC.DATA, work, split=SC.SPLIT, study_ids=[uid], source_directory=True)
                self.GB.build(root=SC.DATA, manifest=str(work / "manifest.json"), output=str(work / "geometry"), split=SC.SPLIT,
                              source_directory=True)
            with open(work / "studies" / (key + ".json"), encoding="utf-8") as fh:
                study = json.load(fh)
            rec = self.bank.record(uid, study, geo_dir=str(work / "geometry"))
            (work / "anchors").mkdir(exist_ok=True)
            with open(work / "anchors" / (key + ".json"), "w", encoding="utf-8") as fh:
                json.dump(rec, fh)
            self.builder.cache, self.builder.anchors = str(work), str(work / "anchors")
            item = self.builder.build(uid)
            px, meta = item["px"].numpy(), item["meta"].numpy().astype(np.float32)
            out = {"uid": uid, "key": key, "error": None, "anchor": rec, "meta": meta, "blocks": item["blocks"],
                   "removed": int(item.get("removed", 0)), "rows": int(len(meta)), "raw_bytes": int(px.nbytes),
                   "series": len(study["series"]), "series_errors": len(study["errors"]), "method": rec.get("method")}
            if self.quality:
                out["jpg"], out["off"] = encode(px, self.quality)
                out["bytes"] = int(out["jpg"].nbytes + out["off"].nbytes + meta.nbytes)
            else:
                out["px"] = px
            if self.pilot:                                   # lossless size and JPEG error, for the cache-format decision
                import zlib
                out["zlib_bytes"] = len(zlib.compress(np.ascontiguousarray(px).tobytes(), 6))
                if self.quality and len(px):
                    d = np.abs(decode(out["jpg"], out["off"]).astype(np.int16) - px.astype(np.int16))
                    out["jpg_mae"] = float(d.mean())
                    out["jpg_p999"] = float(np.percentile(d, 99.9))
                    out["jpg_max"] = int(d.max())
        except Exception as exc:
            out = {"uid": uid, "key": key, "error": repr(exc)[:300]}
        finally:
            shutil.rmtree(work, ignore_errors=True)
        out["s"] = round(time.perf_counter() - t0, 2)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", default=os.path.join(SC.DATA_DIR, "pool_ok.json"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", default="/tmp/s6_build_work")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality per row; 0 = lossless (np.savez_compressed)")
    ap.add_argument("--pilot", action="store_true", help="also measure lossless size and JPEG error per study")
    ap.add_argument("--deadline", type=float, default=0, help="unix time: stop taking new studies after it (resumable)")
    a = ap.parse_args()
    SC.verify_copies(anatomy=True)
    out = Path(a.out)
    (out / "rows").mkdir(parents=True, exist_ok=True)
    (out / "anchors").mkdir(exist_ok=True)
    shutil.rmtree(a.tmp, ignore_errors=True)
    os.makedirs(a.tmp, exist_ok=True)
    order = json.load(open(a.order))
    mine = order[a.shard::a.nshards]
    man_path = out / "manifest.jsonl"
    done = set()
    if man_path.exists():
        for line in open(man_path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("error") is None and (out / "rows" / (r["key"] + ".npz")).exists():
                done.add(r["uid"])
    todo = [u for u in mine if u not in done]
    if a.limit:
        todo = todo[:a.limit]
    print("shard %d/%d: %d studies, %d done, %d to do (workers %d, size %d, quality %d, cuda %s)" % (
        a.shard, a.nshards, len(mine), len(done), len(todo), a.workers, a.size, a.quality, torch.cuda.is_available()), flush=True)
    dl = torch.utils.data.DataLoader(Build(todo, a.tmp, a.size, a.quality, a.pilot), batch_size=None, num_workers=a.workers,
                                     prefetch_factor=2 if a.workers else None, multiprocessing_context="spawn" if a.workers else None,
                                     timeout=900 if a.workers else 0)
    t0, n, n_err = time.time(), 0, 0
    with open(man_path, "a", encoding="utf-8") as man:
        for r in dl:
            n += 1
            if r["error"] is None:
                SC.atomic_json(r["anchor"], str(out / "anchors" / (r["key"] + ".json")))
                arrs = {"meta": r["meta"], "blocks": np.array(json.dumps({"blocks": r["blocks"], "removed": r["removed"]}))}
                tmp = out / "rows" / (r["key"] + ".tmp.npz")
                if "jpg" in r:
                    np.savez(tmp, jpg=r["jpg"], off=r["off"], **arrs)
                else:
                    np.savez_compressed(tmp, px=r["px"], **arrs)
                    r["bytes"] = os.path.getsize(tmp)
                os.replace(tmp, out / "rows" / (r["key"] + ".npz"))
            else:
                n_err += 1
                print("ERROR", r["uid"], r["error"], flush=True)
            man.write(json.dumps({k: v for k, v in r.items() if k not in ("anchor", "meta", "blocks", "jpg", "off", "px")}) + "\n")
            man.flush()
            if n % 50 == 0 or n == len(todo) or n <= 2:
                el = time.time() - t0
                print("%d/%d  %.3f studies/s  errors %d  ETA %.0f min" % (n, len(todo), n / el, n_err, (len(todo) - n) / max(n / el, 1e-9) / 60),
                      flush=True)
            if a.deadline and time.time() > a.deadline:
                print("deadline reached after %d studies; rerun to resume" % n, flush=True)
                break
    summary(out, len(order), a)
    shutil.rmtree(a.tmp, ignore_errors=True)


def summary(out, n_order, a):
    rs = [json.loads(l) for l in open(out / "manifest.jsonl", encoding="utf-8") if l.strip()]
    ok = [r for r in rs if r.get("error") is None]
    m = lambda k: float(np.mean([r[k] for r in ok if k in r])) if any(k in r for r in ok) else None
    s = {"studies_ok": len(ok), "errors": {r["uid"]: r["error"] for r in rs if r.get("error")}, "rows_mean": m("rows"),
         "s_per_study_mean": m("s"), "bytes_mean": m("bytes"), "raw_bytes_mean": m("raw_bytes"), "zlib_bytes_mean": m("zlib_bytes"),
         "jpg_mae_mean": m("jpg_mae"), "jpg_p999_mean": m("jpg_p999"), "jpg_max": max([r["jpg_max"] for r in ok if "jpg_max" in r], default=None),
         "workers": a.workers, "quality": a.quality, "size": a.size}
    if ok:   # projections for all studies of the order (4,321 = 4,263 train + 58 gold)
        G = 2 ** 30
        s["projected_all"] = {"stored_GB": s["bytes_mean"] * n_order / G if s["bytes_mean"] else None,
                              "raw_px_GB": s["raw_bytes_mean"] * n_order / G,
                              "lossless_zlib_GB": s["zlib_bytes_mean"] * n_order / G if s["zlib_bytes_mean"] else None,
                              "rows": s["rows_mean"] * n_order}
    SC.atomic_json(s, str(out / "summary.json"), indent=1)
    print("SUMMARY", json.dumps({k: v for k, v in s.items() if k != "errors"} | {"n_errors": len(s["errors"])}), flush=True)


if __name__ == "__main__":
    main()
