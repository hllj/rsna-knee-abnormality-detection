"""S6F0 anchor/geometry precompute: one SKM-TEA knee anchor JSON per study, by Heliosli's own per-study path, exactly as
compact_infer.CStudies computes it inline at test (compact_infer.py:155-182 / 195-203):
    lean native.prepare (probe_infer.install_native(True)) -> training.geometry_bank.build -> knee_anchor_bank_v2.AnchorBank.record
    (VNet on the GPU in every spawn worker) -> json.dump(record)
plus one validation render (px_cache.StudySlices(size, "v4", anchors).build) that counts rows, stride-1 eval windows and train
candidates, and fingerprints the rows (sha1 of px + meta) so the training loader can be checked against it.

Outputs (all atomic: tmp + os.replace):
  anchors/<sha24(study)>.json   the record (the file px_cache reads; byte-for-byte what compact_infer writes into its work dir)
  anchors/manifest.jsonl        one line per finished study (appended AFTER its anchor file is in place): uid, error, rows,
                                eval1_windows, train_cands, series, series_errors, method, px_sha1, s
  anchors/summary.json          totals, written at the end
Resumable: a study with its anchor file AND a success line in the manifest is skipped; errors are retried on the next start.
Usage (systemd unit rsna-s6f0-anchors-20261010):  python -u precompute_anchors.py [--order data/precompute_order.json] [--workers 8]"""
import argparse, contextlib, hashlib, io, json, os, shutil, subprocess, sys, threading, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s6common as SC
SC.install_paths()
import numpy as np, torch


class Pre(torch.utils.data.Dataset):
    def __init__(self, uids, tmp, size):
        self.uids, self.tmp, self.size, self.ok = uids, tmp, size, False

    def __len__(self):
        return len(self.uids)

    def _setup(self):
        SC.install_paths()
        import cv2
        cv2.setNumThreads(1)
        torch.set_num_threads(1)
        import probe_infer as PI, knee_anchor_bank_v2 as KB, px_cache, compact_train as CT
        import training.geometry_bank as GB
        self.PI, self.KB, self.GB, self.CT = PI, KB, GB, CT
        self.native = SC.lean_native()
        self.bank = KB.AnchorBank(KB.load_model(SC.ANATOMY, "cuda"), "cuda")
        self.builder = px_cache.StudySlices(self.tmp, [], self.size, "v4")
        self.ok = True

    def __getitem__(self, i):
        if not self.ok:
            self._setup()
        uid = self.uids[i]
        key = self.KB.study_key(uid)
        work = Path(self.tmp) / ("a%d_%s" % (os.getpid(), key[:16]))
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
            px, meta = item["px"].numpy(), item["meta"].numpy()
            blocks = [{"n_slices": b["n_slices"], "views": [(0, v, st) for v, st in b["views"]]} for b in item["blocks"]]
            ev = self.CT.window_rows(blocks, lambda f, r: meta[r], "eval", 3, 1, phys_mm=3.5)
            tr = self.CT.window_rows(blocks, lambda f, r: meta[r], "train", 3, 3, 3, 3.5, np.random.default_rng(0), 0.0, "uniform")
            h = hashlib.sha1(np.ascontiguousarray(px).tobytes())
            h.update(np.ascontiguousarray(meta).tobytes())
            out = {"uid": uid, "error": None, "anchor": rec, "rows": int(len(meta)), "eval1_windows": len(ev), "train_cands": len(tr),
                   "series": len(study["series"]), "series_errors": len(study["errors"]), "method": rec.get("method"),
                   "px_sha1": h.hexdigest(), "views": sorted({int(v) for v in meta[:, 3]}) if len(meta) else []}
        except Exception as exc:
            out = {"uid": uid, "error": repr(exc)[:300]}
        finally:
            shutil.rmtree(work, ignore_errors=True)
        out["s"] = round(time.perf_counter() - t0, 2)
        return out


def read_manifest(path):
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:                      # a torn last line from a kill: ignore it (the study is redone)
                    continue
                done[r["uid"]] = r
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", default=os.path.join(SC.DATA_DIR, "precompute_order.json"))
    ap.add_argument("--out", default=SC.ANCHORS)
    ap.add_argument("--tmp", default="/tmp/s6f0_anchors_work")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    SC.verify_copies(anatomy=True)
    os.makedirs(a.out, exist_ok=True)
    shutil.rmtree(a.tmp, ignore_errors=True)
    os.makedirs(a.tmp, exist_ok=True)
    order = json.load(open(a.order))
    man_path = os.path.join(a.out, "manifest.jsonl")
    prev = read_manifest(man_path)
    ok_prev = {u for u, r in prev.items() if r.get("error") is None and os.path.exists(SC.anchor_path(u, a.out))}
    todo = [u for u in order if u not in ok_prev]
    if a.limit:
        todo = todo[:a.limit]
    print("%s anchors: %d studies in order, %d already done, %d to do (workers %d, size %d)" % (
        SC.amman(), len(order), len(ok_prev), len(todo), a.workers, a.size), flush=True)
    if not todo:
        write_summary(a, order, man_path)
        return
    gpu_peak = [0]
    stop = [False]

    def poll():
        while not stop[0]:
            try:
                v = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True,
                                       text=True, timeout=20).stdout.strip())
                gpu_peak[0] = max(gpu_peak[0], v)
            except Exception:
                pass
            time.sleep(5)
    threading.Thread(target=poll, daemon=True).start()
    dl = torch.utils.data.DataLoader(Pre(todo, a.tmp, a.size), batch_size=None, num_workers=a.workers, prefetch_factor=1,
                                     multiprocessing_context="spawn", timeout=900)
    t0 = time.time()
    n = n_err = 0
    with open(man_path, "a", encoding="utf-8") as man:
        for r in dl:
            n += 1
            if r["error"] is None:
                SC.atomic_json(r["anchor"], SC.anchor_path(r["uid"], a.out))
            else:
                n_err += 1
                print("ERROR", r["uid"], r["error"], flush=True)
            man.write(json.dumps({k: v for k, v in r.items() if k != "anchor"}) + "\n")
            man.flush()
            if n % 100 == 0 or n == len(todo) or n <= 2:
                el = time.time() - t0
                rate = n / max(el, 1e-6)
                eta = time.time() + (len(todo) - n) / max(rate, 1e-6)
                print("%s anchors %d/%d  %.2f studies/s  errors %d  GPU peak %d MiB  ETA %s" % (
                    SC.amman(), n, len(todo), rate, n_err, gpu_peak[0], SC.amman(eta)), flush=True)
    stop[0] = True
    write_summary(a, order, man_path)
    shutil.rmtree(a.tmp, ignore_errors=True)


def write_summary(a, order, man_path):
    man = read_manifest(man_path)
    rs = [man[u] for u in order if u in man]
    ok = [r for r in rs if r.get("error") is None and os.path.exists(SC.anchor_path(r["uid"], a.out))]
    meth = {}
    for r in ok:
        meth[r["method"]] = meth.get(r["method"], 0) + 1
    summ = {"when": SC.amman_now().isoformat(timespec="seconds"), "order": len(order), "finished": len(rs), "ok": len(ok),
            "errors": {r["uid"]: r["error"] for r in rs if r.get("error") is not None},
            "missing": len([u for u in order if u not in man]), "methods": meth,
            "rows_mean": round(float(np.mean([r["rows"] for r in ok])), 1) if ok else None,
            "eval1_windows_mean": round(float(np.mean([r["eval1_windows"] for r in ok])), 1) if ok else None,
            "eval1_windows_sum": int(sum(r["eval1_windows"] for r in ok)),
            "train_cands_min": int(min(r["train_cands"] for r in ok)) if ok else None,
            "train_cands_lt64": sum(r["train_cands"] < 64 for r in ok),
            "per_study_s_mean": round(float(np.mean([r["s"] for r in ok])), 2) if ok else None}
    SC.atomic_json(summ, os.path.join(a.out, "summary.json"), indent=1)
    print("%s anchors SUMMARY %s" % (SC.amman(), json.dumps({k: v for k, v in summ.items() if k != "errors"} | {"n_errors": len(summ["errors"])})),
          flush=True)


if __name__ == "__main__":
    main()
