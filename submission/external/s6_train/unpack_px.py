"""hlly34: build_px_shard.py shards -> the px cache train_s6f0_all.py --px reads (heliosli's px_cache.py layout).

  px.u8.npy    [N, S, S] uint8     one row per (series, view, slice), studies in --order order
  meta.f32.npy [N, 12]
  index.json   order, done {uid: [start, n_rows, n_series]}, blocks {uid: [{..., views: [[view_id, global start]]}]}, cursor, ...

  python unpack_px.py --shards SHARD_DIR [SHARD_DIR ...] --out PX_DIR [--order data/pool_ok.json] [--workers 16]
Studies are written in parallel (each worker decodes one study's JPEG rows into the shared memmap)."""
import argparse, glob, json, os, sys, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

META_COLS = ["series_idx", "slot", "plane", "view_id", "rel_pos", "log_spacing", "extent_130", "fluid",
             "fatsat", "slice_order", "n_slices", "real_series_idx"]      # = px_cache.META_COLS


def _write(job):
    path, start, n, px_path, meta_path, N, S = job
    import cv2
    cv2.setNumThreads(1)
    z = np.load(path)
    if "jpg" in z:
        jpg, off = z["jpg"], z["off"]
        px = np.stack([cv2.imdecode(jpg[off[i]:off[i + 1]], cv2.IMREAD_GRAYSCALE) for i in range(n)]) if n else None
    else:
        px = z["px"]
    if n:
        assert px.shape == (n, S, S), (path, px.shape)
        pm = np.load(px_path, mmap_mode="r+")
        mm = np.load(meta_path, mmap_mode="r+")
        pm[start:start + n] = px
        mm[start:start + n] = z["meta"]
        pm.flush(); mm.flush()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--order", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "pool_ok.json"))
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    t0 = time.time()
    order = json.load(open(a.order))
    found = {}
    for d in a.shards:
        for line in open(os.path.join(d, "manifest.jsonl"), encoding="utf-8"):
            r = json.loads(line) if line.strip() else None
            if r and r.get("error") is None and os.path.exists(os.path.join(d, "rows", r["key"] + ".npz")):
                found[r["uid"]] = (os.path.join(d, "rows", r["key"] + ".npz"), int(r["rows"]))
    missing = [u for u in order if u not in found]
    print("studies in order %d, found %d, missing %d" % (len(order), len(found), len(missing)), flush=True)
    if missing:
        print("missing (first 5):", missing[:5], flush=True)
    os.makedirs(a.out, exist_ok=True)
    present = [u for u in order if u in found]
    N, S = sum(found[u][1] for u in present), a.size
    px_path, meta_path = os.path.join(a.out, "px.u8.npy"), os.path.join(a.out, "meta.f32.npy")
    np.lib.format.open_memmap(px_path, mode="w+", dtype=np.uint8, shape=(N, S, S)).flush()
    np.lib.format.open_memmap(meta_path, mode="w+", dtype=np.float32, shape=(N, len(META_COLS))).flush()
    print("rows %d -> px %.1f GB" % (N, N * S * S / 2 ** 30), flush=True)
    index = {"order": present, "n_upper": int(N), "done": {}, "blocks": {}, "cursor": int(N), "errors": {u: "missing in shards" for u in missing},
             "removed_series": 0, "meta_cols": META_COLS,
             "config": {"size": S, "local_mm": 100.0, "policy": "series_companded", "views": "v4", "anchors": "build_px_shard",
                        "view_recipe": "global_local+coronal_bilateral", "duplicates": "collapse_exact",
                        "rows": "one per (series, view, slice), physical order", "gold_first": 0, "source": "unpack_px.py"}}
    jobs, cur = [], 0
    for u in present:
        path, n = found[u]
        z = np.load(path)
        b = json.loads(str(z["blocks"]))
        blocks = b["blocks"]
        for blk in blocks:
            blk["views"] = [[v, cur + off] for v, off in blk["views"]]
        index["done"][u] = [cur, n, len(blocks)]
        index["blocks"][u] = blocks
        index["removed_series"] += int(b.get("removed", 0))
        jobs.append((path, cur, n, px_path, meta_path, N, S))
        cur += n
    with Pool(a.workers) as pool:
        for k, _ in enumerate(pool.imap_unordered(_write, jobs, chunksize=4), 1):
            if k % 250 == 0 or k == len(jobs):
                print("written %d/%d studies  %.0f s" % (k, len(jobs), time.time() - t0), flush=True)
    json.dump(index, open(os.path.join(a.out, "index.json"), "w", encoding="utf-8"))
    print("DONE %s: %d studies, %d rows, %.0f s" % (a.out, len(present), N, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
