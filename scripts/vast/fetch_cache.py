"""
Download a packed cache (notebook <user>/<pack>, default rsna-knee-cache-pack) shard by shard and unpack it.
The cache directory name comes from the manifest (cache_256, cache_256_dense, ...).

Fetching the whole notebook output in one `kaggle kernels output` call died silently on a vast.ai box
(14 x 1.1 GB); one shard per call, each checksum-verified and retried, is reliable and resumable.

    python scripts/vast/fetch_cache.py --data data --user hlly34
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor


def kaggle_get(kernel: str, pattern: str, out: str) -> None:
    subprocess.run(["kaggle", "kernels", "output", kernel, "-p", out, "--file-pattern", pattern],
                   check=True, stdout=subprocess.DEVNULL)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--user", required=True)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--pack", default="rsna-knee-cache-pack", help="pack notebook, e.g. rsna-knee-cache-dense-pack")
    ap.add_argument("--parallel", type=int, default=4,
                    help="shards downloaded at once (one request per shard; slow links benefit a lot)")
    a = ap.parse_args()
    kernel = f"{a.user}/{a.pack}"
    pack, dest = os.path.join(a.data, f"_pack_{a.pack}"), os.path.join(a.data, "processed")
    os.makedirs(pack, exist_ok=True)
    os.makedirs(dest, exist_ok=True)

    man_path = os.path.join(pack, "cache_manifest.json")
    if not os.path.exists(man_path):
        kaggle_get(kernel, r"^cache_manifest\.json$", pack)
    man = json.load(open(man_path))
    done_file = os.path.join(pack, "extracted.txt")
    done = set(open(done_file).read().split()) if os.path.exists(done_file) else set()

    lock = threading.Lock()

    def fetch(i_s):
        i, sh = i_s
        path = os.path.join(pack, sh["file"])
        t = time.time()
        for attempt in range(1, a.retries + 1):
            try:
                kaggle_get(kernel, "^" + re.escape(sh["file"]) + "$", pack)
                if os.path.getsize(path) == sh["bytes"] and sha256(path) == sh["sha256"]:
                    break
                print(f"  {sh['file']}: size/checksum mismatch (attempt {attempt})", flush=True)
            except subprocess.CalledProcessError as e:
                print(f"  {sh['file']}: download failed (attempt {attempt}): {e}", flush=True)
            time.sleep(10 * attempt)
        else:
            raise SystemExit(f"could not fetch {sh['file']}")
        with tarfile.open(path) as tar:
            tar.extractall(dest)
        os.remove(path)  # keep disk usage bounded
        with lock:
            with open(done_file, "a") as f:
                f.write(sh["file"] + "\n")
            print(f"[{i + 1}/{len(man['shards'])}] {sh['file']} {sh['bytes'] / 2**30:.2f} GiB in {time.time() - t:.0f}s",
                  flush=True)

    todo = [(i, sh) for i, sh in enumerate(man["shards"]) if sh["file"] not in done]
    with ThreadPoolExecutor(max(1, a.parallel)) as ex:
        list(ex.map(fetch, todo))

    name = man.get("name", "cache_256")
    n = len([f for f in os.listdir(os.path.join(dest, name)) if f.endswith(".npz")])
    assert n >= man["studies"], (n, man["studies"])  # >=: several part notebooks fill the same cache dir
    print(f"cache ok: {n} studies in {os.path.join(dest, name)}")


if __name__ == "__main__":
    main()
