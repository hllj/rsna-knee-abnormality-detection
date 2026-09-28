"""
Download the packed cache (notebook <user>/rsna-knee-cache-pack) shard by shard and unpack it.

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
import time


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
    a = ap.parse_args()
    kernel = f"{a.user}/rsna-knee-cache-pack"
    pack, dest = os.path.join(a.data, "_pack"), os.path.join(a.data, "processed")
    os.makedirs(pack, exist_ok=True)
    os.makedirs(dest, exist_ok=True)

    man_path = os.path.join(pack, "cache_manifest.json")
    if not os.path.exists(man_path):
        kaggle_get(kernel, r"^cache_manifest\.json$", pack)
    man = json.load(open(man_path))
    done_file = os.path.join(pack, "extracted.txt")
    done = set(open(done_file).read().split()) if os.path.exists(done_file) else set()

    for i, s in enumerate(man["shards"]):
        if s["file"] in done:
            continue
        path = os.path.join(pack, s["file"])
        for attempt in range(1, a.retries + 1):
            t = time.time()
            try:
                kaggle_get(kernel, "^" + re.escape(s["file"]) + "$", pack)
                if os.path.getsize(path) == s["bytes"] and sha256(path) == s["sha256"]:
                    break
                print(f"  {s['file']}: size/checksum mismatch (attempt {attempt})")
            except subprocess.CalledProcessError as e:
                print(f"  {s['file']}: download failed (attempt {attempt}): {e}")
            time.sleep(10 * attempt)
        else:
            raise SystemExit(f"could not fetch {s['file']}")
        with tarfile.open(path) as tar:
            tar.extractall(dest)
        os.remove(path)  # keep disk usage at ~1 shard + the extracted cache
        with open(done_file, "a") as f:
            f.write(s["file"] + "\n")
        print(f"[{i + 1}/{len(man['shards'])}] {s['file']} {s['bytes'] / 2**30:.2f} GiB in {time.time() - t:.0f}s")

    n = len([f for f in os.listdir(os.path.join(dest, "cache_256")) if f.endswith(".npz")])
    assert n == man["studies"], (n, man["studies"])
    print(f"cache ok: {n} studies in {os.path.join(dest, 'cache_256')}")


if __name__ == "__main__":
    main()
