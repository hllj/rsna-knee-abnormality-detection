"""Parallel download of a Kaggle kernel's output files (kaggle kernels output fetches them one by one, ~6 s per file from a
far-away box). Resumable: a file already on disk with the server's size is skipped; summary.json is fetched last, so its
presence means the shard is complete (scripts/vast/iter14_s6.sh skips such shards).

  python scripts/vast/fetch_kernel_output.py hlly34/rsna-knee-s6-cache-0 data/s6_shards/0 [--threads 32]"""
import argparse, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest


def list_files(api, kernel):
    owner, slug = kernel.split("/")
    files, token = [], None
    with api.build_kaggle_client() as kaggle:
        while True:
            req = ApiListKernelSessionOutputRequest()
            req.user_name, req.kernel_slug, req.page_size = owner, slug, 200
            if token:
                req.page_token = token
            resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
            files += [(f.file_name, f.url) for f in resp.files or []]
            token = resp.next_page_token
            if not token:
                return files


def fetch(name, url, out, tries=5):
    path = os.path.join(out, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for t in range(tries):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                size = int(r.headers.get("Content-Length", -1))
                if size >= 0 and os.path.exists(path) and os.path.getsize(path) == size:
                    return 0
                tmp = path + ".part"
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 22):
                        fh.write(chunk)
                if size >= 0 and os.path.getsize(tmp) != size:
                    raise IOError("short read %s" % name)
                os.replace(tmp, path)
                return os.path.getsize(path)
        except Exception as e:
            if t == tries - 1:
                raise
            time.sleep(5 * (t + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel")
    ap.add_argument("out")
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    api = KaggleApi()
    api.authenticate()
    files = list_files(api, a.kernel)
    last = [f for f in files if f[0].endswith("summary.json")]
    rest = [f for f in files if f not in last]
    print("%s: %d files" % (a.kernel, len(files)), flush=True)
    t0, got, n = time.time(), 0, 0
    with ThreadPoolExecutor(a.threads) as ex:
        futs = [ex.submit(fetch, name, url, a.out) for name, url in rest]
        for f in as_completed(futs):
            got += f.result()
            n += 1
            if n % 200 == 0 or n == len(futs):
                el = time.time() - t0
                print("  %d/%d files, %.1f GB new, %.0f MB/s" % (n, len(futs), got / 2 ** 30, got / 2 ** 20 / max(el, 1e-6)), flush=True)
    for name, url in last:   # completion marker last
        fetch(name, url, a.out)
    print("done %s in %.0f s" % (a.kernel, time.time() - t0), flush=True)


if __name__ == "__main__":
    sys.exit(main())
