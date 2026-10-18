# RSNA Knee: S6 px-cache shard builder (hlly34, ITER-14). Renders the training DICOMs (mounted here) once into the S6
# px-cache rows + SKM-TEA anchors: heliosli's StudySlices / AnchorBank code, run by nartaa's precompute path
# (submission/external/s6_train/build_px_shard.py). The output shard is unpacked on the training box by unpack_px.py.
# Parameters below are set by scripts/s6_cache_push.sh; generated, do not edit on kaggle.com.
SHARD, NSHARDS, LIMIT, PILOT, QUALITY = 0, 1, 50, True, 95
HOURS = 11.3                     # stop taking new studies after this (Kaggle 12 h limit); the shard stays consistent
import json, os, shutil, subprocess, sys, time
from pathlib import Path

T0 = time.time()
INPUT = Path("/kaggle/input")
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=False)
print("cpus", os.cpu_count(), flush=True)


def find(marker, max_depth=6):
    """Shallowest directory under /kaggle/input holding `marker`; never descends into the DICOM trees (~750k files)."""
    hits = []
    for d, dirs, files in os.walk(INPUT):
        depth = len(Path(d).relative_to(INPUT).parts)
        dirs[:] = [] if depth >= max_depth else [x for x in dirs if x not in ("train_series", "test_series", "cloud_code")]
        if marker in files:
            hits.append(Path(d))
    if not hits:
        raise RuntimeError("not attached: " + marker)
    return min(hits, key=lambda p: len(p.parts))


DATA = find("train_series.csv")
ORTHO = find("anatomy.pt")
S6_SRC = find("build_px_shard.py")
TMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
S6 = TMP / "s6"
shutil.copytree(S6_SRC, S6, dirs_exist_ok=True)
print("data", DATA, "| ortho", ORTHO, "| code", S6_SRC, flush=True)
env = dict(os.environ, S6_DATA=str(DATA), S6_ANATOMY=str(ORTHO / "anatomy.pt"), PYTHONDONTWRITEBYTECODE="1",
           HF_HUB_OFFLINE="1")
subprocess.run([sys.executable, str(S6 / "helios_sources.py"), "--dest", str(S6 / "heliosli"), "--ortho", str(ORTHO), "--wheels",
                "--stage", str(S6_SRC.parent / "heliosli_compact_stage.py")], check=True, env=env)
OUT = Path("/kaggle/working/shard%d" % SHARD)
cmd = [sys.executable, "-u", str(S6 / "build_px_shard.py"), "--shard", str(SHARD), "--nshards", str(NSHARDS), "--out", str(OUT),
       "--tmp", str(TMP / "s6_work"), "--workers", str(os.cpu_count()), "--quality", str(QUALITY),
       "--deadline", str(T0 + HOURS * 3600)]
if LIMIT:
    cmd += ["--limit", str(LIMIT)]
if PILOT:
    cmd += ["--pilot"]
subprocess.run(cmd, check=True, env=env)
s = json.load(open(OUT / "summary.json"))
print("PILOT_SUMMARY" if PILOT else "SHARD_SUMMARY", json.dumps({k: v for k, v in s.items() if k != "errors"}), flush=True)
du = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 2 ** 30
print("shard size %.2f GB | total %.0f s" % (du, time.time() - T0), flush=True)
