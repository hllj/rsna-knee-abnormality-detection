"""
Generate every Kaggle notebook of the pipeline (and its kernel-metadata.json) from one place.

    python submission/build_kernels.py            # writes submission/<kernel>/...
    kaggle kernels push -p submission/<kernel>    # push one of them

Pipeline (each box is one Kaggle notebook; arrows are "attached as input"):

    rsna-knee-code (dataset: src/ configs/ scripts/)
            │
    rsna-knee-cache (CPU) ──► rsna-knee-train-teacher (GPU, 5 folds) ──► rsna-knee-train-student (GPU, 5 folds,
            │                         │                                     pseudo-labels from teacher OOF)
            │                         └──────────────┬──────────────────────────┘
            │                                        ▼
            └──────────────────────────────► rsna-knee-inference (GPU, internet OFF) ──► submission.csv

Experiment ids (EXP-NNN) are set in EXPERIMENTS below; each notebook prints its id and writes
experiment.json so runs can be registered with scripts/track.py.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
USER = "hlly34"
COMP = "rsna-knee-abnormality-detection"
LABEL_DATASETS = ["stevenleehans/rsna-knee-llm-report-labels", "pilkwang/rsna-knee-llm-labels"]

# One entry per notebook version you intend to run. Bump the id when anything that changes results changes.
EXPERIMENTS = {
    "dryrun": {"id": "EXP-000", "what": "subset sanity run: 58 gold + 600 studies, EfficientNet-B0 fold 0, 8 epochs"},
    "cache": {"id": "EXP-001", "what": "full training cache: 4,407 studies, 256 px, 72 slices, 140 mm, canonical knee"},
    "cache-pack": {"id": "EXP-001", "what": "EXP-001 cache re-packed into ~1 GB tar shards so it can be downloaded "
                   "off Kaggle (vast.ai) in a few requests instead of 4,407"},
    # ITER-04: denser-slice cache (88 slices/study; more sagittal/coronal slices for menisci)
    "cache-dense": {"id": "EXP-023", "what": "dense training cache: 4,407 studies, 256 px, 88 slices (sag-FS 26, sag 14, "
                    "cor-FS 20, cor 10, ax-FS 14, ax 4), 140 mm, canonical knee", "cfg": "convnext_288_dense",
                    "cache_name": "cache_256_dense"},
    "cache-dense-pack": {"id": "EXP-023", "what": "EXP-023 dense cache re-packed into ~1 GB tar shards for vast.ai",
                         "cache_name": "cache_256_dense"},
    **{f"cache-raptor-p{i}": {"id": "EXP-043", "what": f"Raptor-repro cache part {i + 1}/3: 80 slices (22/18/15/10/15) over "
                              "2-98%, 140 mm @336 px, 2-98 pct, no canonicalisation", "cfg": "raptor_repro",
                              "cache_name": "cache_336_raptor", "part": i, "parts": 3} for i in range(3)},
    **{f"cache-r96-p{i}": {"id": "EXP-066", "what": f"96-slice native-384 cache part {i + 1}/5 (nartaa geometry: "
                           "26/22/18/12/18 over 2-98%, 140 mm @384 px, 2-98 pct, no canonicalisation)",
                           "cfg": "raptor_v2_d96_alldata", "cache_name": "cache_384_d96", "part": i, "parts": 5}
       for i in range(5)},
    **{f"cache-reader-p{i}": {"id": "EXP-044", "what": f"reader-repro cache part {i + 1}/4: every series, native slices "
                              "<= 64, LPS in-plane frame, 0.55 mm/px @280, 1-99.5 pct", "cfg": "reader_repro",
                              "cache_name": "cache_280_reader", "part": i, "parts": 4} for i in range(4)},
    "train-teacher": {"id": "EXP-002", "what": "teacher: EfficientNet-B0 @224, FOLD 0 ONLY x 15 epochs, blended report "
                      "labels (check one fold before spending GPU on all five; folds 1-4 = EXP-005)",
                      "exp": "effnet_224", "epochs": 15, "folds": [0], "extra": []},
    "train-student": {"id": "EXP-003", "what": "student: same model, targets = 50/50 report labels + teacher OOF "
                      "(Synovitis 70%, PF/Lateral OA 60%)", "exp": "teacher_pseudo", "epochs": 15,
                      "folds": [0, 1, 2, 3, 4], "extra": [], "teacher": "train-teacher"},
    "bench": {"id": "EXP-011", "what": "inference speed-up benchmark: old cache path vs streaming (threaded/raw "
              "DICOM reads, CPU/GPU overlap, one pass for all checkpoints) vs selective decode; EXP-009 weights"},
    # ITER-02 fold-0 checks (docs/WORKFLOW.md step 1); bar = EXP-009 fold 0: OOF 0.865, gold 0.876
    "iter02-a": {"id": "EXP-013/014", "what": "ITER-02 fold 0: EfficientNet-B0 @288 (EXP-013, GPU0) and ConvNeXt-S @288 "
                 "(EXP-014, GPU1), both upsampled from the 256 cache",
                 # v1 OOM on the 14.5 GB T4 (bs16/bs8 x 16 windows at 288): same effective batch via accumulation
                 "jobs": [{"exp": "EXP-013", "gpu": 0, "train": ["effnet_288", 0, [
                              "training.batch_size=8", "training.accumulate_grad_batches=2"]]},
                          {"exp": "EXP-014", "gpu": 1, "train": ["convnext_288", 0, [
                              "training.batch_size=4", "training.accumulate_grad_batches=2",
                              "model.grad_checkpointing=true"]]}]},
    "iter02-b": {"id": "EXP-015/016", "what": "ITER-02 fold 0: CoAtNet-RMLP-2 @384 (EXP-015, GPU0, 10 epochs, bs 4, "
                 "val every 2, 9 h cap) and EXP-009 fold 0 re-scored with 64 vs 48 eval windows (EXP-016, GPU1)",
                 "jobs": [{"exp": "EXP-015", "gpu": 0, "train": ["coatnet_384", 0, [
                              "training.num_epochs=10", "training.batch_size=4", "training.val_every=2",
                              "training.max_time=00:09:00:00"]]},
                          {"exp": "EXP-016", "gpu": 1, "eval": ["effnet_b0_224/fold0.pt", [48, 64]]}]},
    "inference": {"id": "EXP-004", "what": "rank-average ensemble of all teacher + student folds",
                  "sources": ["train-teacher", "train-student"]},
}

# ---------------------------------------------------------------------------------------------------------
# shared code cells
FIND = '''import glob, json, os, subprocess, sys, time

def walk_find(pred, root="/kaggle/input", max_depth=7):
    """Depth-limited search of the attached inputs that never descends into the DICOM trees."""
    root = root.rstrip("/")
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        dirnames[:] = [] if depth >= max_depth else sorted(d for d in dirnames if not d.endswith("_series"))
        hit = pred(dirpath, filenames)
        if hit:
            return hit
    return None

# Kaggle mounts inputs at /kaggle/input/{competitions,datasets,notebooks}/... - locate them instead of hardcoding
CODE = walk_find(lambda d, f: d if os.path.isfile(os.path.join(d, "src", "models", "inference.py")) else None)
COMP = walk_find(lambda d, f: d if "test.csv" in f and os.path.isdir(os.path.join(d, "test_series")) else None)
assert CODE and COMP, (CODE, COMP)
sys.path.insert(0, CODE)
print("code:", CODE, "| competition:", COMP)
'''

SETUP_DATA = '''DATA = "/kaggle/tmp/data"            # scratch space (not saved); outputs go to /kaggle/working
os.makedirs(f"{DATA}/labels", exist_ok=True)
if not os.path.exists(f"{DATA}/raw"):
    os.symlink(COMP, f"{DATA}/raw")
for name in ["llm_labels_v4_blend.csv", "llm_labels_full.csv", "report_labels_v2.csv"]:
    src = walk_find(lambda d, f: os.path.join(d, name) if name in f else None)
    print(f"{name:28s} <- {src}")
    if src and not os.path.exists(f"{DATA}/labels/{name}"):
        os.symlink(src, f"{DATA}/labels/{name}")
ENV = dict(os.environ, DATA_DIR=DATA, PYTHONPATH=CODE)
os.chdir("/kaggle/working")

def run(*args, env=None, log=None):
    """Run a repo script/module as a subprocess (fresh CUDA context, clean logs)."""
    t = time.time()
    out = open(log, "w") if log else None
    subprocess.run([sys.executable, *args], env=env or ENV, check=True, stdout=out, stderr=subprocess.STDOUT if out else None)
    print(f"[{' '.join(map(str, args[:2]))}] {time.time() - t:.0f}s")
'''

PIP = '''subprocess.run([sys.executable, "-m", "pip", "install", "-q", "iterative-stratification", "lightning",
                "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], check=True)
import torch, timm, pydicom
print("torch", torch.__version__, "| timm", timm.__version__, "| pydicom", pydicom.__version__,
      "| GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], "| CPUs:", os.cpu_count())'''


def header(key: str, title: str, body: str) -> str:
    e = EXPERIMENTS[key]
    return (f"# {title}\n\n"
            f"| Experiment | Notebook | What this version does |\n|---|---|---|\n"
            f"| **{e['id']}** | `{USER}/rsna-knee-{key}` | {e['what']} |\n\n"
            f"{body}\n\n"
            "Pipeline code lives in the attached dataset `rsna-knee-code` (repo `src/`, `configs/`, `scripts/`); "
            "this notebook only orchestrates it. Results are registered with `scripts/track.py` into "
            "`experiments/registry.csv` using the id above.")


def exp_json(key: str) -> str:
    e = EXPERIMENTS[key]
    return (f'EXP = {json.dumps({"id": e["id"], "notebook": f"{USER}/rsna-knee-{key}", "what": e["what"]})}\n'
            'print("EXPERIMENT", EXP["id"], "-", EXP["what"])')


# ---------------------------------------------------------------------------------------------------------
def nb_dryrun():
    return [
        ("md", header("dryrun", "RSNA Knee — dry run on a subset",
                      "Checks the whole pipeline on real DICOMs quickly: build the cache for the 58 gold studies + 600 "
                      "random ones, train one fold, score it against the radiologist labels, and run test inference "
                      "exactly as the submission notebook does.")),
        ("code", exp_json("dryrun") + "\nLIMIT, EPOCHS, CFG = 600, 8, 'effnet_224'\n" + FIND),
        ("md", "## 1. Environment\nPackages missing from the Kaggle image are installed (internet is on here). The "
               "pylibjpeg wheels are also saved to the output so the offline inference notebook can install them."),
        ("code", PIP + '''
os.makedirs("/kaggle/working/wheels", exist_ok=True)
subprocess.run([sys.executable, "-m", "pip", "download", "-q", "-d", "/kaggle/working/wheels",
                "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], check=False)'''),
        ("md", "## 2. Labels\nPublic report-derived label sets are scored against the 58 gold studies, then blended "
               "(`configs/default.yaml → labels`)."),
        ("code", SETUP_DATA + '\nrun(f"{CODE}/scripts/eval_labels.py")'),
        ("md", "## 3. DICOM → cached stacks\nFor each study: 6 plane/sequence slots, slices sorted by position, "
               "knee side from geometry (left knees mirrored), 140 mm crop, 256 px, 72 slices → one `.npz`."),
        ("code", '''os.makedirs(f"{DATA}/processed", exist_ok=True)
run(f"{CODE}/scripts/preprocess.py", f"exp={CFG}", f"preprocess.limit={LIMIT}", "preprocess.num_workers=4")
import numpy as np
files = sorted(glob.glob(f"{DATA}/processed/cache_256/*.npz"))
sides = [int(np.load(f)["side"]) for f in files]
print(f"cached {len(files)} | knee R/L/unknown = {sides.count(1)}/{sides.count(-1)}/{sides.count(0)}")'''),
        ("code", '''import matplotlib.pyplot as plt
d = {s: next(f for f, x in zip(files, sides) if x == s) for s in (1, -1) if s in sides}
fig, ax = plt.subplots(len(d), 6, figsize=(18, 3 * len(d)), squeeze=False)
for r, (s, f) in enumerate(d.items()):
    z = np.load(f)
    for c in range(6):
        idx = np.where(z["slot"] == c)[0]
        ax[r, c].imshow(z["image"][idx[len(idx) // 2]], cmap="gray"); ax[r, c].axis("off")
        ax[r, c].set_title(f"{'R' if s == 1 else 'L'} knee slot {c}")
plt.tight_layout(); plt.savefig("/kaggle/working/canonical_examples.png", dpi=60); plt.show()'''),
        ("md", "## 4. Train fold 0 and score against gold"),
        ("code", '''run("-m", "src.models.train_pl", f"exp={CFG}", "fold=0", f"training.num_epochs={EPOCHS}",
    "resources.num_workers=4", "training.batch_size=16")
print(open(glob.glob("models/*/fold0_metrics.json")[0]).read())'''),
        ("md", "## 5. Test inference (same code path as the submission notebook)"),
        ("code", '''run("-m", "src.models.inference", "--data_dir", COMP, "--ckpts", "models/**/fold*.pt",
    "--cache_dir", "/kaggle/tmp/test_cache", "--workers", "4", "--out", "/kaggle/working/submission.csv")
json.dump(EXP, open("/kaggle/working/experiment.json", "w"))'''),
    ]


def nb_cache(key: str = "cache", cfg: str = "effnet_224", name: str = "cache_256"):
    return [
        ("md", header(key, "RSNA Knee — build the full training cache (CPU)",
                      "Decoding ~570 GB of DICOM is CPU work, so it runs here without spending GPU quota. The output "
                      f"(`{name}/<StudyInstanceUID>.npz`, compressed to fit Kaggle's 20 GB output limit) is attached "
                      f"to the training notebooks / packed for vast.ai. Slot layout from `configs/exp/{cfg}.yaml`.")),
        ("code", exp_json(key) + f"\nCFG, NAME = {cfg!r}, {name!r}\n" + FIND),
        ("md", "## What happens to each study\n"
               "1. **Pick series** — 6 slots: sagittal/coronal/axial × fluid-sensitive/not (same-plane fallback).\n"
               "2. **Order slices** by `ImagePositionPatient` along the slice normal (file names are random).\n"
               "3. **Canonical knee** — side from the image-centre x coordinate (tag as fallback); left knees are "
               "mirrored, sagittal runs lateral → medial.\n"
               "4. **Intensity** — per-series 1–99th percentile → uint8.\n"
               "5. **Geometry** — 140 mm physical centre crop → 256×256; slices sampled evenly over 4–96% of each "
               "series, per-slot counts from the config (`preprocess.slots`).\n"
               "6. **Save** `image (D,256,256)`, `slot`, `valid`, `side`."),
        ("code", '''subprocess.run([sys.executable, "-m", "pip", "install", "-q", "iterative-stratification",
                "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], check=True)
DATA = "/kaggle/tmp/data"; os.makedirs(DATA, exist_ok=True)
if not os.path.exists(f"{DATA}/raw"):
    os.symlink(COMP, f"{DATA}/raw")
ENV = dict(os.environ, DATA_DIR=DATA, PYTHONPATH=CODE)
t = time.time()
subprocess.run([sys.executable, f"{CODE}/scripts/preprocess.py", f"exp={CFG}", "preprocess.compress=true",
                f"preprocess.num_workers={os.cpu_count()}", f"paths.cache_dir=/kaggle/working/{NAME}"],
               env=ENV, check=True)
print(f"preprocess {(time.time() - t) / 60:.1f} min")'''),
        ("md", "## Sanity checks"),
        ("code", '''import numpy as np
files = sorted(glob.glob(f"/kaggle/working/{NAME}/*.npz"))
sides, cover = [], []
for f in files[::10]:
    z = np.load(f); sides.append(int(z["side"])); cover.append(z["valid"].mean())
size = sum(os.path.getsize(f) for f in files) / 2**30
EXP.update(studies=len(files), size_gib=round(size, 2), side_R=sides.count(1), side_L=sides.count(-1),
           side_unknown=sides.count(0), slot_coverage=round(float(np.mean(cover)), 4),
           slices_per_study=int(np.load(files[0])["image"].shape[0]))
print(EXP)
json.dump(EXP, open("/kaggle/working/experiment.json", "w"), indent=1)'''),
    ]


def nb_cache_part(key: str, cfg: str, name: str, part: int, parts: int, steps: str):
    """One shard (studies[part::parts]) of a cache too big for one notebook's 20 GB output, written straight
    as ~1 GiB tar shards + cache_manifest.json (the format scripts/vast/fetch_cache.py downloads)."""
    return [
        ("md", header(key, f"RSNA Knee — {name} cache, part {part + 1}/{parts} (CPU)",
                      f"Builds studies `[{part}::{parts}]` of `{name}` with `configs/exp/{cfg}.yaml` and packs them "
                      "into tar shards for vast.ai (`CACHE_PACKS=... bash scripts/vast/get_data.sh`).\n\n" + steps)),
        ("code", exp_json(key) + f"\nCFG, NAME, PART, PARTS = {cfg!r}, {name!r}, {part}, {parts}\n" + FIND),
        ("code", '''subprocess.run([sys.executable, "-m", "pip", "install", "-q", "iterative-stratification",
                "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], check=True)
DATA = "/kaggle/tmp/data"; os.makedirs(DATA, exist_ok=True)
if not os.path.exists(f"{DATA}/raw"):
    os.symlink(COMP, f"{DATA}/raw")
OUT = f"/kaggle/tmp/{NAME}"
ENV = dict(os.environ, DATA_DIR=DATA, PYTHONPATH=CODE)
t = time.time()
subprocess.run([sys.executable, f"{CODE}/scripts/preprocess.py", f"exp={CFG}", "preprocess.compress=true",
                f"preprocess.part={PART}", f"preprocess.parts={PARTS}",
                f"preprocess.num_workers={os.cpu_count()}", f"paths.cache_dir={OUT}"], env=ENV, check=True)
print(f"preprocess {(time.time() - t) / 60:.1f} min")'''),
        ("md", "## Sanity checks, then pack"),
        ("code", '''import hashlib, tarfile
import numpy as np
files = sorted(f for f in os.listdir(OUT) if f.endswith(".npz"))
slices, cover = [], []
for f in files[::25]:
    z = np.load(os.path.join(OUT, f)); slices.append(z["image"].shape[0]); cover.append(float(z["valid"].mean()))
total = sum(os.path.getsize(os.path.join(OUT, f)) for f in files)
n_shards = max(1, round(total / 2**30))
manifest = {"name": NAME, "studies": len(files), "bytes": total, "shards": []}
for k in range(n_shards):
    shard = f"{NAME}_p{PART}_{k:02d}.tar"
    with tarfile.open(f"/kaggle/working/{shard}", "w") as tar:
        for f in files[k::n_shards]:
            tar.add(os.path.join(OUT, f), arcname=f"{NAME}/{f}")
    for f in files[k::n_shards]:
        os.remove(os.path.join(OUT, f))
    h = hashlib.sha256()
    with open(f"/kaggle/working/{shard}", "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    manifest["shards"].append({"file": shard, "sha256": h.hexdigest(), "bytes": os.path.getsize(f"/kaggle/working/{shard}")})
json.dump(manifest, open("/kaggle/working/cache_manifest.json", "w"), indent=1)
EXP.update(studies=len(files), size_gib=round(total / 2**30, 2), shards=n_shards, part=PART, parts=PARTS,
           slices_per_study_mean=round(float(np.mean(slices)), 1), slices_per_study_max=int(np.max(slices)),
           slot_coverage=round(float(np.mean(cover)), 4))
json.dump(EXP, open("/kaggle/working/experiment.json", "w"), indent=1)
print(EXP)'''),
    ]


def nb_train(key: str):
    e = EXPERIMENTS[key]
    teacher = e.get("teacher")
    body = ("Trains every fold of one config. Folds come from the committed `configs/folds_5.csv` (same split in "
            "every round; the 58 gold studies are never trained on). Each fold saves a top-3-epoch weight-averaged "
            "model, its out-of-fold predictions and gold predictions; the fold predictions are then merged into a "
            "CV summary. With 2 GPUs two folds train at once.")
    if teacher:
        body += (f"\n\n**Pseudo-labels:** targets are `(1-w)·report + w·teacher_OOF` using the out-of-fold predictions "
                 f"of `{USER}/rsna-knee-{teacher}` ({EXPERIMENTS[teacher]['id']}). Each training study's teacher "
                 "prediction comes from the fold model that did not see it.")
    cells = [
        ("md", header(key, f"RSNA Knee — train {e['exp']} ({'student' if teacher else 'teacher'})", body)),
        ("code", exp_json(key) + f"\nCFG, EPOCHS, FOLDS, EXTRA = {e['exp']!r}, {e['epochs']}, {e['folds']}, {e['extra']}\n"
         + FIND),
        ("md", "## 1. Environment and inputs"),
        ("code", PIP + "\n" + SETUP_DATA + '''
CACHE = walk_find(lambda d, f: d if os.path.basename(d) == "cache_256" and any(x.endswith(".npz") for x in f[:50]) else None)
print("cache:", CACHE, len(os.listdir(CACHE)), "studies")
os.makedirs("/kaggle/working/wheels", exist_ok=True)   # offline wheels for the inference notebook
subprocess.run([sys.executable, "-m", "pip", "download", "-q", "-d", "/kaggle/working/wheels",
                "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], check=False)
EXTRA = EXTRA + [f"paths.cache_dir={CACHE}"]'''),
    ]
    if teacher:
        cells += [
            ("md", "## 2. Teacher out-of-fold predictions → pseudo-labels"),
            ("code", '''CV_JSON = walk_find(lambda d, f: next((os.path.join(d, x) for x in f if x.endswith("_cv.json")), None))
TEACHER_OOF = CV_JSON.replace("_cv.json", ".csv")
print("teacher:", json.load(open(CV_JSON)))
print("teacher OOF:", TEACHER_OOF)
EXTRA = EXTRA + [f"labels.pseudo.oof_files=[{TEACHER_OOF}]"]
run(f"{CODE}/scripts/eval_labels.py", f"exp={CFG}", *EXTRA)   # last line = targets incl. pseudo-labels vs gold'''),
        ]
    cells += [
        ("md", f"## {3 if teacher else 2}. Train all folds\nEach fold is a separate process (logs in "
               "`/kaggle/working/logs/fold*.txt`); folds are distributed over the available GPUs."),
        ("code", '''import torch
n_gpu = max(1, torch.cuda.device_count())
workers = max(2, os.cpu_count() // n_gpu)
import psutil
queue, running, failed = list(FOLDS), [], []
os.makedirs("logs", exist_ok=True)
t0 = last_mem = time.time()
while queue or running:
    if time.time() - last_mem > 600:   # RAM watchdog: EXP-002 v1 was OOM-killed (exit -9) after 107 min
        vm = psutil.virtual_memory(); last_mem = time.time()
        print(f"{(time.time() - t0) / 60:6.1f} min  RAM used {vm.used / 2**30:.1f}/{vm.total / 2**30:.1f} GiB")
    while queue and len(running) < n_gpu:
        fold = queue.pop(0)
        gpu = [g for g in range(n_gpu) if g not in [r[2] for r in running]][0]
        env = dict(ENV, CUDA_VISIBLE_DEVICES=str(gpu))
        cmd = [sys.executable, "-m", "src.models.train_pl", f"exp={CFG}", f"fold={fold}",
               f"training.num_epochs={EPOCHS}", f"resources.num_workers={workers}", *EXTRA]
        running.append((subprocess.Popen(cmd, env=env, stdout=open(f"logs/fold{fold}.txt", "w"),
                                         stderr=subprocess.STDOUT), fold, gpu))
        print(f"{(time.time() - t0) / 60:6.1f} min  start fold {fold} on GPU {gpu}")
    for r in list(running):
        if r[0].poll() is not None:
            running.remove(r)
            print(f"{(time.time() - t0) / 60:6.1f} min  fold {r[1]} exit {r[0].returncode}")
            print("".join(l for l in open(f"logs/fold{r[1]}.txt") if "macro AUC" in l or "Error" in l)[-2000:])
            if r[0].returncode != 0:   # keep the other folds going; report at the end
                failed.append(r[1]); print(open(f"logs/fold{r[1]}.txt").read()[-3000:])
    time.sleep(20)
print("failed folds:", failed or "none")'''),
        ("md", f"## {4 if teacher else 3}. Cross-validation summary\n"
               "* **oof_auc** — all folds' out-of-fold predictions pooled, vs report targets (noise ≈ 0.003).\n"
               "* **gold_auc_ensemble** — the 5 fold models rank-averaged on the 58 gold studies, i.e. what the "
               "submission ensemble does (noise ≈ ±0.02)."),
        ("code", '''exp_name = json.load(open(sorted(glob.glob("models/*/fold*_metrics.json"))[0]))["exp_name"]
run(f"{CODE}/scripts/merge_oof.py", "--exp", exp_name, "--oof_dir", "models/oof",
    "--train", f"{COMP}/train.csv", "--targets", "models/oof/report_targets.csv")
cv = json.load(open(f"models/oof/{exp_name}_cv.json"))
import pandas as pd
print(pd.DataFrame({k: cv[k] for k in ("oof_per_label", "gold_per_label") if k in cv}).round(3).T.to_string())
EXP.update({k: v for k, v in cv.items() if "per_label" not in k}, minutes=round((time.time() - t0) / 60, 1))
json.dump(EXP, open("/kaggle/working/experiment.json", "w"), indent=1)
print(EXP)'''),
    ]
    return cells


def nb_inference(weights: str = None, family_weights: str = None, infer_args: str = None,
                 ext_weight: float = None, ext_weight_map: dict = None, ext224: float = None,
                 ext_hel: float = None):
    srcs = ", ".join(f"`{USER}/rsna-knee-{s}` ({EXPERIMENTS[s]['id']})" for s in EXPERIMENTS["inference"]["sources"])
    if weights:
        srcs = f"the weights dataset `{weights}` (uploaded by `scripts/vast/submit.sh`)"
    return [
        ("md", header("inference", "RSNA Knee — ensemble inference (internet OFF)",
                      f"Scores the hidden test set with every fold model attached from {srcs}. Test DICOMs go "
                      "through exactly the same preprocessing as training (`src/dataset/dicom.py`); each model's "
                      "predictions are converted to per-finding ranks (AUC only depends on order) and averaged.")),
        ("code", exp_json("inference") + f"\nFAMILY_WEIGHTS = {family_weights!r}\nINFER_ARGS = {(infer_args or '').split()!r}\n"
         f"EXT_WEIGHT = {ext_weight!r}   # rank weight of the external nartaa 0.949 stage (None = off)\n"
         f"EXT_MAP = {ext_weight_map or {}!r}   # per-finding override of EXT_WEIGHT (finding-group routing)\n"
         f"EXT224 = {ext224!r}   # share of nartaa's 0.945 224-crop model inside the external anchor (None = off)\n"
         f"EXT_HEL = {ext_hel!r}   # rank weight of heliosli's public compact ConvNeXt-S stage (None = off)\n" + FIND),
        ("md", "## 1. Offline dependencies and checkpoints\nThe pylibjpeg wheels come from a training notebook's "
               "output (needed for JPEG-compressed DICOMs)."),
        ("code", '''WHEELS = walk_find(lambda d, f: d if any(x.startswith("pylibjpeg") and x.endswith(".whl") for x in f) else None)
if WHEELS:
    # --no-deps: never let an offline wheel (e.g. numpy) replace the image's own packages
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--find-links", WHEELS,
                    "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], check=False)
CKPTS = []
# our checkpoints only: the external stage's weights (no cfg inside) are run by its own script
walk_find(lambda d, f: CKPTS.extend(os.path.join(d, x) for x in sorted(f) if x.endswith(".pt")
                                    and "rsna-knee-publication-swa-weights" not in d) and None)
print("wheels:", WHEELS)
try:
    import pylibjpeg, libjpeg, openjpeg  # noqa: F401  JPEG / JPEG-2000 DICOM decoders
    print("JPEG decoders: OK")
except ImportError as e:
    print("JPEG decoders MISSING:", e)
for c in CKPTS: print(" ", c)
assert CKPTS, "attach the training notebooks' outputs"'''),
        *([("md", "## 1b. External stage — nartaa's 0.949 CoAtNet (verbatim scored source, OAI-trained; fallback pick)"),
           ("code", '''import shutil
t_ext = time.time()
subprocess.run([sys.executable, f"{CODE}/submission/external/nartaa_0949_check.py"], check=True)
subprocess.run([sys.executable, f"{CODE}/submission/external/nartaa_0949_inference.py"], check=True, cwd="/kaggle/working")
shutil.move("/kaggle/working/submission.csv", "/kaggle/working/ext_nartaa.csv")
if EXT224:  # second view of the anchor: nartaa's 224-crop efficiency checkpoint (verbatim scored source)
    subprocess.run([sys.executable, f"{CODE}/submission/external/nartaa_0945_check.py"], check=True)
    subprocess.run([sys.executable, f"{CODE}/submission/external/nartaa_0945_inference.py"], check=True, cwd="/kaggle/working")
    shutil.move("/kaggle/working/submission.csv", "/kaggle/working/ext_nartaa224.csv")
if EXT_HEL:  # heliosli's public compact ConvNeXt-S model (verbatim stage, writes ours_sub.csv)
    subprocess.run([sys.executable, f"{CODE}/submission/external/heliosli_compact_stage.py"], check=True, cwd="/kaggle/working")
    shutil.move("/kaggle/working/ours_sub.csv", "/kaggle/working/ext_heliosli.csv")
EXT_SECONDS = round(time.time() - t_ext)
print(f"external stage {EXT_SECONDS}s")''')] if ext_weight else []),
        ("md", "## 2. Predict and rank-average"),
        ("code", '''t0 = time.time()
from src.models.inference import main
sys.argv = ["inference", "--data_dir", COMP, "--ckpts", *CKPTS, "--rank_average", "--cache_dir", "/tmp/knee_cache",
            *(["--family_weights", FAMILY_WEIGHTS] if FAMILY_WEIGHTS else []), *INFER_ARGS,
            "--workers", str(os.cpu_count()), "--out", "/kaggle/working/submission.csv"]
main()
print(f"inference {time.time() - t0:.0f}s")'''),
        *([("code", '''import pandas as pd
# rank fusion: EXT_WEIGHT x nartaa (external) + (1 - EXT_WEIGHT) x our blend, per finding, then re-rank
ours = pd.read_csv("/kaggle/working/submission.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
ext = pd.read_csv("/kaggle/working/ext_nartaa.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
assert set(ours.index) == set(ext.index), "external and our predictions cover different studies"
ext = ext.loc[ours.index, ours.columns]
anchor = ext.rank(pct=True)
if EXT224:
    e224 = pd.read_csv("/kaggle/working/ext_nartaa224.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    assert set(e224.index) == set(ours.index), "224-crop predictions cover different studies"
    anchor = (1 - EXT224) * anchor + EXT224 * e224.loc[ours.index, ours.columns].rank(pct=True)
w = pd.Series({c: float(EXT_MAP.get(c, EXT_WEIGHT)) for c in ours.columns})   # nartaa share per finding
wh = float(EXT_HEL or 0.0)
assert (w + wh <= 1 + 1e-9).all(), "external weights exceed 1"
fused = anchor.mul(w, axis=1) + ours.rank(pct=True).mul(1 - w - wh, axis=1)
if wh:
    hel = pd.read_csv("/kaggle/working/ext_heliosli.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    assert set(hel.index) == set(ours.index), "heliosli predictions cover different studies"
    fused = fused + wh * hel.loc[ours.index, ours.columns].rank(pct=True)
fused = fused.rank(pct=True)
assert fused.notna().all().all()
fused.reset_index().to_csv("/kaggle/working/submission.csv", index=False)
print("fused", fused.shape, "| ext weight per finding", w.round(3).to_dict(), "| ext224", EXT224, "| heliosli", EXT_HEL)''')] if ext_weight else []),
        ("code", '''import numpy as np, pandas as pd
sub = pd.read_csv("/kaggle/working/submission.csv")
files = glob.glob("/tmp/knee_cache/*/*.npz")   # only written in --mode cache (stream keeps stacks in memory)
EXP.update(models=len(CKPTS), test_studies=len(sub), seconds=round(time.time() - t0),
           ckpts=[os.path.join(os.path.basename(os.path.dirname(c)), os.path.basename(c)) for c in CKPTS],
           ckpt_paths=CKPTS, family_weights=FAMILY_WEIGHTS, ext_weight=EXT_WEIGHT,
           ext_seconds=globals().get("EXT_SECONDS"),
           upload_ids=[])
walk_find(lambda d, f: EXP["upload_ids"].extend(sorted(x for x in f if x.startswith("UPLOAD_"))) and None)
if files:
    EXP["test_slot_coverage"] = round(float(np.mean([np.load(f)["valid"].mean() for f in files])), 4)
json.dump(EXP, open("/kaggle/working/experiment.json", "w"), indent=1)
print(EXP); sub.head()'''),
    ]


def nb_pack(key: str = "cache-pack", name: str = "cache_256", source: str = "rsna-knee-cache"):
    return [
        ("md", header(key, "RSNA Knee — pack the cache into shards",
                      "Kaggle's API downloads notebook outputs file by file, so 4,407 `.npz` files are impractical to "
                      "fetch on a rented GPU. This notebook re-packs `" + source + "` into a few plain `.tar` shards "
                      "(the npz files are already compressed) plus a manifest. `scripts/vast/get_data.sh` downloads "
                      "and unpacks them.")),
        ("code", exp_json(key) + f"\nNAME = {name!r}\n" + FIND),
        ("code", '''import hashlib, tarfile
CACHE = walk_find(lambda d, f: d if os.path.basename(d) == NAME and any(x.endswith(".npz") for x in f[:50]) else None)
files = sorted(f for f in os.listdir(CACHE) if f.endswith(".npz"))
total = sum(os.path.getsize(os.path.join(CACHE, f)) for f in files)
n_shards = max(1, round(total / 2**30))            # ~1 GiB per shard
manifest = {"name": NAME, "studies": len(files), "bytes": total, "shards": []}
for k in range(n_shards):
    name = f"{NAME}_part{k:02d}.tar"
    with tarfile.open(f"/kaggle/working/{name}", "w") as tar:
        for f in files[k::n_shards]:
            tar.add(os.path.join(CACHE, f), arcname=f"{NAME}/{f}")
    h = hashlib.sha256(open(f"/kaggle/working/{name}", "rb").read()).hexdigest()
    manifest["shards"].append({"file": name, "sha256": h, "bytes": os.path.getsize(f"/kaggle/working/{name}")})
    print(name, manifest["shards"][-1]["bytes"] / 2**20, "MiB")
json.dump(manifest, open("/kaggle/working/cache_manifest.json", "w"), indent=1)
EXP.update(studies=len(files), size_gib=round(total / 2**30, 2), shards=n_shards)
json.dump(EXP, open("/kaggle/working/experiment.json", "w"), indent=1)
print(EXP)'''),
    ]


def nb_bench():
    return [
        ("md", header("bench", "RSNA Knee — inference speed-up benchmark",
                      "Runs `scripts/bench_inference.py` on training DICOMs with the 5 EXP-009 checkpoints. Each "
                      "variant is timed on its own fresh studies (no file-cache advantage), then all variants predict "
                      "the same 58 gold studies and are compared with the original path.\n\n"
                      "| Variant | What |\n|---|---|\n"
                      "| A_cache_full | original: npz cache for all studies, then one DataLoader pass per checkpoint |\n"
                      "| B_stream_full | + threaded/raw DICOM reads, CPU/GPU overlap, one data pass for all checkpoints |\n"
                      "| C_stream_selective | + decode only the ~72 sampled slices; intensity window from them |\n"
                      "| D_stream_selective_pct4 | as C, plus every 4th slice for the intensity window |")),
        ("code", exp_json("bench") + "\n" + FIND),
        ("code", '''WHEELS = walk_find(lambda d, f: d if any(x.startswith("pylibjpeg") and x.endswith(".whl") for x in f) else None)
if WHEELS:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--find-links", WHEELS,
                    "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], check=False)
CKPTS = []
walk_find(lambda d, f: CKPTS.extend(os.path.join(d, x) for x in sorted(f) if x.endswith(".pt")) and None)
print(len(CKPTS), "checkpoints:", CKPTS)
t0 = time.time()
subprocess.run([sys.executable, f"{CODE}/scripts/bench_inference.py", "--data_dir", COMP, "--ckpts", *CKPTS,
                "--n_time", "150", "--out", "/kaggle/working/bench_inference.json"],
               env=dict(os.environ, PYTHONPATH=CODE), check=True)
EXP.update(json.load(open("/kaggle/working/bench_inference.json")), minutes=round((time.time() - t0) / 60, 1))
json.dump(EXP, open("/kaggle/working/experiment.json", "w"), indent=1)'''),
    ]


def nb_jobs(key: str):
    e = EXPERIMENTS[key]
    rows = "\n".join(f"| {j['exp']} | GPU {j['gpu']} | " + (f"train `{j['train'][0]}` fold {j['train'][1]} {' '.join(j['train'][2])}"
                     if "train" in j else f"re-score `{j['eval'][0]}` with k_eval {j['eval'][1]}") + " |" for j in e["jobs"])
    return [
        ("md", header(key, "RSNA Knee — ITER-02 fold-0 checks",
                      "Each GPU runs its own queue of jobs (logs in `/kaggle/working/logs/<exp>.txt`). Training jobs "
                      "save `models/<exp_name>/fold0.pt`, OOF + gold predictions and a CV json; the summary compares "
                      "every job with the current best's fold 0 (EXP-009: OOF 0.865, gold 0.876).\n\n"
                      "| Exp | GPU | Job |\n|---|---|---|\n" + rows)),
        ("code", exp_json(key) + f"\nJOBS = {json.dumps(e['jobs'])}\n" + FIND),
        ("code", PIP + "\n" + SETUP_DATA + '''
CACHE = walk_find(lambda d, f: d if os.path.basename(d) == "cache_256" and any(x.endswith(".npz") for x in f[:50]) else None)
print("cache:", CACHE, len(os.listdir(CACHE)), "studies")'''),
        ("md", "## Run the job queues (one thread per GPU)"),
        ("code", '''import threading, psutil, torch
n_gpu = max(1, torch.cuda.device_count())
workers = max(2, os.cpu_count() // n_gpu)
os.makedirs("logs", exist_ok=True)
t0 = time.time(); status = {}

def command(j):
    if "train" in j:
        cfg, fold, extra = j["train"]
        return [sys.executable, "-m", "src.models.train_pl", f"exp={cfg}", f"fold={fold}",
                f"resources.num_workers={workers}", f"paths.cache_dir={CACHE}", *extra]
    ck_rel, ks = j["eval"]
    ck = walk_find(lambda d, f: os.path.join(d, os.path.basename(ck_rel)) if d.endswith(os.path.dirname(ck_rel))
                   and os.path.basename(ck_rel) in f else None)
    return [sys.executable, f"{CODE}/scripts/eval_ckpt.py", "--ckpt", ck, "--k_eval", *map(str, ks),
            "--out", f"/kaggle/working/eval_{j['exp']}.json", f"paths.data_dir={DATA}", f"paths.cache_dir={CACHE}"]

def gpu_worker(gpu):
    for j in [j for j in JOBS if j["gpu"] % n_gpu == gpu]:
        t = time.time()
        with open(f"logs/{j['exp']}.txt", "w") as log:
            rc = subprocess.run(command(j), env=dict(ENV, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=log,
                                stderr=subprocess.STDOUT).returncode
        status[j["exp"]] = (rc, round((time.time() - t) / 60, 1))
        print(f"{(time.time() - t0) / 60:6.1f} min  {j['exp']} on GPU {gpu} exit {rc} ({status[j['exp']][1]} min)", flush=True)

threads = [threading.Thread(target=gpu_worker, args=(g,)) for g in range(n_gpu)]
[t.start() for t in threads]
while any(t.is_alive() for t in threads):
    time.sleep(600)
    vm = psutil.virtual_memory()
    print(f"{(time.time() - t0) / 60:6.1f} min  RAM {vm.used / 2**30:.1f}/{vm.total / 2**30:.1f} GiB  running", flush=True)
print(status)'''),
        ("md", "## Summary vs EXP-009 fold 0 (OOF 0.865 / gold 0.876)"),
        ("code", '''import pandas as pd
rows = []
for j in JOBS:
    r = {"exp": j["exp"], "exit": status.get(j["exp"], ("?", 0))[0], "minutes": status.get(j["exp"], ("?", 0))[1]}
    if "train" in j:
        from src.config import get_config
        name = get_config([f"exp={j['train'][0]}", *j["train"][2]]).exp_name
        f = f"models/{name}/fold{j['train'][1]}_metrics.json"
        if os.path.exists(f):
                m = json.load(open(f))
                run(f"{CODE}/scripts/merge_oof.py", "--exp", name, "--oof_dir", "models/oof", "--train", f"{COMP}/train.csv",
                    "--targets", "models/oof/report_targets.csv")
                cv = json.load(open(f"models/oof/{name}_cv.json"))
                r.update(name=name, backbone=m["backbone"], img=m["img_size"], epochs=m["epochs"], oof=cv.get("oof_auc"),
                         gold=cv.get("gold_auc_mean_fold"), lat_men_oof=cv.get("oof_per_label", {}).get("Lateral Meniscus"),
                         lat_men_gold=cv.get("gold_per_label", {}).get("Lateral Meniscus"), topk=m.get("topk_scores"))
    else:
        f = f"/kaggle/working/eval_{j['exp']}.json"
        if os.path.exists(f):
            for k, v in json.load(open(f))["results"].items():
                r[f"oof@k{k}"] = v["oof_auc"]; r[f"gold@k{k}"] = v["gold_auc"]
                r[f"lat_men_oof@k{k}"] = v["oof_per_label"].get("Lateral Meniscus")
    rows.append(r)
summary = pd.DataFrame(rows)
print(summary.to_string())
EXP.update(summary=summary.to_dict("records"), minutes=round((time.time() - t0) / 60, 1))
json.dump(EXP, open("/kaggle/working/experiment.json", "w"), indent=1, default=str)'''),
    ]


def metadata(key: str, gpu: bool, internet: bool, datasets=(), kernels=()):
    return {"id": f"{USER}/rsna-knee-{key}", "title": f"rsna-knee-{key}", "code_file": f"rsna-knee-{key}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": gpu,
            "enable_internet": internet, "dataset_sources": [f"{USER}/rsna-knee-code", *datasets],
            "competition_sources": [COMP], "kernel_sources": [f"{USER}/rsna-knee-{k}" for k in kernels],
            "model_sources": []}


def write(key: str, cells, meta):
    d = os.path.join(HERE, key)
    os.makedirs(d, exist_ok=True)
    nb = {"cells": [{"cell_type": "markdown" if t == "md" else "code", "metadata": {}, "source": s.splitlines(True),
                     **({"execution_count": None, "outputs": []} if t == "code" else {})} for t, s in cells],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(os.path.join(d, meta["code_file"]), "w"), indent=1)
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=2)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dataset", default=None,
                    help="inference notebook loads *.pt (+ wheels) from this dataset, e.g. hlly34/rsna-knee-weights "
                         "(scripts/vast/submit.sh), instead of the Kaggle training notebooks' outputs")
    ap.add_argument("--inference-id", default=None, help="experiment id for the inference notebook, e.g. EXP-006")
    ap.add_argument("--inference-what", default=None, help="one-line description of that inference experiment")
    ap.add_argument("--model-sources", nargs="*", default=None,
                    help="Kaggle Model variations to attach, e.g. hlly34/rsna-knee-effnet-b0-224/pyTorch/exp-009-fold0/1")
    ap.add_argument("--extra-datasets", nargs="*", default=[], help="extra datasets to attach (e.g. hlly34/rsna-knee-wheels)")
    ap.add_argument("--family-weights", default=None, help='e.g. "convnext_small_288=0.7,effnet_b0_224=0.3"')
    ap.add_argument("--ext-weight", type=float, default=None,
                    help="add the external nartaa 0.949 stage (submission/external) with this rank weight")
    ap.add_argument("--ext-weight-map", default=None,
                    help='per-finding external weight, e.g. "Lateral OA=0.6,PF OA=0.6" (others use --ext-weight)')
    ap.add_argument("--ext-hel", type=float, default=None,
                    help="rank weight of heliosli's public compact ConvNeXt-S stage (needs its two datasets attached)")
    ap.add_argument("--ext224", type=float, default=None,
                    help="share of nartaa's 0.945 224-crop model inside the external anchor (second external stage)")
    ap.add_argument("--infer-args", default=None,
                    help='extra src/models/inference.py args, e.g. "--split train --limit 64" (debug a failed hidden run)')
    args = ap.parse_args()
    if args.inference_id:
        EXPERIMENTS["inference"]["id"] = args.inference_id
    if args.inference_what:
        EXPERIMENTS["inference"]["what"] = args.inference_what
    write("cache-pack", nb_pack(), metadata("cache-pack", False, False, (), ["cache"]))
    e = EXPERIMENTS["cache-dense"]
    write("cache-dense", nb_cache("cache-dense", e["cfg"], e["cache_name"]), metadata("cache-dense", False, True))
    write("cache-dense-pack", nb_pack("cache-dense-pack", e["cache_name"], "rsna-knee-cache-dense"),
          metadata("cache-dense-pack", False, False, (), ["cache-dense"]))
    steps = {"raptor": "1. **Series per slot** — sag fluid 22, 2nd sag 18, cor fluid 15, 2nd cor 10, axial 15 slices.\n"
                       "2. **Slices** sorted by position, sampled evenly over 2–98% of each series; no left/right mirroring.\n"
                       "3. **Crop** 140 mm → 336 px (0.417 mm/px); per-series 2–98 percentile → uint8.",
             "r96": "1. **Series per slot**: sag fluid 26, 2nd sag 22, cor fluid 18, 2nd cor 12, axial 18 slices.\n"
                    "2. **Slices** sorted by position, sampled evenly over 2–98% of each series; no left/right mirroring.\n"
                    "3. **Crop** 140 mm → native 384 px (0.365 mm/px); per-series 2–98 percentile → uint8.",
             "reader": "1. **Every series** (largest 2 per plane × fat-suppression slot), native slices (≤ 64).\n"
                       "2. **Reorient** to a fixed patient frame (transpose / flips from ImageOrientationPatient).\n"
                       "3. **Resample** to 0.55 mm/px in a 154 mm frame (280 px); 1–99.5 percentile → uint8."}
    for key, e in EXPERIMENTS.items():
        if key.startswith(("cache-raptor-p", "cache-reader-p", "cache-r96-p")):
            write(key, nb_cache_part(key, e["cfg"], e["cache_name"], e["part"], e["parts"], steps[key.split("-")[1]]),
                  metadata(key, False, True))
    write("bench", nb_bench(), metadata("bench", True, False, (f"{USER}/rsna-knee-weights",), ()))
    write("iter02-a", nb_jobs("iter02-a"), metadata("iter02-a", True, True, LABEL_DATASETS, ["cache"]))
    write("iter02-b", nb_jobs("iter02-b"),
          metadata("iter02-b", True, True, (*LABEL_DATASETS, f"{USER}/rsna-knee-weights"), ["cache"]))
    write("dryrun", nb_dryrun(), metadata("dryrun-train", True, True, LABEL_DATASETS))
    write("cache", nb_cache(), metadata("cache", False, True))
    write("train-teacher", nb_train("train-teacher"), metadata("train-teacher", True, True, LABEL_DATASETS, ["cache"]))
    write("train-student", nb_train("train-student"),
          metadata("train-student", True, True, LABEL_DATASETS, ["cache", "train-teacher"]))
    if args.model_sources is not None:   # checkpoints straight from Kaggle Models (no re-upload)
        EXPERIMENTS["inference"]["sources"] = []
        meta = metadata("inference", True, False, tuple(args.extra_datasets), ())
        meta["model_sources"] = list(args.model_sources)
        write("inference", nb_inference(weights="Kaggle Models: " + ", ".join(sorted({m.split("/")[1] for m in args.model_sources})),
                                        family_weights=args.family_weights, infer_args=args.infer_args,
                                        ext_weight=args.ext_weight,
                                        ext_weight_map={k.strip(): float(v) for k, v in (kv.split("=") for kv in
                                                        (args.ext_weight_map or "").split(",") if kv)},
                                        ext224=args.ext224, ext_hel=args.ext_hel), meta)
    elif args.weights_dataset:
        EXPERIMENTS["inference"]["sources"] = []
        write("inference", nb_inference(weights=args.weights_dataset, family_weights=args.family_weights),
              metadata("inference", True, False, (args.weights_dataset, *args.extra_datasets), ()))
    else:
        write("inference", nb_inference(), metadata("inference", True, False, (), EXPERIMENTS["inference"]["sources"]))
    print("written:", sorted(k for k in os.listdir(HERE) if os.path.isdir(os.path.join(HERE, k))))
