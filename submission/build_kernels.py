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
    "train-teacher": {"id": "EXP-002", "what": "teacher: EfficientNet-B0 @224, 5 folds x 15 epochs, blended report labels",
                      "exp": "effnet_224", "epochs": 15, "folds": [0, 1, 2, 3, 4], "extra": []},
    "train-student": {"id": "EXP-003", "what": "student: same model, targets = 50/50 report labels + teacher OOF "
                      "(Synovitis 70%, PF/Lateral OA 60%)", "exp": "teacher_pseudo", "epochs": 15,
                      "folds": [0, 1, 2, 3, 4], "extra": [], "teacher": "train-teacher"},
    "inference": {"id": "EXP-004", "what": "rank-average ensemble of all teacher + student folds",
                  "sources": ["train-teacher", "train-student"]},
}

# ---------------------------------------------------------------------------------------------------------
# shared code cells
FIND = '''import glob, json, os, subprocess, sys, time

def walk_find(pred, root="/kaggle/input", max_depth=6):
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


def nb_cache():
    return [
        ("md", header("cache", "RSNA Knee — build the full training cache (CPU)",
                      "Decoding ~570 GB of DICOM is CPU work, so it runs here without spending GPU quota. The output "
                      "(`cache_256/<StudyInstanceUID>.npz`, compressed to fit Kaggle's 20 GB output limit) is attached "
                      "to the training notebooks.")),
        ("code", exp_json("cache") + "\n" + FIND),
        ("md", "## What happens to each study\n"
               "1. **Pick series** — 6 slots: sagittal/coronal/axial × fluid-sensitive/not (same-plane fallback).\n"
               "2. **Order slices** by `ImagePositionPatient` along the slice normal (file names are random).\n"
               "3. **Canonical knee** — side from the image-centre x coordinate (tag as fallback); left knees are "
               "mirrored, sagittal runs lateral → medial.\n"
               "4. **Intensity** — per-series 1–99th percentile → uint8.\n"
               "5. **Geometry** — 140 mm physical centre crop → 256×256; 72 slices sampled over 4–96% of each series "
               "(20/10/16/8/14/4 per slot).\n"
               "6. **Save** `image (72,256,256)`, `slot`, `valid`, `side`."),
        ("code", '''subprocess.run([sys.executable, "-m", "pip", "install", "-q", "iterative-stratification",
                "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], check=True)
DATA = "/kaggle/tmp/data"; os.makedirs(DATA, exist_ok=True)
if not os.path.exists(f"{DATA}/raw"):
    os.symlink(COMP, f"{DATA}/raw")
ENV = dict(os.environ, DATA_DIR=DATA, PYTHONPATH=CODE)
t = time.time()
subprocess.run([sys.executable, f"{CODE}/scripts/preprocess.py", "exp=effnet_224", "preprocess.compress=true",
                f"preprocess.num_workers={os.cpu_count()}", "paths.cache_dir=/kaggle/working/cache_256"],
               env=ENV, check=True)
print(f"preprocess {(time.time() - t) / 60:.1f} min")'''),
        ("md", "## Sanity checks"),
        ("code", '''import numpy as np
files = sorted(glob.glob("/kaggle/working/cache_256/*.npz"))
sides, cover = [], []
for f in files[::10]:
    z = np.load(f); sides.append(int(z["side"])); cover.append(z["valid"].mean())
size = sum(os.path.getsize(f) for f in files) / 2**30
EXP.update(studies=len(files), size_gib=round(size, 2), side_R=sides.count(1), side_L=sides.count(-1),
           side_unknown=sides.count(0), slot_coverage=round(float(np.mean(cover)), 4))
print(EXP)
json.dump(EXP, open("/kaggle/working/experiment.json", "w"), indent=1)'''),
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


def nb_inference():
    srcs = ", ".join(f"`{USER}/rsna-knee-{s}` ({EXPERIMENTS[s]['id']})" for s in EXPERIMENTS["inference"]["sources"])
    return [
        ("md", header("inference", "RSNA Knee — ensemble inference (internet OFF)",
                      f"Scores the hidden test set with every fold model attached from {srcs}. Test DICOMs go "
                      "through exactly the same preprocessing as training (`src/dataset/dicom.py`); each model's "
                      "predictions are converted to per-finding ranks (AUC only depends on order) and averaged.")),
        ("code", exp_json("inference") + "\n" + FIND),
        ("md", "## 1. Offline dependencies and checkpoints\nThe pylibjpeg wheels come from a training notebook's "
               "output (needed for JPEG-compressed DICOMs)."),
        ("code", '''WHEELS = walk_find(lambda d, f: d if os.path.basename(d) == "wheels" and any(x.endswith(".whl") for x in f) else None)
if WHEELS:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", WHEELS,
                    "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], check=False)
CKPTS = []
walk_find(lambda d, f: CKPTS.extend(os.path.join(d, x) for x in sorted(f) if x.endswith(".pt")) and None)
print("wheels:", WHEELS)
for c in CKPTS: print(" ", c)
assert CKPTS, "attach the training notebooks' outputs"'''),
        ("md", "## 2. Predict and rank-average"),
        ("code", '''t0 = time.time()
from src.models.inference import main
sys.argv = ["inference", "--data_dir", COMP, "--ckpts", *CKPTS, "--rank_average", "--cache_dir", "/tmp/knee_cache",
            "--workers", str(os.cpu_count()), "--out", "/kaggle/working/submission.csv"]
main()
print(f"inference {time.time() - t0:.0f}s")'''),
        ("code", '''import numpy as np, pandas as pd
sub = pd.read_csv("/kaggle/working/submission.csv")
files = glob.glob("/tmp/knee_cache/*/*.npz")
EXP.update(models=len(CKPTS), test_studies=len(sub), seconds=round(time.time() - t0),
           test_slot_coverage=round(float(np.mean([np.load(f)["valid"].mean() for f in files])), 4))
json.dump(EXP, open("/kaggle/working/experiment.json", "w"), indent=1)
print(EXP); sub.head()'''),
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
    write("dryrun", nb_dryrun(), metadata("dryrun-train", True, True, LABEL_DATASETS))
    write("cache", nb_cache(), metadata("cache", False, True))
    write("train-teacher", nb_train("train-teacher"), metadata("train-teacher", True, True, LABEL_DATASETS, ["cache"]))
    write("train-student", nb_train("train-student"),
          metadata("train-student", True, True, LABEL_DATASETS, ["cache", "train-teacher"]))
    write("inference", nb_inference(), metadata("inference", True, False, (), EXPERIMENTS["inference"]["sources"]))
    print("written:", sorted(k for k in os.listdir(HERE) if os.path.isdir(os.path.join(HERE, k))))
