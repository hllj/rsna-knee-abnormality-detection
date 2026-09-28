"""Generate the Kaggle notebooks in submission/kernel and submission/dryrun (run after edits)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

FIND = '''import glob, os, subprocess, sys, time

def walk_find(pred, root="/kaggle/input", max_depth=5):
    """Depth-limited search that never descends into the DICOM trees."""
    root = root.rstrip("/")
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        dirnames[:] = [] if depth >= max_depth else [d for d in dirnames if not d.endswith("_series")]
        hit = pred(dirpath, filenames)
        if hit:
            return hit
    return None

CODE = walk_find(lambda d, f: d if os.path.isfile(os.path.join(d, "src", "models", "inference.py")) else None)
COMP = walk_find(lambda d, f: d if "test.csv" in f and os.path.isdir(os.path.join(d, "test_series")) else None)
assert CODE and COMP, (CODE, COMP)
sys.path.insert(0, CODE)
print("code:", CODE, "| competition:", COMP)
'''

DRYRUN = [
    "# RSNA Knee — dry run on a subset (train + gold eval + test inference)\n"
    "# Verifies the whole pipeline on real DICOMs: 58 gold + N random studies, EfficientNet-B0 fold 0.\n"
    "LIMIT, EPOCHS, EXP = 600, 8, 'effnet_224'\n" + FIND,
    # deps + offline wheels for the inference kernel
    '''subprocess.run([sys.executable, "-m", "pip", "install", "-q", "iterative-stratification", "lightning",
                "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], check=True)
os.makedirs("/kaggle/working/wheels", exist_ok=True)
subprocess.run([sys.executable, "-m", "pip", "download", "-q", "-d", "/kaggle/working/wheels",
                "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], check=False)
import torch, timm, pydicom
print(torch.__version__, timm.__version__, pydicom.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")''',
    # data layout: DATA_DIR/raw -> competition, DATA_DIR/labels <- public label sets
    '''DATA = "/kaggle/tmp/data"
os.makedirs(f"{DATA}/labels", exist_ok=True); os.makedirs(f"{DATA}/processed", exist_ok=True)
if not os.path.exists(f"{DATA}/raw"):
    os.symlink(COMP, f"{DATA}/raw")
for name in ["llm_labels_v4_blend.csv", "llm_labels_full.csv", "report_labels_v2.csv"]:
    src = walk_find(lambda d, f: os.path.join(d, name) if name in f else None)
    print(name, "->", src)
    if src and not os.path.exists(f"{DATA}/labels/{name}"):
        os.symlink(src, f"{DATA}/labels/{name}")
ENV = dict(os.environ, DATA_DIR=DATA, PYTHONPATH=CODE)
os.chdir("/kaggle/working")

def run(*args):
    t = time.time()
    subprocess.run([sys.executable, *args], env=ENV, check=True)
    print(f"[{args[0]} {args[1] if len(args) > 1 else ''}] {time.time() - t:.0f}s")

run(f"{CODE}/scripts/eval_labels.py")''',
    # preprocess subset
    '''run(f"{CODE}/scripts/preprocess.py", f"exp={EXP}", f"preprocess.limit={LIMIT}", "preprocess.num_workers=4")
import numpy as np
files = sorted(glob.glob(f"{DATA}/processed/cache_256/*.npz"))
sides = [int(np.load(f)["side"]) for f in files]
cover = np.mean([np.load(f)["valid"].mean() for f in files])
print(f"cached {len(files)} studies | knee side R/L/unknown = {sides.count(1)}/{sides.count(-1)}/{sides.count(0)} | slot coverage {cover:.1%}")''',
    # look at a right and a left knee after canonicalisation
    '''import matplotlib.pyplot as plt
d = {s: next(f for f, x in zip(files, sides) if x == s) for s in (1, -1) if s in sides}
fig, ax = plt.subplots(len(d), 6, figsize=(18, 3 * len(d)), squeeze=False)
for r, (s, f) in enumerate(d.items()):
    z = np.load(f)
    for c in range(6):
        idx = np.where(z["slot"] == c)[0]
        ax[r, c].imshow(z["image"][idx[len(idx) // 2]], cmap="gray"); ax[r, c].axis("off")
        ax[r, c].set_title(f"{'R' if s == 1 else 'L'} knee slot {c}")
plt.tight_layout(); plt.savefig("/kaggle/working/canonical_examples.png", dpi=60); plt.show()''',
    # train fold 0
    '''run("-m", "src.models.train_pl", f"exp={EXP}", "fold=0", f"training.num_epochs={EPOCHS}",
    "resources.num_workers=4", "training.batch_size=16")''',
    # evaluate OOF and gold vs the label source
    '''import pandas as pd
sys.path.insert(0, CODE)
from src.constants import LABELS, ID_COL
from src.evaluation.metrics import macro_auc
train = pd.read_csv(f"{COMP}/train.csv", dtype={ID_COL: str}).set_index(ID_COL)
gold = train.dropna(subset=LABELS)[LABELS]
g = pd.read_csv(glob.glob("models/oof/*_gold_fold0.csv")[0], dtype={ID_COL: str}).set_index(ID_COL).reindex(gold.index)
m, per = macro_auc(gold.values, g[LABELS].values)
lab = pd.read_csv(f"{DATA}/labels/llm_labels_v4_blend.csv", dtype={ID_COL: str}).set_index(ID_COL).reindex(gold.index)
ml, perl = macro_auc(gold.values, lab[LABELS].fillna(0.5).values)
print(f"GOLD-58 macro AUC  image model {m:.4f}  |  report labels (teacher) {ml:.4f}")
print(pd.DataFrame({"model": per, "labels": perl}).round(3).T.to_string())''',
    # test inference exactly as in the submission kernel
    '''run("-m", "src.models.inference", "--data_dir", COMP, "--ckpts", "models/**/fold*.pt",
    "--cache_dir", "/kaggle/tmp/test_cache", "--workers", "4", "--out", "/kaggle/working/submission.csv")
print(pd.read_csv("/kaggle/working/submission.csv"))''',
]

INFER = [
    "# RSNA Knee — offline inference (internet OFF). Weights: any attached *.pt export.\n" + FIND,
    '''WHEELS = walk_find(lambda d, f: d if os.path.basename(d) == "wheels" and any(x.endswith(".whl") for x in f) else None)
if WHEELS:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", WHEELS,
                    "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], check=False)
CKPTS = []
walk_find(lambda d, f: CKPTS.extend(os.path.join(d, x) for x in f if x.endswith(".pt")) and None)
print("wheels:", WHEELS); print("ckpts:", CKPTS)
assert CKPTS, "attach a weights dataset / training kernel output"''',
    '''t0 = time.time()
from src.models.inference import main
sys.argv = ["inference", "--data_dir", COMP, "--ckpts", *CKPTS, "--cache_dir", "/tmp/knee_cache",
            "--workers", str(os.cpu_count()), "--out", "/kaggle/working/submission.csv"]
main()
print(f"inference {time.time() - t0:.0f}s")''',
    '''import numpy as np, pandas as pd
sub = pd.read_csv("/kaggle/working/submission.csv")
files = glob.glob("/tmp/knee_cache/*/*.npz")
print(sub.shape, "| slot coverage on test:", np.mean([np.load(f)["valid"].mean() for f in files]).round(3))
sub.head()''',
]


def write(cells, path):
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                     "source": c.splitlines(keepends=True)} for c in cells],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(path, "w"), indent=1)


if __name__ == "__main__":
    write(DRYRUN, os.path.join(HERE, "dryrun", "rsna-knee-dryrun-train.ipynb"))
    write(INFER, os.path.join(HERE, "kernel", "rsna-knee-inference.ipynb"))
    print("notebooks written")
