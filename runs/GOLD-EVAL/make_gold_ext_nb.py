"""Generate the gold-58 evaluation notebook for the external stages + our 5-fold partners (rsna-knee-gold-eval).

Evaluation only: patched copies of the verbatim external scripts read a stand-in test root built from the 58 gold
studies (train DICOMs symlinked); the submission notebooks keep the unmodified sources.
"""
import json
import os
import sys

sys.path.insert(0, "/Users/hllj/Projects/rsna-knee-abnormality-detection/.claude/worktrees/dense-f0")
import importlib.util
spec = importlib.util.spec_from_file_location(
    "bk", "/Users/hllj/Projects/rsna-knee-abnormality-detection/.claude/worktrees/dense-f0/submission/build_kernels.py")
src = open(spec.origin).read()
FIND = src.split("FIND = '''", 1)[1].split("'''", 1)[0]

OUT = "/Users/hllj/.claude/jobs/171aabfa/tmp/gold_nb4"
os.makedirs(OUT, exist_ok=True)

c_setup = FIND + '''
WHEELS = walk_find(lambda d, f: d if any(x.startswith("pylibjpeg") and x.endswith(".whl") for x in f) else None)
if WHEELS:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--find-links", WHEELS,
                    "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], check=False)
import shutil
import pandas as pd
tr = pd.read_csv(f"{COMP}/train.csv", dtype={"StudyInstanceUID": str})
LAB = [c for c in tr.columns if c not in ("StudyInstanceUID", "Report")]
gold = tr.dropna(subset=LAB)
G = "/kaggle/working/goldroot"
os.makedirs(G + "/test_series", exist_ok=True)
gold[["StudyInstanceUID"]].to_csv(G + "/test.csv", index=False)
ts = pd.read_csv(f"{COMP}/train_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
ts[ts.StudyInstanceUID.isin(gold.StudyInstanceUID)].to_csv(G + "/test_series.csv", index=False)
for u in gold.StudyInstanceUID:
    os.symlink(f"{COMP}/train_series/{u}", f"{G}/test_series/{u}")
ss = pd.DataFrame({"StudyInstanceUID": gold.StudyInstanceUID.values})
for c in LAB:
    ss[c] = 0.5
ss.to_csv(G + "/sample_submission.csv", index=False)
print("gold studies", len(gold), "| series", ts.StudyInstanceUID.isin(gold.StudyInstanceUID).sum())
'''

c_ext = '''# evaluation-only patched copies: read the gold stand-in root instead of the competition test set
E = f"{CODE}/submission/external"
os.makedirs("/kaggle/working/ext", exist_ok=True)


def patched(name, dst, subs):
    s = open(f"{E}/{name}").read()
    for a, b in subs:
        assert a in s, (name, a)
        s = s.replace(a, b)
    open(dst, "w").write(s)
    return dst


NAR = [('cands = ["/kaggle/input/competitions/rsna-knee-abnormality-detection",',
        'cands = ["/kaggle/working/goldroot", "/kaggle/input/competitions/rsna-knee-abnormality-detection",')]
runs = [("n384", "nartaa_0949_check.py", "nartaa_0949_inference.py", NAR, "submission.csv"),
        ("n224", "nartaa_0945_check.py", "nartaa_0945_inference.py", NAR, "submission.csv"),
        ("hel", None, "heliosli_compact_stage.py",
         [("DATA = find_mount(COMPETITION, 'test.csv')", "DATA = Path('/kaggle/working/goldroot')")], "ours_sub.csv")]
for tag, check, script, subs, out in runs:
    t = time.time()
    if check:
        subprocess.run([sys.executable, f"{E}/{check}"], check=True)
    p = patched(script, f"/kaggle/working/{tag}.py", subs)
    r = subprocess.run([sys.executable, p], cwd="/kaggle/working")
    ok = r.returncode == 0 and os.path.exists(f"/kaggle/working/{out}")
    if ok:
        shutil.move(f"/kaggle/working/{out}", f"/kaggle/working/ext/{tag}.csv")
    print(f"[{tag}] rc={r.returncode} ok={ok} {time.time() - t:.0f}s", flush=True)
'''

c_ours = '''# our 5-fold partners on the 58 gold studies (plain + mirror passes saved raw)
CKPTS = []
walk_find(lambda d, f: CKPTS.extend(os.path.join(d, x) for x in sorted(f) if x.endswith(".pt")
                                    and ("convnext-s-288-dense-pl" in d or "reader-repro" in d)) and None)
for c in CKPTS: print(" ", c)
from src.models.inference import main
sys.argv = ["inference", "--data_dir", COMP, "--ckpts", *CKPTS, "--rank_average", "--cache_dir", "/tmp/knee_cache",
            "--split", "train", "--fold_eval", "99", "--mirror_tta", "--save_raw", "/kaggle/working/raw_partners.npz",
            "--workers", str(os.cpu_count()), "--out", "/kaggle/working/ours_gold.csv"]
main()
print(sorted(os.listdir("/kaggle/working/ext")))
'''

cells = [("markdown", "# Gold-58 evaluation: nartaa 384/224, heliosli compact, our pseudo-label ConvNeXt 5f and reader 5f\n"
                      "Evaluation only. External scripts are run as patched copies that read a stand-in test root of the 58 "
                      "gold studies; submissions use the unmodified sources."),
         ("code", c_setup), ("code", c_ext), ("code", c_ours)]
nb = {"cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(True), **({"outputs": [], "execution_count": None}
                 if t == "code" else {})} for t, s in cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(f"{OUT}/rsna-knee-gold-eval.ipynb", "w"), indent=1)
M = "hlly34/rsna-knee-convnext-s-288-dense-pl/pyTorch"
R = "hlly34/rsna-knee-reader-repro/pyTorch"
meta = {"id": "hlly34/rsna-knee-gold-eval", "title": "rsna-knee-gold-eval", "code_file": "rsna-knee-gold-eval.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": False,
        "dataset_sources": ["hlly34/rsna-knee-code", "hlly34/rsna-knee-wheels",
                            "nartaa/rsna-knee-publication-swa-weights-20261007", "heliosli/rsna-knee-compact-v1",
                            "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"],
        "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [],
        "model_sources": [f"{M}/exp-079-fold0/1"] + [f"{M}/exp-087-fold{i}/1" for i in range(1, 5)]
                         + [f"{R}/exp-042-fold0/1"] + [f"{R}/exp-088-fold{i}/1" for i in range(1, 5)]}
json.dump(meta, open(f"{OUT}/kernel-metadata.json", "w"), indent=2)
print("written", OUT)
