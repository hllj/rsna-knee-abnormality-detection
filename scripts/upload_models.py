"""
Back up trained checkpoints to Kaggle Models, one variation per checkpoint (private by default).

    python scripts/upload_models.py --model rsna-knee-effnet-b0-224 --exp EXP-009 \
        runs/EXP-009/models/effnet_b0_224/fold*.pt

Creates <user>/<model> if needed, then for every checkpoint a PyTorch variation "<exp>-fold<k>" holding
fold<k>.pt, its fold<k>_metrics.json (if present) and the usage notes below. Re-running for an existing
variation adds a new version instead.

Load later (Kaggle notebook: attach the model variation):
    ck = torch.load(".../fold0.pt", map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ck["cfg"]); model = build_model(cfg, pretrained=False); model.load_state_dict(ck["state_dict"])
or just point the inference notebook / src.models.inference --ckpts at the .pt files.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile

import torch

KAGGLE = os.environ.get("KAGGLE_BIN", "kaggle")


def user_name() -> str:
    return json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))["username"]


def kaggle(*args, check=True) -> subprocess.CompletedProcess:
    r = subprocess.run([KAGGLE, *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return r


def usage_md(ck_path: str, cfg: dict, metrics: dict, exp: str) -> str:
    m = cfg["model"]
    lines = [
        "# Model Format",
        f"PyTorch checkpoint `{os.path.basename(ck_path)}`: dict with `state_dict` and the full training `cfg` "
        "(OmegaConf) of the RSNA knee pipeline (repo `src/models/model.py::build_model`).",
        "",
        "# Training Data",
        "RSNA Knee Abnormality Detection 2026 training studies (report-derived soft labels blended from "
        f"{', '.join(cfg['labels']['sources'])}); fold {cfg['fold']} of `configs/folds_5.csv` held out, "
        "58 radiologist-labelled studies never trained on.",
        "",
        "# Model Inputs",
        f"Per study: bag of {cfg['windows']['k_eval']} windows of 3 adjacent MRI slices (RGB) at "
        f"{m['img_size']} px from the cached 6-slot stack ({cfg['preprocess']['crop_mm']} mm crop, canonical right knee), "
        "plus slot ids and a window mask.",
        "",
        "# Model Outputs",
        "12 logits: ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, "
        "Synovitis, Baker's, Contusion, Fracture.",
        "",
        "# Model Usage",
        "```python\nck = torch.load('fold.pt', map_location='cpu', weights_only=False)\n"
        "model = build_model(OmegaConf.create(ck['cfg']), pretrained=False)\nmodel.load_state_dict(ck['state_dict'])\n```",
        "Or pass the .pt files to `python -m src.models.inference --ckpts ...`.",
        "",
        "# Changelog",
        f"{exp}: backbone `{m['backbone']}`, {cfg['training']['num_epochs']} epochs, top-"
        f"{cfg['training']['top_k_average']} epoch weight average.",
    ]
    if metrics:
        lines += ["", "# Evaluation Results",
                  f"* OOF macro AUC (held-out fold, report labels): {metrics.get('oof_auc')}",
                  f"* Gold-58 macro AUC (radiologist labels): {metrics.get('gold_auc')}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--model", required=True, help="model slug, e.g. rsna-knee-effnet-b0-224")
    ap.add_argument("--exp", required=True, help="experiment id, used in the variation slug")
    ap.add_argument("--title", default=None)
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--license", default="Apache 2.0")
    a = ap.parse_args()
    user = user_name()

    # 1. the model (container)
    if kaggle("models", "get", f"{user}/{a.model}", check=False).returncode != 0:
        d = tempfile.mkdtemp()
        json.dump({"ownerSlug": user, "title": a.title or a.model, "slug": a.model, "subtitle":
                   "RSNA Knee 2026: 2.5D attention-MIL fold checkpoints", "isPrivate": not a.public,
                   "description": "# Model Summary\nFold checkpoints of the RSNA Knee Abnormality Detection pipeline. "
                   "One variation per checkpoint (<experiment>-fold<k>). See each variation's usage notes.",
                   "publishTime": "", "provenanceSources": ""},
                  open(os.path.join(d, "model-metadata.json"), "w"), indent=1)
        print(kaggle("models", "create", "-p", d).stdout.strip())
        shutil.rmtree(d)

    # 2. one variation per checkpoint
    for ck_path in a.ckpts:
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        fold = cfg["fold"]
        slug = re.sub(r"[^a-z0-9-]", "-", f"{a.exp}-fold{fold}".lower())
        mpath = ck_path[:-3] + "_metrics.json"
        metrics = json.load(open(mpath)) if os.path.exists(mpath) else {}
        d = tempfile.mkdtemp()
        shutil.copy(ck_path, d)
        if metrics:
            shutil.copy(mpath, d)
        meta = {"ownerSlug": user, "modelSlug": a.model, "instanceSlug": slug, "framework": "pyTorch",
                "overview": f"{a.exp} fold {fold}: {cfg['model']['backbone']} @ {cfg['model']['img_size']} px"
                            + (f" — OOF {metrics.get('oof_auc')}, gold-58 {metrics.get('gold_auc')}" if metrics else ""),
                "usage": usage_md(ck_path, cfg, metrics, a.exp), "licenseName": a.license, "fineTunable": True,
                "trainingData": [], "modelInstanceType": "Unspecified", "baseModelInstanceId": 0,
                "externalBaseModelUrl": ""}
        json.dump(meta, open(os.path.join(d, "model-instance-metadata.json"), "w"), indent=1)
        exists = kaggle("models", "instances", "get", f"{user}/{a.model}/pyTorch/{slug}", check=False).returncode == 0
        if exists:
            r = kaggle("models", "instances", "versions", "create", f"{user}/{a.model}/pyTorch/{slug}",
                       "-p", d, "-n", f"{a.exp} re-upload")
        else:
            r = kaggle("models", "instances", "create", "-p", d)
        print(f"{slug}: {'new version' if exists else 'created'} <- {ck_path}")
        print("   ", r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip())
        shutil.rmtree(d)


if __name__ == "__main__":
    main()
