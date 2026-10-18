"""Gold-58 AUC of S6 models on the px cache, the way train_s6f0_all.predict scores (eval windows, centre crop, bf16).

  python scripts/vast/s6_gold.py --px data/s6_px320 --labels data/raw/train.csv \
      --model ours=runs/EXP-128/models/E_prior_s43_fall.pt --model nartaa=data/s6fall/E_prior_s42_fall.pt [--out gold.json]
Prints per-finding and macro AUC per model and for the rank-average of all models."""
import argparse, csv, json, os, sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

S6 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "submission", "external", "s6_train")
sys.path.insert(0, S6)
import train_s6f0_all as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--px", required=True)
    ap.add_argument("--labels", required=True, help="competition train.csv (gold rows carry the 12 findings)")
    ap.add_argument("--targets", default=os.path.join(S6, "data", "targets_s6f0.csv"))
    ap.add_argument("--model", action="append", required=True, help="name=path of a deploy state dict")
    ap.add_argument("--backbone", default="convnext_small")
    ap.add_argument("--crop", type=int, default=288)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    index = json.load(open(os.path.join(a.px, "index.json")))
    done = {u for u, v in index["done"].items() if v[1] > 0}
    gold = [r["StudyInstanceUID"] for r in csv.DictReader(open(a.targets)) if r["is_gold"] == "1" and r["StudyInstanceUID"] in done]
    lab = pd.read_csv(a.labels, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[gold, T.LABELS].astype(float)
    print("gold studies", len(gold), flush=True)
    ds = T.Studies(a.px, gold, 64, "eval", 0, eval_stride=3, phys_mm=3.5, slices=3)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=4, collate_fn=T.collate)
    o = (320 - a.crop) // 2
    preds = {}
    for spec in a.model:
        name, path = spec.split("=", 1)
        net = T.Net(a.backbone, None, crop=a.crop, in_chans=3, spatial=True).cuda().eval()
        net.load_deploy_state_dict(torch.load(path, map_location="cpu"))
        out = np.zeros((len(gold), T.C), np.float32)
        with torch.no_grad():
            for img, meta, mask, idx in dl:
                img = img[..., o:o + a.crop, o:o + a.crop].cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = net(img, meta.cuda(), mask.cuda(), chunk=T.PREDICT_CHUNK)
                out[idx.numpy()] = z.float().cpu().numpy()
        preds[name] = pd.DataFrame(out, index=gold, columns=T.LABELS)
        del net
        torch.cuda.empty_cache()
    if len(preds) > 1:
        preds["rank_mean"] = sum(p.rank(pct=True) for p in preds.values()) / len(preds)
    rows = {n: {c: roc_auc_score(lab[c], p[c]) for c in T.LABELS} for n, p in preds.items()}
    df = pd.DataFrame(rows).T
    df["macro"] = df.mean(1)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(), flush=True)
    if a.out:
        json.dump({"auc": df.round(4).to_dict(orient="index"),
                   "preds": {n: p.round(5).to_dict(orient="index") for n, p in preds.items()}}, open(a.out, "w"))


if __name__ == "__main__":
    main()
