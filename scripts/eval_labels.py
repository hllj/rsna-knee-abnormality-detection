"""
Score label sources (and the configured blend) against the 58 radiologist-labelled studies.

    python scripts/eval_labels.py
    python scripts/eval_labels.py data/labels/my_llm_labels.csv models/oof/coatnet2_384_gold_fold0.csv

Differences under ~0.02 macro are within the noise of a 58-study ruler.
"""
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_config  # noqa: E402
from src.constants import ID_COL, LABELS  # noqa: E402
from src.dataset.folds import gold_ids  # noqa: E402
from src.evaluation.metrics import macro_auc  # noqa: E402
from src.labeling.blend import build_targets, load_source  # noqa: E402


def score(name, df, gold):
    pred = df.reindex(gold.index)[LABELS].fillna(0.5).values
    m, per = macro_auc(gold.values, pred)
    worst = sorted(per.items(), key=lambda t: t[1])[:3]
    print(f"{m:.4f}  {name:45s} worst: " + ", ".join(f"{k}={v:.3f}" for k, v in worst))


def main():
    paths = [a for a in sys.argv[1:] if "=" not in a]
    cfg = get_config([a for a in sys.argv[1:] if "=" in a])
    train = pd.read_csv(cfg.data.train_file, dtype={ID_COL: str})
    gold = train[train[ID_COL].isin(gold_ids(train))].set_index(ID_COL)[LABELS]
    if not paths:
        paths = sorted(glob.glob(os.path.join(cfg.paths.labels_dir, "*.csv")))
    for p in paths:
        score(os.path.basename(p), load_source(p)[0], gold)
    score("== configured blend (build_targets) ==", build_targets(cfg), gold)


if __name__ == "__main__":
    main()
