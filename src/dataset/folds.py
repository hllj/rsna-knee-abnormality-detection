"""
Fold assignment.

* The 58 radiologist-labelled ("gold") studies get fold = -1: never trained on, scored by
  every fold model as an independent (small, noisy) check.
* The rest are split with multilabel-stratified K-fold on binarised report labels.
  Validate on report labels of the held-out fold: the gold set is too small to resolve
  differences under ~0.02 macro AUC.
"""
import numpy as np
import pandas as pd

from src.constants import ID_COL, LABELS


def gold_ids(train: pd.DataFrame) -> list:
    """StudyInstanceUIDs whose 12 labels are all present in train.csv."""
    return train.loc[train[LABELS].notna().all(axis=1), ID_COL].astype(str).tolist()


def make_folds(train: pd.DataFrame, targets: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Return DataFrame[StudyInstanceUID, fold]."""
    # imported here: iterative-stratification is not in the offline Kaggle image, and gold_ids() is used there
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

    gold = set(gold_ids(train))
    ids = [u for u in train[ID_COL].astype(str) if u not in gold]
    y = (targets.reindex(ids).fillna(0).values >= 0.5).astype(int)
    folds = np.zeros(len(ids), dtype=int)
    mskf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(mskf.split(np.zeros(len(ids)), y)):
        folds[va] = k
    out = pd.DataFrame({ID_COL: ids, "fold": folds})
    return pd.concat([out, pd.DataFrame({ID_COL: sorted(gold), "fold": -1})], ignore_index=True)
