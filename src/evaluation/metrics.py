"""
Competition metric: mean over the 12 findings of ROC AUC.
"""
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from src.constants import LABELS


def macro_auc(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Tuple[float, Dict[str, float]]:
    """
    Macro AUC. Soft targets are binarised at `threshold`; columns with a single class are skipped.

    Returns:
        (macro, {label: auc})
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    per = {}
    for i, name in enumerate(LABELS):
        t = (y_true[:, i] >= threshold).astype(int)
        if t.min() == t.max():
            continue
        per[name] = float(roc_auc_score(t, y_pred[:, i]))
    return (float(np.mean(list(per.values()))) if per else float("nan")), per
