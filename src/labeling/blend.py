"""
Build soft training targets from report-derived label sources.

Pipeline:
  1. Load each source csv (StudyInstanceUID + 12 columns in [0, 1]).
     Cells equal to 0.5, or with a `<label>__verdict` of UNK, count as "report is silent".
  2. Fill silent cells with a per-finding prior (config `labels.silence`):
       * a number: silence carries information (Baker's: 3% positive when not mentioned)
       * "from_effusion": copy the Effusion cell (synovitis is rarely written down but
         co-occurs with effusion: P(syn|eff)=0.63 vs 0.22 on the gold set)
  3. Weighted mean across sources.
  4. Optional pseudo-label blend with out-of-fold teacher predictions:
       y = (1 - w_label) * y_report + w_label * y_oof
     Only ever use OUT-OF-FOLD predictions; in-fold predictions just echo the labels back.
"""
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.constants import ID_COL, LABELS

SILENT = 0.5


def load_source(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load one label csv/parquet, returning (uid-indexed values, uid-indexed silent mask)."""
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    df[ID_COL] = df[ID_COL].astype(str)
    df = df.drop_duplicates(ID_COL).set_index(ID_COL)
    vals = df[LABELS].astype(float)
    silent = np.isclose(vals.values, SILENT)
    verdicts = [f"{c}__verdict" for c in LABELS]
    if all(v in df.columns for v in verdicts):
        silent |= df[verdicts].astype(str).isin(["UNK", "unknown", "not_addressed"]).values
    return vals, pd.DataFrame(silent, index=vals.index, columns=LABELS)


def fill_silence(vals: pd.DataFrame, silent: pd.DataFrame, silence_cfg: Dict) -> pd.DataFrame:
    """Replace silent cells with the per-finding prior from config."""
    out = vals.copy()
    default = float(silence_cfg.get("default", 0.1))
    for c in LABELS:
        rule = silence_cfg.get(c, default)
        m = silent[c].values
        if isinstance(rule, str) and rule == "from_effusion":
            eff = vals["Effusion"].where(~silent["Effusion"], other=np.nan)
            out.loc[m, c] = eff[m].fillna(default).values
        else:
            out.loc[m, c] = float(rule)
    return out


def blend_sources(
    paths: List[str],
    silence_cfg: Optional[Dict] = None,
    weights: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Weighted mean of silence-filled sources over the union of their study ids."""
    weights = weights or [1.0] * len(paths)
    num, den = None, None
    for p, w in zip(paths, weights):
        vals, silent = load_source(p)
        v = fill_silence(vals, silent, silence_cfg) if silence_cfg else vals
        wv = v * w
        wd = v.notna().astype(float) * w
        num = wv.fillna(0) if num is None else num.add(wv.fillna(0), fill_value=0)
        den = wd if den is None else den.add(wd, fill_value=0)
    return (num / den.replace(0, np.nan)).clip(0, 1)


def apply_pseudo(
    y: pd.DataFrame, oof_paths: List[str], weight: float = 0.5, per_label: Optional[Dict] = None
) -> pd.DataFrame:
    """Blend report labels with (averaged) out-of-fold teacher predictions."""
    oofs = [load_source(p)[0] for p in oof_paths]
    oof = sum(o.reindex(y.index) for o in oofs) / len(oofs)
    out = y.copy()
    for c in LABELS:
        w = float((per_label or {}).get(c, weight))
        has = oof[c].notna()
        out.loc[has, c] = (1 - w) * y.loc[has, c] + w * oof.loc[has, c]
    return out


def build_targets(cfg) -> pd.DataFrame:
    """Targets for every training study from cfg.labels (uid-indexed, 12 columns)."""
    lc = cfg.labels
    paths = [p if os.path.isabs(p) else os.path.join(cfg.paths.labels_dir, p) for p in lc.sources]
    silence = dict(lc.silence) if lc.get("silence") else None
    y = blend_sources(paths, silence, list(lc.weights) if lc.get("weights") else None)
    if lc.pseudo.enabled and lc.pseudo.oof_files:
        y = apply_pseudo(y, list(lc.pseudo.oof_files), lc.pseudo.weight, dict(lc.pseudo.per_label_weight or {}))
    if lc.get("hard"):
        y = (y >= 0.5).astype(float)
    return y.fillna(0.0)
