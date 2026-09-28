#!/usr/bin/env bash
# Progress of the runs on this box: per-epoch AUCs of every fold, GPU/CPU load, tmux sessions.
#   bash scripts/vast/status.sh
cd "$(dirname "$0")/../.."
PY=${PYTHON:-python}
"$PY" - <<'PY'
import glob, os, time
import pandas as pd
for f in sorted(glob.glob("logs/*/fold*/metrics.csv")):
    m = pd.read_csv(f)
    cols = [c for c in ("epoch", "val_auc_report", "val_auc_gold") if c in m]
    v = m[cols].dropna(subset=[cols[-1]]) if len(cols) > 1 else m.iloc[:0]
    age = (time.time() - os.path.getmtime(f)) / 60
    print(f"== {f}  (updated {age:.1f} min ago)")
    print(v.to_string(index=False) if len(v) else "  no finished epoch yet")
for f in sorted(glob.glob("models/*/fold*_metrics.json")):
    print("done:", f)
PY
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
uptime
tmux ls 2>/dev/null
