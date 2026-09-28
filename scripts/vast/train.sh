#!/usr/bin/env bash
# Train folds of one experiment config on the local GPU, then summarise CV.
#
#   bash scripts/vast/train.sh <config> "<folds>" [extra overrides...]
#   bash scripts/vast/train.sh effnet_224 0                              # teacher, fold 0 only (EXP-002 plan)
#   bash scripts/vast/train.sh effnet_224 "0 1 2 3 4"                    # all folds
#   bash scripts/vast/train.sh coatnet_384 0 preprocess.res=256 model.img_size=384   # bigger model, same cache
#   bash scripts/vast/train.sh teacher_pseudo "0 1 2 3 4" "labels.pseudo.oof_files=[models/oof/effnet_b0_224.csv]"
#
# Outputs: models/<exp_name>/fold<k>.pt (+ _metrics.json), models/oof/*, logs/
# The cache is 256 px; configs asking for another preprocess.res must override it (as above) or re-cache.
set -euo pipefail
cd "$(dirname "$0")/../.."
CFG=${1:?config name, e.g. effnet_224}
FOLDS=${2:-0}
shift $(( $# >= 2 ? 2 : 1 ))
DATA=${DATA_DIR:-data}
PY=${PYTHON:-python}
NW=${NUM_WORKERS:-$(( $(nproc) > 16 ? 16 : $(nproc) ))}
EXP_NAME=$("$PY" -c "from src.config import get_config; print(get_config(['exp=$CFG', *'$*'.split()]).exp_name)")
mkdir -p logs
echo "config=$CFG exp_name=$EXP_NAME folds=[$FOLDS] workers=$NW extra=[$*]"
for f in $FOLDS; do
  echo "== fold $f  $(date)"
  "$PY" -m src.models.train_pl "exp=$CFG" "fold=$f" "paths.data_dir=$DATA" "resources.num_workers=$NW" "$@" \
    2>&1 | tee "logs/${EXP_NAME}_fold${f}.txt" | grep -E "macro AUC|Error|train [0-9]+ \|" || true
  [ -f "models/$EXP_NAME/fold$f.pt" ] || { echo "fold $f failed; see logs/${EXP_NAME}_fold${f}.txt"; exit 1; }
done
"$PY" scripts/merge_oof.py --exp "$EXP_NAME" --oof_dir models/oof --train "$DATA/raw/train.csv" \
  --targets models/oof/report_targets.csv
echo "next: bash scripts/vast/submit.sh EXP-NNN \"<description>\" models/$EXP_NAME/fold*.pt"
