#!/usr/bin/env bash
# ITER-07 on one rented RTX 3090 (2026-10-08):
#   1. EXP-050  reader_v2 fold 0           (top-3 EMA averaging, 11 epochs, our labels)  -> rsna-knee-reader-v2
#   2. EXP-051  raptor_repro folds 1-4     (identical recipe to EXP-041, LB 0.928 f0)    -> rsna-knee-raptor-repro
#   3. EXP-052  reader folds 1-4 of the config named in reader_choice.txt (reader_v2 | reader_repro), written after
#               the LB check of reader_v2 fold 0 vs reader_repro fold 0 (0.922); waits until the file exists
# Every fold is backed up to Kaggle Models as soon as it finishes (so blends can be submitted progressively).
#
#   tmux new -d -s chain 'bash scripts/vast/iter07_chain.sh > logs_chain_full.txt 2>&1'   # progress: logs_chain.txt
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs models/oof
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$* $(date -u +%H:%M)" | tee -a logs_chain.txt; }
[ -f models/oof/report_targets.csv ] || { log "missing models/oof/report_targets.csv (EXP-024 ruler)"; exit 1; }

fetch() {  # $1 cache name, rest part notebooks
  local name=$1; shift
  until CACHE_PACKS="$*" CACHE_NAME=$name bash scripts/vast/get_data.sh >> "logs/get_$name.txt" 2>&1 \
        && [ "$(ls "data/processed/$name" | wc -l)" -ge 4407 ]; do
    log "retrying fetch of $name"; sleep 300
  done
  log "$name studies: $(ls "data/processed/$name" | wc -l)"
}

fold() {  # $1 config, $2 fold, $3 Kaggle model slug, $4 EXP id, rest: extra overrides
  local cfg=$1 f=$2 slug=$3 exp=$4; shift 4
  local name
  name=$(python -c "from src.config import get_config; print(get_config(['exp=$cfg']).exp_name)")
  if [ ! -f "models/$name/fold$f.pt" ]; then
    rm -rf "logs/$name/fold$f"
    bash scripts/vast/train.sh "$cfg" "$f" "$@" > "logs/train_${name}_fold$f.txt" 2>&1
  fi
  if [ ! -f "models/$name/fold$f.pt" ]; then
    log "$name fold $f: retry at batch 4 x accumulate 2"
    rm -rf "logs/$name/fold$f"
    bash scripts/vast/train.sh "$cfg" "$f" "$@" training.batch_size=4 training.accumulate_grad_batches=2 \
      > "logs/train_${name}_fold$f.txt" 2>&1
  fi
  [ -f "models/$name/fold$f.pt" ] || { log "$name fold $f FAILED"; return 1; }
  if [ ! -f "models/oof/${name}_oof_fold$f.csv" ]; then  # e.g. the final prediction pass died
    python -m src.models.train_pl "exp=$cfg" "fold=$f" paths.data_dir=data predict_only=true "$@" \
      > "logs/predict_${name}_fold$f.txt" 2>&1
  fi
  log "$name fold $f: $(python -c "import json; d=json.load(open('models/$name/fold${f}_metrics.json')); print('OOF', d.get('oof_auc'), 'gold', d.get('gold_auc'), 'min', d.get('minutes'))" 2>&1 | tail -1)"
  python scripts/upload_models.py --model "$slug" --exp "$exp" "models/$name/fold$f.pt" >> logs/upload.txt 2>&1 \
    && log "$name fold $f UPLOADED ($slug/$(echo "$exp" | tr 'A-Z' 'a-z')-fold$f)"
}

fetch cache_280_reader rsna-knee-cache-reader-p0 rsna-knee-cache-reader-p1 rsna-knee-cache-reader-p2 \
  rsna-knee-cache-reader-p3
fold reader_v2 0 rsna-knee-reader-v2 EXP-050

fetch cache_336_raptor rsna-knee-cache-raptor-p0 rsna-knee-cache-raptor-p1 rsna-knee-cache-raptor-p2
for f in 1 2 3 4; do fold raptor_repro "$f" rsna-knee-raptor-repro EXP-051; done
python scripts/merge_oof.py --exp raptor_repro --oof_dir models/oof --train data/raw/train.csv \
  --targets models/oof/report_targets.csv > logs/merge_raptor_repro.txt 2>&1 && log "raptor_repro 5-fold merged"

until [ -f reader_choice.txt ]; do log "waiting for reader_choice.txt (reader_v2 | reader_repro)"; sleep 1800; done
CHOICE=$(tr -d '[:space:]' < reader_choice.txt)
SLUG=$([ "$CHOICE" = reader_v2 ] && echo rsna-knee-reader-v2 || echo rsna-knee-reader-repro)
log "reader choice: $CHOICE -> $SLUG"
for f in 1 2 3 4; do fold "$CHOICE" "$f" "$SLUG" EXP-052; done
python scripts/merge_oof.py --exp "$CHOICE" --oof_dir models/oof --train data/raw/train.csv \
  --targets models/oof/report_targets.csv > "logs/merge_$CHOICE.txt" 2>&1 && log "$CHOICE 5-fold merged"
log CHAIN_DONE
