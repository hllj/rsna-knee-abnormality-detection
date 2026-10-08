#!/usr/bin/env bash
# Shared helpers for the ITER-07 two-box run (source it): fetch a multi-part cache, train + back up one fold.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$* $(date -u +%H:%M)" | tee -a logs_chain.txt; }

fetch() {  # $1 cache name, rest part notebooks
  local name=$1; shift
  until CACHE_PACKS="$*" CACHE_NAME=$name bash scripts/vast/get_data.sh >> "logs/get_$name.txt" 2>&1 \
        && [ "$(ls "data/processed/$name" | wc -l)" -ge 4407 ]; do
    log "retrying fetch of $name"; sleep 300
  done
  log "$name studies: $(ls "data/processed/$name" | wc -l)"
}
fetch_reader() { fetch cache_280_reader rsna-knee-cache-reader-p0 rsna-knee-cache-reader-p1 rsna-knee-cache-reader-p2 \
  rsna-knee-cache-reader-p3; }
fetch_raptor() { fetch cache_336_raptor rsna-knee-cache-raptor-p0 rsna-knee-cache-raptor-p1 rsna-knee-cache-raptor-p2; }

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
