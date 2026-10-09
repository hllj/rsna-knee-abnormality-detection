#!/usr/bin/env bash
# ITER-08 on one RTX 3090: step A (EXP-064, raptor_v2_alldata on the 80-slice cache) then step B (EXP-065,
# raptor_v2_d96_alldata on the 96-slice native-384 cache built by rsna-knee-cache-r96-p0..p4). Both all-data (fold=-1),
# best-3 epochs by gold-58. Logs to logs_chain.txt; ends with ITER08_DONE.
#   tmux new -d -s chain 'bash scripts/vast/iter08_chain.sh > logs_chain_full.txt 2>&1'
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs models/oof data/labels
source scripts/vast/iter07_lib.sh
NW=$(( $(nproc) > 16 ? 16 : $(nproc) ))

[ -f data/labels/HYBRID_labels_scores.csv ] || {
  kaggle datasets download riadmohamed42/jev-knee-labels -f HYBRID_labels_scores.csv -p data/labels -q
  [ -f data/labels/HYBRID_labels_scores.csv.zip ] && (cd data/labels && unzip -o -q HYBRID_labels_scores.csv.zip \
    && rm HYBRID_labels_scores.csv.zip)
}

alldata() {  # $1 config, $2 Kaggle model slug, $3 EXP id
  local cfg=$1 slug=$2 exp=$3 name
  name=$(python -c "from src.config import get_config; print(get_config(['exp=$cfg']).exp_name)")
  if [ ! -f "models/$name/fold-1.pt" ]; then
    python -m src.models.train_pl "exp=$cfg" fold=-1 paths.data_dir=data "resources.num_workers=$NW" \
      > "logs/train_$name.txt" 2>&1
  fi
  if [ ! -f "models/$name/fold-1.pt" ]; then  # OOM fallback, same effective batch 16
    log "$name: retry at batch 4 x accumulate 4"; rm -rf "logs/$name/fold-1"
    python -m src.models.train_pl "exp=$cfg" fold=-1 paths.data_dir=data "resources.num_workers=$NW" \
      training.batch_size=4 training.accumulate_grad_batches=4 > "logs/train_$name.txt" 2>&1
  fi
  [ -f "models/$name/fold-1.pt" ] || { log "$name FAILED"; return 1; }
  log "$name: $(python -c "import json; d=json.load(open('models/$name/fold-1_metrics.json')); print('gold', d.get('gold_auc'), 'min', d.get('minutes'), 'best-3 by gold', d.get('topk_scores'))" 2>&1 | tail -1)"
  python scripts/upload_models.py --model "$slug" --exp "$exp" "models/$name/fold-1.pt" >> logs/upload.txt 2>&1 \
    && log "$name UPLOADED ($slug)"
}

fetch_raptor
alldata raptor_v2_alldata rsna-knee-raptor-v2 EXP-064
fetch cache_384_d96 rsna-knee-cache-r96-p0 rsna-knee-cache-r96-p1 rsna-knee-cache-r96-p2 rsna-knee-cache-r96-p3 \
  rsna-knee-cache-r96-p4
alldata raptor_v2_d96_alldata rsna-knee-raptor-v2-d96 EXP-065
log ITER08_DONE
