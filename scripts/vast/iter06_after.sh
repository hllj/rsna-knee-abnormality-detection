#!/usr/bin/env bash
# Takes over from a paused iter06_chain.sh while its Raptor training keeps running (2026-10-07).
# The running process predates the OOM-safe predict(): its final OOF / gold pass (78 windows x 4 studies of
# CoAtNet @384) would OOM, and the chain would then retrain at 4 x 2. Instead: wait for training to exit, rebuild
# the predictions with `predict_only=true`, merge, back up, then run the reader repro with the fixed code.
#
#   kill -STOP <pid of "bash scripts/vast/iter06_chain.sh">; tmux new -d -s after 'bash scripts/vast/iter06_after.sh <pid>'
# (pass the pid: a pattern match would also hit the tmux server, whose command line contains the script name)
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NW=$(( $(nproc) > 16 ? 16 : $(nproc) ))
log() { echo "$* $(date -u +%H:%M)" | tee -a logs_chain.txt; }

CHAIN=${1:-}
log "after: waiting for raptor_repro training (paused chain pid ${CHAIN:-none})"
while pgrep -f "src.models.train_p[l]" > /dev/null; do sleep 60; done
[ -n "$CHAIN" ] && kill -9 "$CHAIN" 2>/dev/null

finish() {  # $1 exp_name, $2 Kaggle model slug, $3 EXP id
  if [ ! -f "models/$1/fold0.pt" ]; then log "$1: no fold0.pt"; return 1; fi
  if [ ! -f "models/oof/$1_oof_fold0.csv" ] || [ ! -f "models/oof/$1_gold_fold0.csv" ]; then
    log "$1: rebuilding OOF / gold predictions (predict_only)"
    python -m src.models.train_pl "exp=$1" fold=0 paths.data_dir=data "resources.num_workers=$NW" predict_only=true \
      > "logs/predict_$1.txt" 2>&1 || log "$1 predict_only FAILED"
  fi
  python scripts/merge_oof.py --exp "$1" --oof_dir models/oof --train data/raw/train.csv \
    --targets models/oof/report_targets.csv > "logs/merge_$1.txt" 2>&1
  log "$1 metrics: $(python -c "import json; d=json.load(open('models/$1/fold0_metrics.json')); print('OOF', d.get('oof_auc'), 'gold', d.get('gold_auc'), 'min', d.get('minutes'))" 2>&1 | tail -1)"
  python scripts/upload_models.py --model "$2" --exp "$3" "models/$1/fold0.pt" >> logs/upload.txt 2>&1 \
    && log "$1 UPLOADED"
}

finish raptor_repro rsna-knee-raptor-repro EXP-041

# ---- EXP-042 reader repro (fixed code: OOM-safe predict)
until CACHE_PACKS="rsna-knee-cache-reader-p0 rsna-knee-cache-reader-p1 rsna-knee-cache-reader-p2 rsna-knee-cache-reader-p3" \
      CACHE_NAME=cache_280_reader bash scripts/vast/get_data.sh >> logs/get_cache_280_reader.txt 2>&1 \
      && [ "$(ls data/processed/cache_280_reader | wc -l)" -ge 4407 ]; do
  log "waiting for cache_280_reader parts"; sleep 600
done
log "cache_280_reader studies: $(ls data/processed/cache_280_reader | wc -l)"
if [ ! -f models/reader_repro/fold0.pt ]; then
  bash scripts/vast/train.sh reader_repro 0 > logs/train_reader_repro.txt 2>&1 \
    || log "reader_repro train.sh exit $? (checking outputs)"
fi
finish reader_repro rsna-knee-reader-repro EXP-042
log CHAIN_DONE
