#!/usr/bin/env bash
# Box A, after its Raptor folds (iter07_hold.sh logs BOXA_RAPTOR_DONE): all-data Raptor (EXP-054, fold=-1,
# configs/exp/raptor_alldata.yaml) -> Kaggle Model rsna-knee-raptor-alldata/exp-054-fold-1, then BOXA_DONE.
#   tmux new -d -s alldata 'bash scripts/vast/iter07_boxA_alldata.sh > logs_alldata.txt 2>&1'
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs models/oof data/labels
source scripts/vast/iter07_lib.sh
NW=$(( $(nproc) > 16 ? 16 : $(nproc) ))
until grep -q "BOXA_RAPTOR_DONE" logs_chain.txt 2>/dev/null; do sleep 300; done
log "alldata: starting (Raptor folds done)"
[ -f data/labels/HYBRID_labels_scores.csv ] || {
  kaggle datasets download riadmohamed42/jev-knee-labels -f HYBRID_labels_scores.csv -p data/labels -q
  [ -f data/labels/HYBRID_labels_scores.csv.zip ] && (cd data/labels && unzip -o -q HYBRID_labels_scores.csv.zip \
    && rm HYBRID_labels_scores.csv.zip)
}
[ -f data/labels/HYBRID_labels_scores.csv ] || { log "alldata: JEV labels missing"; exit 1; }
train() {
  python -m src.models.train_pl exp=raptor_alldata fold=-1 paths.data_dir=data "resources.num_workers=$NW" "$@" \
    > logs/train_raptor_alldata.txt 2>&1
}
train || true
if [ ! -f models/raptor_alldata/fold-1.pt ]; then
  log "alldata: retry at batch 4 x accumulate 2"
  rm -rf logs/raptor_alldata/fold-1
  train training.batch_size=4 training.accumulate_grad_batches=2 || true
fi
if [ -f models/raptor_alldata/fold-1.pt ]; then
  log "alldata: $(python -c "import json; d=json.load(open('models/raptor_alldata/fold-1_metrics.json')); print('gold', d.get('gold_auc'), 'min', d.get('minutes'), 'last-3', d.get('topk_scores'))" 2>&1 | tail -1)"
  python scripts/upload_models.py --model rsna-knee-raptor-alldata --exp EXP-054 models/raptor_alldata/fold-1.pt \
    >> logs/upload.txt 2>&1 && log "raptor_alldata UPLOADED (rsna-knee-raptor-alldata/exp-054-fold-1)"
else
  log "alldata FAILED"
fi
log BOXA_DONE
