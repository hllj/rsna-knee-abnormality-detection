#!/usr/bin/env bash
# ITER-11 on one RTX 3090: "Raptor v3" = raptor_v3_d96_alldata_pl (EXP-081). Fetch the 96-slice cache and the JEV
# labels, map the multi-architecture teacher onto this config's label blend, train all-data (best-3 by gold-58), upload.
# Logs to logs_chain.txt; ends with ITER11_DONE.
#   tmux new -d -s chain 'bash scripts/vast/iter11_chain.sh > logs_chain_full.txt 2>&1'
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs models/oof data/labels
source scripts/vast/iter07_lib.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NW=$(( $(nproc) > 16 ? 16 : $(nproc) ))
CFG=raptor_v3_d96_alldata_pl SLUG=rsna-knee-raptor-v3-d96-pl EXP=EXP-081

[ -f data/labels/HYBRID_labels_scores.csv ] || {
  kaggle datasets download riadmohamed42/jev-knee-labels -f HYBRID_labels_scores.csv -p data/labels -q
  [ -f data/labels/HYBRID_labels_scores.csv.zip ] && (cd data/labels && unzip -o -q HYBRID_labels_scores.csv.zip \
    && rm HYBRID_labels_scores.csv.zip)
}
fetch cache_384_d96 rsna-knee-cache-r96-p0 rsna-knee-cache-r96-p1 rsna-knee-cache-r96-p2 rsna-knee-cache-r96-p3 \
  rsna-knee-cache-r96-p4

python scripts/make_pseudo_teacher.py --rank_file configs/teachers/cnx5_raptor5_rank.csv --exp "$CFG" \
  --out configs/teachers/cnx5_raptor5_qmap_jev.csv > logs/teacher.txt 2>&1 \
  || { log "teacher mapping FAILED"; log ITER11_DONE; exit 1; }
log "teacher: $(head -1 logs/teacher.txt)"

NAME=$(python -c "from src.config import get_config; print(get_config(['exp=$CFG']).exp_name)")
if [ ! -f "models/$NAME/fold-1.pt" ]; then
  python -m src.models.train_pl "exp=$CFG" fold=-1 paths.data_dir=data "resources.num_workers=$NW" \
    > "logs/train_$NAME.txt" 2>&1
fi
if [ ! -f "models/$NAME/fold-1.pt" ]; then  # OOM fallback, same effective batch 16
  log "$NAME: retry at batch 4 x accumulate 4"; rm -rf "logs/$NAME/fold-1"
  python -m src.models.train_pl "exp=$CFG" fold=-1 paths.data_dir=data "resources.num_workers=$NW" \
    training.batch_size=4 training.accumulate_grad_batches=4 > "logs/train_$NAME.txt" 2>&1
fi
if [ -f "models/$NAME/fold-1.pt" ]; then
  log "$NAME: $(python -c "import json; d=json.load(open('models/$NAME/fold-1_metrics.json')); print('gold', d.get('gold_auc'), 'min', d.get('minutes'), 'best-3 by gold', d.get('topk_scores'))" 2>&1 | tail -1)"
  python scripts/upload_models.py --model "$SLUG" --exp "$EXP" "models/$NAME/fold-1.pt" >> logs/upload.txt 2>&1 \
    && log "$NAME UPLOADED ($SLUG)" || log "$NAME UPLOAD FAILED"
else
  log "$NAME FAILED"
fi
log ITER11_DONE
