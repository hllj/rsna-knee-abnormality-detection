#!/usr/bin/env bash
# ITER-12 on one RTX 3090: fetch one cache, train several folds of one config (each uploaded as a Kaggle Model
# variation right after it finishes). Logs to logs_chain.txt; ends with ITER12_DONE.
#   box A: tmux new -d -s chain 'bash scripts/vast/iter12_folds.sh convnext_288_dense_pl dense rsna-knee-convnext-s-288-dense-pl EXP-087 1 2 3 4 > logs_chain_full.txt 2>&1'
#   box B: tmux new -d -s chain 'bash scripts/vast/iter12_folds.sh reader_repro reader rsna-knee-reader-repro EXP-088 1 2 3 4 > logs_chain_full.txt 2>&1'
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs models/oof data/labels
source scripts/vast/iter07_lib.sh
CFG=$1 CACHE=$2 SLUG=$3 EXP=$4; shift 4

case "$CACHE" in
  dense) fetch cache_256_dense rsna-knee-cache-dense-pack ;;
  raptor) fetch_raptor ;;
  reader) fetch_reader ;;
  *) log "unknown cache $CACHE"; exit 1 ;;
esac
for f in "$@"; do
  fold "$CFG" "$f" "$SLUG" "$EXP" || true
done
log ITER12_DONE
