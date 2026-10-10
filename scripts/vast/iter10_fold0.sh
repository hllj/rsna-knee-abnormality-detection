#!/usr/bin/env bash
# ITER-10 on one RTX 3090: fetch one cache, train fold 0 of one config (OOF + gold-58 metrics), upload it as a Kaggle
# Model variation. Logs to logs_chain.txt; ends with ITER10_DONE.
#   box A (direction 2): tmux new -d -s chain 'bash scripts/vast/iter10_fold0.sh convnext_288_dense_pl dense rsna-knee-convnext-s-288-dense-pl EXP-079 > logs_chain_full.txt 2>&1'
#   box B (direction 3): tmux new -d -s chain 'bash scripts/vast/iter10_fold0.sh convnext_288_native raptor rsna-knee-convnext-s-288-native EXP-080 > logs_chain_full.txt 2>&1'
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs models/oof data/labels
source scripts/vast/iter07_lib.sh
CFG=$1 CACHE=$2 SLUG=$3 EXP=$4

case "$CACHE" in
  dense) fetch cache_256_dense rsna-knee-cache-dense-pack ;;
  raptor) fetch_raptor ;;
  *) log "unknown cache $CACHE"; exit 1 ;;
esac
fold "$CFG" 0 "$SLUG" "$EXP" || true
log ITER10_DONE
