#!/usr/bin/env bash
# Box B, after its Raptor folds (iter07_hold.sh logs BOXB_RAPTOR_DONE): reader_v3 fold 0 (EXP-053).
#   tmux new -d -s v3 'bash scripts/vast/iter07_boxB_v3.sh > logs_v3.txt 2>&1'
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs models/oof
source scripts/vast/iter07_lib.sh
until grep -q "BOXB_RAPTOR_DONE" logs_chain.txt 2>/dev/null; do sleep 300; done
log "v3: starting after Raptor folds"
fetch_reader
fold reader_v3 0 rsna-knee-reader-v3 EXP-053
log BOXB_DONE
