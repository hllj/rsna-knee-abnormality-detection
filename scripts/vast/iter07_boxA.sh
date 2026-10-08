#!/usr/bin/env bash
# ITER-07 box A (two-box schedule, 2026-10-08). Takes over from the paused iter07_chain.sh while its reader_v2
# fold-0 training keeps running:
#   finish reader_v2 fold 0 (EXP-050) -> Raptor folds 1-2 (EXP-051) -> reader_v2 folds 1-2 (EXP-052, speculative:
#   the LB check of reader_v2 vs reader_repro happens after the 00:00 UTC submission reset)
#
#   kill -STOP <pid of "bash scripts/vast/iter07_chain.sh">
#   tmux new -d -s boxA 'bash scripts/vast/iter07_boxA.sh <that pid> > logs_boxA.txt 2>&1'
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs models/oof
source scripts/vast/iter07_lib.sh
CHAIN=${1:-}
log "boxA: waiting for reader_v2 fold 0 training (paused chain pid ${CHAIN:-none})"
while pgrep -f "src.models.train_p[l]" > /dev/null; do sleep 60; done
sleep 30  # let train.sh finish merge_oof
[ -n "$CHAIN" ] && kill -9 "$CHAIN" 2>/dev/null
fold reader_v2 0 rsna-knee-reader-v2 EXP-050
fetch_raptor
for f in 1 2; do fold raptor_repro "$f" rsna-knee-raptor-repro EXP-051; done
for f in 1 2; do fold reader_v2 "$f" rsna-knee-reader-v2 EXP-052; done
log BOXA_DONE
