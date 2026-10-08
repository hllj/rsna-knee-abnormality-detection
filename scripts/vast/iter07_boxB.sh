#!/usr/bin/env bash
# ITER-07 box B (two-box schedule, 2026-10-08): Raptor folds 3-4 (EXP-051), then reader_v2 folds 3-4 (EXP-052).
#
#   tmux new -d -s boxB 'bash scripts/vast/iter07_boxB.sh > logs_boxB.txt 2>&1'
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs models/oof
source scripts/vast/iter07_lib.sh
[ -f models/oof/report_targets.csv ] || { log "missing models/oof/report_targets.csv (EXP-024 ruler)"; exit 1; }
fetch_raptor
for f in 3 4; do fold raptor_repro "$f" rsna-knee-raptor-repro EXP-051; done
fetch_reader
for f in 3 4; do fold reader_v2 "$f" rsna-knee-reader-v2 EXP-052; done
log BOXB_DONE
