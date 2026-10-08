#!/usr/bin/env bash
# Replaces a paused iter07_boxA/boxB.sh (2026-10-08): reader_v2 folds 1-4 are ON HOLD (fold 0 did not improve
# gold-58 and held-out CV together), so a box only finishes its Raptor folds.
#   kill -STOP <pid of "bash scripts/vast/iter07_boxX.sh">
#   tmux new -d -s hold 'bash scripts/vast/iter07_hold.sh <that pid> <box label> <raptor folds...>'
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs models/oof
source scripts/vast/iter07_lib.sh
PAUSED=$1; BOX=$2; shift 2
log "hold($BOX): reader_v2 folds on hold; waiting for the current step to finish (paused pid $PAUSED)"
while pgrep -f "src.models.train_p[l]" > /dev/null || pgrep -f "fetch_cach[e].py" > /dev/null \
      || pgrep -f "get_dat[a].sh" > /dev/null; do sleep 60; done
sleep 30
kill -9 "$PAUSED" 2>/dev/null
fetch_raptor
for f in "$@"; do fold raptor_repro "$f" rsna-knee-raptor-repro EXP-051; done
log "${BOX}_RAPTOR_DONE"
