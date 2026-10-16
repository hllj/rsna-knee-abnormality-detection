#!/usr/bin/env bash
# ITER-13b: DINOv2-S v2 (EXP-122, whole encoder, 20 epochs) on GPU 0 after EXP-120 missed the gold gate (0.841);
# reuses iter13_newfam.sh's run() (sourced with RUN_ONLY=1). Logs to logs_chain.txt; ends with ITER13B_DONE.
#   tmux new -d -s chainb 'bash scripts/vast/iter13b_dinov2_full.sh > logs_chainb_full.txt 2>&1'
set -uo pipefail
cd "$(dirname "$0")/../.."
RUN_ONLY=1 source scripts/vast/iter13_newfam.sh
run 0 dinov2_s_d96_alldata_full rsna-knee-dinov2-s-d96 EXP-122
log ITER13B_DONE
