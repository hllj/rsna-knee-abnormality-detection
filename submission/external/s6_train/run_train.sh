#!/usr/bin/env bash
# S6F0 fold-0 training entry for the systemd user unit rsna-s6f0-20261010 (see README.md for the launch command).
# Always resumes from the last epoch checkpoint (runs/train/ckpt/); exits 0 at once when runs/train/DONE_E_prior_s42_f0.json and the
# model exist. Waits (up to 2 h each) for the S6F0 lock and for a free GPU so it never overlaps the anchor precompute, a smoke or the
# OOF step. Normally started by run_all.sh (the unit's entry), which chains the OOF + gate after the DONE marker. Logs append to
# runs/train.log. Never pass --stop-after-epoch here (exit code 3 would count as a failure and be restarted).
set -uo pipefail
S=/home/danial/rsna-knee/research/gap-hunt-20261008/exec/S6F0
PY=/home/danial/rsna-knee/.venv312/bin/python
INIT=/home/danial/.cache/huggingface/hub/models--timm--convnext_small.fb_in22k_ft_in1k/snapshots/fe6b358aa409414cdb522ae3d43a2c9922f7cc1b/model.safetensors
OUT=${S6F0_OUT:-$S/runs/train}
LOG=${S6F0_LOG:-$S/runs/train.log}
mkdir -p "$OUT" "$(dirname "$LOG")"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# one S6F0 trainer at a time on this machine (they share the tmpfs work root /tmp/s6f0_work, wiped below), smokes included
exec 9>"$S/runs/.train.lock"
# review fix 10-10: WAIT for the lock (up to 2 h) instead of exiting 1, so a smoke or test holding it cannot burn the unit's restarts
if ! flock -n 9; then
  echo "$(TZ=Asia/Amman date '+%F %T') run_train.sh: $S/runs/.train.lock is held (another S6F0 trainer, smoke or OOF step), waiting up to 2 h" >> "$LOG"
  flock -w 7200 9 || { echo "$(TZ=Asia/Amman date '+%F %T') run_train.sh: lock still held after 2 h, giving up" >> "$LOG"; exit 75; }
fi
used=99999
for i in $(seq 1 120); do
  # review fix 10-10: a wedged driver must not hang the wrapper forever (no answer = busy)
  used=$(timeout 30 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ "${used:-99999}" -lt 1500 ] 2>/dev/null && break
  echo "$(TZ=Asia/Amman date '+%F %T') run_train.sh: GPU busy (${used} MiB used), waiting" >> "$LOG"
  sleep 60
done
[ "${used:-99999}" -lt 1500 ] 2>/dev/null || { echo "$(TZ=Asia/Amman date '+%F %T') run_train.sh: GPU still busy after 2 h, giving up" >> "$LOG"; exit 75; }
rm -rf /tmp/s6f0_work
echo "==== $(TZ=Asia/Amman date '+%F %T') run_train.sh start, pid $$, out $OUT" >> "$LOG"
exec "$PY" -u "$S/train_s6f0.py" \
  --pool "$S/data/pool_ok.json" --folds-json /home/danial/rsna-knee/public/raptor_train/folds_v0.json \
  --targets "$S/data/targets_s6f0.csv" --out "$OUT" --arms E_prior \
  --backbone convnext_small --init "$INIT" --seeds 1 --seed-base 42 --folds 5 --only-folds 0 \
  --expect-train "${S6F0_EXPECT_TRAIN:-3408}" --expect-holdout "${S6F0_EXPECT_HOLDOUT:-855}" \
  --epochs 16 --batch 2 --train-windows 64 --size 320 --views v4 --crop 288 --slices 3 --phys-mm 3.5 \
  --spatial-attn --grad-ckpt --lr 1e-4 --head-lr 1e-3 --wd 0.05 --aug crop --train-sampling uniform --ema 0 \
  --workers 8 --loader-timeout 600 --max-fail-frac 0.005 --log-every 200 --work-tmp /tmp/s6f0_work \
  --resume --no-final-predict --i-have-user-authorisation-to-finetune "$@" >> "$LOG" 2>&1
