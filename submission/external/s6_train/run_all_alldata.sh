#!/usr/bin/env bash
# S6F0_ALL unit entry = the systemd user unit rsna-s6f0-all-20261011 (launch command: PREREG_ALL.md §6; NOT launched by the builder).
# One S6F0 model on ALL data (pool_ok.json's 4,321 minus the 58 gold = 4,263 studies, nothing held out), the GO run's recipe otherwise,
# then the gold-58 diagnostic.  Authorisation: Danial 2026-10-11 05:25 Amman ("yes to 1 and 2", "i give a go ahead for SKM-TEA anatomy
# network").
# Steps (every one is safe to repeat; a restart resumes):
#   0. fast exit 0 only when the DONE marker + model exist AND runs/ALL_COMPLETE.json exists.  ALL_COMPLETE.json is written by this
#      script alone, as its very last step (review 1 fix: a REPORT_ALL.json written by hand mid-run no longer ends the unit early).
#   1. pins (else 65): train_s6f0_all.py has the reviewed sha256; inputs.sha256 (the manifest of every training input: S6F0's
#      s6common.py / fly_data.py, data/targets_s6f0.csv / pool_ok.json / gold58.json, folds_v0.json, the convnext_small init, every
#      file of S6F0/heliosli/ and S6F0/anchors/; all frozen before the GO run started) has its reviewed sha256.
#      Precheck (CPU, ~15 s, only while training is unfinished), into a per-pid file that is then appended to runs/precheck.log:
#      `sha256sum -c --strict` of inputs.sha256, then the real trainer main() with --check-targets on the production argv (4,263
#      train, 0 held out, gold excluded, targets finite).  A failure is deterministic -> 65 (RestartPreventExitStatus, not retried).
#   2. own lock runs/.all.lock (one S6F0_ALL instance; waits up to 2 h, then 75).
#   3-4 only while training or gold-58 is pending (GPU work left; the CPU report alone needs neither):
#   3. wait while the LR queue unit rsna-s6f0-lr-20261011 is anything but inactive / failed / unknown: active, activating (its
#      auto-restart gap), deactivating, ... and also an empty answer or a systemctl error (fail closed).  Poll 60 s, up to 24 h, then 75.
#   4. the S6F0 trainer lock S6F0/runs/.train.lock, opened READ-ONLY exactly as lr/lr_lib.sh take_lock does (flock needs no write
#      access; the file is never created, truncated or touched), held to the end: every S6F0 trainer wipes the shared tmpfs work root
#      /tmp/s6f0_work at start, so no two may overlap.  Waits up to 2 h, then 75.  The LR unit is inactive by now, so this never
#      competes with the queue (its lock is held continuously from its take_lock to its exit, and we start only after that).
#   5. training (unless the DONE marker + model exist): free disk >= 5120 MiB first (df -BM; peak use ~2 GB: checkpoint + .prev + .tmp
#      + model; polled 5 min up to 6 h, then 75), then a free GPU (< 1500 MiB used, nvidia-smi with a 30 s timeout, no answer = busy;
#      polled 60 s up to 6 h, then 75), then train_s6f0_all.py with --resume (continues from runs/train/ckpt/; the trainer wipes
#      /tmp/s6f0_work itself).  A non-zero exit is passed on -> systemd restarts after 120 s and training resumes.
#   6. gold-58 (GPU ~3 min, diagnostic only, never a gate): a free GPU, then S6F0/oof_step.py (read-only, run with --models / --out in
#      S6F0_ALL) -> runs/gold58/gold58_s6f0_all.parquet; one in-process retry; an inference failure is recorded in its receipt and
#      reported, never restarted (only a GPU that stays busy for 6 h exits 75, so the step is tried again; training is not redone).
#   7. report_all.py -> runs/REPORT_ALL.json (loss curve vs the GO run, run_config vs GO, gold-58 macro vs the fold-0 model's 0.9306),
#      then runs/ALL_COMPLETE.json.  Exit 0 (a report_all.py failure is deterministic: 65).
# Exit: 0 done | 65 deterministic refusal (no restart) | 75 busy after the long waits (restarted) | other = trainer crash (restarted).
# Logs: this script -> stdout (the unit appends it to runs/run_all.log); trainer -> runs/train.log; precheck -> runs/precheck.log;
# gold-58 -> runs/gold58.log.
# ALL_* variables exist ONLY for the dry-run tests (tests/test_run_all_dry.sh); the unit sets none of them.
set -uo pipefail
AC=/home/danial/rsna-knee/research/gap-hunt-20261008/exec/S6F0_ALL        # code (this directory)
A=${ALL_ROOT:-$AC}                                                          # outputs
S=/home/danial/rsna-knee/research/gap-hunt-20261008/exec/S6F0              # READ-ONLY: data, heliosli copies, oof_step.py
PY=${ALL_PY:-/home/danial/rsna-knee/.venv312/bin/python}
INIT=/home/danial/.cache/huggingface/hub/models--timm--convnext_small.fb_in22k_ft_in1k/snapshots/fe6b358aa409414cdb522ae3d43a2c9922f7cc1b/model.safetensors
TRAINER=$AC/train_s6f0_all.py
TRAINER_SHA256=ae18d2e11de9a16202934787a9428ecdcefc0543d7944a75182926fb53965967
INPUTS=$AC/inputs.sha256
INPUTS_SHA256=77b6ce7a3f1fcdcd7deee0bbb88faff857b4bb9efdcb19860698b65649aa3ce2
OUT=$A/runs/train
LOG=$A/runs/train.log
GOLD=$A/runs/gold58
DONE=$OUT/DONE_E_prior_s42_fall.json
MODEL=$OUT/models/E_prior_s42_fall.pt
COMPLETE=$A/runs/ALL_COMPLETE.json
WAIT_UNIT=${ALL_WAIT_UNIT:-rsna-s6f0-lr-20261011}
SYSTEMCTL=${ALL_SYSTEMCTL:-systemctl}
POLL=${ALL_POLL:-60}
MAXW=${ALL_MAX_WAIT_S:-86400}
S6LOCK=${ALL_S6F0_LOCK:-$S/runs/.train.lock}
LOCK_WAIT=${ALL_LOCK_WAIT_S:-7200}
NVSMI=${ALL_NVIDIA_SMI:-nvidia-smi}
GPU_POLL=${ALL_GPU_POLL:-60}
GPU_WAIT_N=${ALL_GPU_WAIT_N:-360}
MIN_FREE_MB=${ALL_MIN_FREE_MB:-5120}
DISK_POLL=${ALL_DISK_POLL:-300}
DISK_WAIT_N=${ALL_DISK_WAIT_N:-72}
GOLD_RETRY_SLEEP=${ALL_GOLD_RETRY_SLEEP:-60}

# The production argv = S6F0/run_train.sh's, except --out, --expect-train / --expect-holdout and --all-data (tests/test_argv_diff.py).
ARGV=(--pool "$S/data/pool_ok.json" --folds-json /home/danial/rsna-knee/public/raptor_train/folds_v0.json
  --targets "$S/data/targets_s6f0.csv" --out "$OUT" --arms E_prior
  --backbone convnext_small --init "$INIT" --seeds 1 --seed-base 42 --folds 5 --only-folds 0
  --expect-train 4263 --expect-holdout 0 --all-data
  --epochs 16 --batch 2 --train-windows 64 --size 320 --views v4 --crop 288 --slices 3 --phys-mm 3.5
  --spatial-attn --grad-ckpt --lr 1e-4 --head-lr 1e-3 --wd 0.05 --aug crop --train-sampling uniform --ema 0
  --workers 8 --loader-timeout 600 --max-fail-frac 0.005 --log-every 200 --work-tmp /tmp/s6f0_work
  --resume --no-final-predict --i-have-user-authorisation-to-finetune)
if [ "${ALL_PRINT_ARGV:-}" = 1 ]; then printf '%s\n' "${ARGV[@]}"; exit 0; fi

export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
ts() { TZ=Asia/Amman date '+%F %T'; }
say() { echo "$(ts) run_all.sh: $*"; }
trained() { [ -f "$DONE" ] && [ -f "$MODEL" ]; }
gold_tried() { [ -f "$GOLD/gold58_s6f0_all.receipt.json" ]; }
mkdir -p "$A/runs"
say "==== start, pid $$, outputs $A/runs"

# 0. everything done?  (ALL_COMPLETE.json is written only by step 7 below, never by report_all.py)
if trained && [ -f "$COMPLETE" ]; then say "training done and runs/ALL_COMPLETE.json exists: nothing to do"; exit 0; fi

# 1. pins + CPU precheck
got=$(sha256sum "$TRAINER" 2>/dev/null | cut -d' ' -f1)
[ "$got" = "$TRAINER_SHA256" ] || { say "train_s6f0_all.py sha256 ${got:-missing} is not the reviewed $TRAINER_SHA256: refusing (65)"; exit 65; }
got=$(sha256sum "$INPUTS" 2>/dev/null | cut -d' ' -f1)
[ "$got" = "$INPUTS_SHA256" ] || { say "inputs.sha256 sha256 ${got:-missing} is not the reviewed $INPUTS_SHA256: refusing (65)"; exit 65; }
if ! trained; then
  pc="$A/runs/precheck.$$.log"
  { echo "==== $(ts) pid $$ precheck 1/2: pinned training inputs (sha256sum -c --strict $INPUTS, $(wc -l < "$INPUTS") files)"
    sha256sum -c --strict --quiet "$INPUTS"; } > "$pc" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    cat "$pc" >> "$A/runs/precheck.log"; rm -f "$pc"
    say "precheck FAILED: a training input differs from inputs.sha256 (exit $rc; see $A/runs/precheck.log): refusing (65)"; exit 65
  fi
  echo "==== $(ts) pid $$ precheck 2/2: --check-targets on the production argv (CPU)" >> "$pc"
  CUDA_VISIBLE_DEVICES="" "$PY" -B -u "$TRAINER" "${ARGV[@]}" --check-targets >> "$pc" 2>&1
  rc=$?
  okline=$(grep '^ALLDATA: train 4263 ' "$pc" | tail -n 1)
  if [ $rc -ne 0 ] || [ -z "$okline" ] || ! grep '^check-targets OK' "$pc" > /dev/null; then
    cat "$pc" >> "$A/runs/precheck.log"; rm -f "$pc"
    say "precheck FAILED (exit $rc; see $A/runs/precheck.log): refusing (65)"; exit 65
  fi
  cat "$pc" >> "$A/runs/precheck.log"; rm -f "$pc"
  say "precheck OK: inputs match inputs.sha256; $okline"
fi

# 2. own lock
exec 9>"$A/runs/.all.lock"
if ! flock -n 9; then
  say "$A/runs/.all.lock is held (another S6F0_ALL run), waiting up to ${LOCK_WAIT} s"
  flock -w "$LOCK_WAIT" 9 || { say "own lock still held after ${LOCK_WAIT} s: giving up (75)"; exit 75; }
fi
say "holding $A/runs/.all.lock"

gpu_wait() {
  local used=99999 i
  for i in $(seq 1 "$GPU_WAIT_N"); do
    used=$(timeout 30 $NVSMI --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || true)
    [ "${used:-99999}" -lt 1500 ] 2>/dev/null && { say "GPU free (${used} MiB used)"; return 0; }
    { [ "$i" -eq 1 ] || [ $((i % 30)) -eq 0 ]; } && say "GPU busy (${used:-no answer} MiB used), waiting (poll $i of $GPU_WAIT_N)"
    sleep "$GPU_POLL"
  done
  say "GPU still busy after $GPU_WAIT_N polls: giving up (75)"
  return 75
}

if ! trained || ! gold_tried; then
  # 3. wait for the LR queue unit; only an explicit inactive / failed / unknown ends the wait (fail closed on errors / no answer)
  unit_state() { local st; st=$($SYSTEMCTL --user is-active "$WAIT_UNIT" 2>/dev/null); echo "${st%%$'\n'*}"; }
  waited=0
  while :; do
    st=$(unit_state)
    case "$st" in inactive|failed|unknown) break;; esac
    [ "$waited" -eq 0 ] && say "unit $WAIT_UNIT is ${st:-<no answer from systemctl>}: waiting for it to finish (poll ${POLL} s, up to ${MAXW} s)"
    [ "$waited" -ge "$MAXW" ] && { say "unit $WAIT_UNIT still ${st:-<no answer from systemctl>} after ${MAXW} s: giving up (75)"; exit 75; }
    [ "$waited" -gt 0 ] && [ $((waited % 3600)) -lt "$POLL" ] && say "still waiting for $WAIT_UNIT (${st:-<no answer>}, ${waited} s)"
    sleep "$POLL"; waited=$((waited + POLL))
  done
  say "unit $WAIT_UNIT is '$st' (waited ${waited} s)"

  # 4. the shared S6F0 trainer lock, read-only
  [ -e "$S6LOCK" ] || { say "S6F0 lock file $S6LOCK missing: refusing (65)"; exit 65; }
  exec 8<"$S6LOCK"
  if ! flock -n 8; then
    say "$S6LOCK is held (an S6F0 trainer / OOF / smoke), waiting up to ${LOCK_WAIT} s"
    flock -w "$LOCK_WAIT" 8 || { say "S6F0 lock still held after ${LOCK_WAIT} s: giving up (75)"; exit 75; }
  fi
  say "holding the S6F0 lock $S6LOCK (read-only open)"
else
  say "training and gold-58 done earlier: only the CPU report is left (no unit / S6F0 lock / GPU wait)"
fi

# 5. training: disk first, then the GPU, then the trainer
if ! trained; then
  free=0
  for i in $(seq 1 "$DISK_WAIT_N"); do
    free=$(df -BM --output=avail "$A" 2>/dev/null | tail -1 | tr -dc '0-9')
    [ "${free:-0}" -ge "$MIN_FREE_MB" ] && break
    { [ "$i" -eq 1 ] || [ $((i % 12)) -eq 0 ]; } && say "only ${free:-?} MiB free, need ${MIN_FREE_MB} MiB: waiting (poll $i of $DISK_WAIT_N)"
    sleep "$DISK_POLL"
  done
  [ "${free:-0}" -ge "$MIN_FREE_MB" ] || { say "still only ${free:-?} MiB free: giving up (75)"; exit 75; }
  say "disk ${free} MiB free (need ${MIN_FREE_MB})"
  gpu_wait || exit 75
  say "training (log $LOG)"
  echo "==== $(ts) run_all.sh training start, pid $$, out $OUT, trainer sha256 $TRAINER_SHA256, inputs.sha256 $INPUTS_SHA256" >> "$LOG"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -B -u "$TRAINER" "${ARGV[@]}" >> "$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then say "trainer exited $rc (systemd restarts; training resumes from $OUT/ckpt)"; exit $rc; fi
  trained || { say "trainer exited 0 without $DONE and $MODEL"; exit 1; }
  say "training done: $(tr -d '\n' < "$DONE" | cut -c1-400)"
else
  say "training done earlier ($DONE)"
fi

# 6. gold-58 diagnostic (never a gate; never restarted)
if ! gold_tried; then
  for att in 1 2; do
    gpu_wait || { say "gold-58 not run yet: GPU busy (75, restarted; training is not redone)"; exit 75; }
    say "gold-58 inference, attempt $att of 2 (log $A/runs/gold58.log)"
    echo "==== $(ts) gold-58 attempt $att" >> "$A/runs/gold58.log"
    ( cd "$A" && env -u PYTORCH_CUDA_ALLOC_CONF "$PY" -B -u "$S/oof_step.py" --models "$OUT/models" --out "$GOLD" --workers 3 \
        --sets gold58:gold58_s6f0_all --required "" --expect-epochs 16 --expect-train 4263 ) >> "$A/runs/gold58.log" 2>&1
    rc=$?
    say "oof_step.py exited $rc"
    [ $rc -eq 0 ] && [ -f "$GOLD/gold58_s6f0_all.parquet" ] && break
    [ "$att" -eq 1 ] && sleep "$GOLD_RETRY_SLEEP"
  done
else
  say "gold-58 done earlier ($GOLD/gold58_s6f0_all.receipt.json)"
fi

# 7. report (CPU), then the completion marker (the only thing step 0 trusts)
CUDA_VISIBLE_DEVICES="" "$PY" -B "$AC/report_all.py" --root "$A"
rc=$?
[ $rc -eq 0 ] || { say "report_all.py failed (exit $rc, deterministic): refusing (65); rerun by hand"; exit 65; }
msha=$(sha256sum "$MODEL" | cut -d' ' -f1)
printf '{"complete": "%s", "by": "run_all.sh pid %s", "model": "%s", "model_sha256": "%s", "trainer_sha256": "%s", "inputs_sha256": "%s", "report": "%s"}\n' \
  "$(TZ=Asia/Amman date -Iseconds)" "$$" "$MODEL" "$msha" "$TRAINER_SHA256" "$INPUTS_SHA256" "$A/runs/REPORT_ALL.json" > "$COMPLETE.tmp" \
  && mv -f "$COMPLETE.tmp" "$COMPLETE" || { say "could not write $COMPLETE"; exit 1; }
say "==== done: $A/runs/REPORT_ALL.json, $COMPLETE (model sha256 $msha)"
exit 0
