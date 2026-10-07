#!/usr/bin/env bash
# ITER-06 on one rented RTX 3090: Raptor repro (EXP-041) then reader repro (EXP-042), fold 0 each.
#
#   tmux new -d -s chain 'bash scripts/vast/iter06_chain.sh'      # progress: tail -f logs_chain.txt
#
# Before: setup.sh done, models/oof/report_targets.csv = EXP-024's (the common OOF ruler), Kaggle creds.
# Each step logs one line to logs_chain.txt; the reader cache is fetched once its 4 part notebooks finish.
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs models/oof
# the probe's 20.6 GiB at batch 8 still OOMed in Lightning (a 3.38 GiB block did not fit): fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$* $(date -u +%H:%M)" | tee -a logs_chain.txt; }
[ -f models/oof/report_targets.csv ] || { log "missing models/oof/report_targets.csv (EXP-024 ruler)"; exit 1; }

probe() {  # $1 config, rest batch sizes -> echo "BS ACC" (largest batch with <= 22.5 GiB peak, else 4 x 2)
  local cfg=$1; shift
  python scripts/mem_probe.py "exp=$cfg" --batches "$@" > "logs/probe_$cfg.txt" 2>&1
  python - "$cfg" <<'PY'
import re, sys
best = None
for line in open(f"logs/probe_{sys.argv[1]}.txt"):
    m = re.match(r"batch\s+(\d+) .*peak ([\d.]+)", line)
    if m and float(m.group(2)) <= 22.5:
        best = max(best or 0, int(m.group(1)))
print(f"{best} 1" if best and best >= 8 else "4 2")
PY
}

fetch() {  # $1 cache name, rest part notebooks; retries until every part has finished and all 4,407 studies exist
  local name=$1; shift
  until CACHE_PACKS="$*" CACHE_NAME=$name bash scripts/vast/get_data.sh >> "logs/get_$name.txt" 2>&1 \
        && [ "$(ls "data/processed/$name" | wc -l)" -ge 4407 ]; do
    log "waiting for $name parts"; sleep 600
  done
  log "$name studies: $(ls "data/processed/$name" | wc -l)"
}

# ---- EXP-041 Raptor repro
fetch cache_336_raptor rsna-knee-cache-raptor-p0 rsna-knee-cache-raptor-p1 rsna-knee-cache-raptor-p2
read BS ACC < <(probe raptor_repro 6 8)
log "raptor_repro batch $BS x accumulate $ACC ($(grep -h batch logs/probe_raptor_repro.txt | tr '\n' ';'))"
train_retry() {  # $1 config, $2 batch, $3 accumulate: on failure retry once at 4 x 2 (same effective batch)
  bash scripts/vast/train.sh "$1" 0 "training.batch_size=$2" "training.accumulate_grad_batches=$3" \
    > "logs/train_$1.txt" 2>&1 && return 0
  log "$1 FAILED at batch $2 x $3 ($(grep -o 'OutOfMemoryError' "logs/$1_fold0.txt" | head -1)); retry 4 x 2"
  rm -rf "logs/$1/fold0"
  bash scripts/vast/train.sh "$1" 0 training.batch_size=4 training.accumulate_grad_batches=2 \
    > "logs/train_$1.txt" 2>&1 || log "$1 FAILED again"
}
[ -f models/raptor_repro/fold0.pt ] || train_retry raptor_repro "$BS" "$ACC"
[ -f models/raptor_repro/fold0.pt ] && python scripts/upload_models.py --model rsna-knee-raptor-repro --exp EXP-041 \
  models/raptor_repro/fold0.pt >> logs/upload.txt 2>&1 && log "RAPTOR_UPLOADED"

# ---- EXP-042 reader repro
fetch cache_280_reader rsna-knee-cache-reader-p0 rsna-knee-cache-reader-p1 rsna-knee-cache-reader-p2 \
  rsna-knee-cache-reader-p3
python scripts/mem_probe.py exp=reader_repro --batches 8 > logs/probe_reader_repro.txt 2>&1
log "reader_repro probe: $(grep -h batch logs/probe_reader_repro.txt | tr '\n' ';')"
[ -f models/reader_repro/fold0.pt ] || train_retry reader_repro 8 1
[ -f models/reader_repro/fold0.pt ] && python scripts/upload_models.py --model rsna-knee-reader-repro --exp EXP-042 \
  models/reader_repro/fold0.pt >> logs/upload.txt 2>&1 && log "READER_UPLOADED"
log CHAIN_DONE
