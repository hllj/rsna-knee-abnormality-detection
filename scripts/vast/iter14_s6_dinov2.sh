#!/usr/bin/env bash
# ITER-14 on a vast box: S6 recipe (nartaa, discussion 749359) with a DINOv2 ViT-B backbone, all data, from the Kaggle-built
# px cache (no DICOMs downloaded). Run in a named tmux session from the repo root:
#   NSHARDS=2 bash scripts/vast/iter14_s6_dinov2.sh            # data + unpack + check + full run (resumes if re-run)
#   STEP=data|check|train NSHARDS=2 bash scripts/vast/iter14_s6_dinov2.sh
# The run itself is the speed probe: it logs windows/s and the ETA every 50 steps (runs/EXP-128/train.log).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PYTHON:-python}
K=${KAGGLE_BIN:-kaggle}
EXP=${EXP:-EXP-128}
NSHARDS=${NSHARDS:-2}
STEP=${STEP:-all}
D=${DATA_DIR:-data}
S6=submission/external/s6_train
OUT=runs/$EXP
PX=$D/s6_px320
mkdir -p "$D/s6_shards" "$OUT"

if [ "$STEP" = all ] || [ "$STEP" = data ]; then
  echo "== heliosli code (cloud_code, anatomy.pt) for the trainer imports"
  [ -f "$D/ortho/anatomy.pt" ] || "$K" datasets download heliosli/rsna-orthofoundation-fold0-epoch7-20261004 -p "$D/ortho" --unzip -q
  "$PY" "$S6/helios_sources.py" --dest "$S6/heliosli" --ortho "$D/ortho"
  echo "== px-cache shards from Kaggle (hlly34/rsna-knee-s6-cache-<k>)"
  for k in $(seq 0 $((NSHARDS - 1))); do
    [ -f "$D/s6_shards/$k/shard$k/summary.json" ] || "$K" kernels output "hlly34/rsna-knee-s6-cache-$k" -p "$D/s6_shards/$k" -q
  done
  [ -f "$PX/index.json" ] || "$PY" "$S6/unpack_px.py" --shards "$D"/s6_shards/*/shard* --out "$PX" --workers "$(nproc)"
  rm -rf "$D/s6_shards"                        # the unpacked cache is all the trainer reads
fi

# every non-gold cached study with targets trains (ALLDATA check); count it from the cache index instead of assuming 4,263
EXPECT=$("$PY" - "$PX/index.json" "$S6/data/targets_s6f0.csv" <<'PY'
import csv, json, sys
done = {u for u, v in json.load(open(sys.argv[1]))["done"].items() if v[1] > 0}
rows = list(csv.DictReader(open(sys.argv[2])))
print(sum(r["is_gold"] == "0" and r["StudyInstanceUID"] in done for r in rows))
PY
)
echo "training studies in the cache: $EXPECT"
ARGV=(--px "$PX" --targets "$S6/data/targets_s6f0.csv" --out "$OUT" --arms E_prior
  --backbone vit_base_patch14_dinov2.lvd142m --init timm --seeds 1 --seed-base 42 --folds 5 --only-folds 0
  --expect-train "$EXPECT" --expect-holdout 0 --all-data
  --epochs 16 --batch 2 --train-windows 64 --size 320 --views v4 --crop 280 --slices 3 --phys-mm 3.5
  --spatial-attn --grad-ckpt --lr 5e-5 --llrd 0.75 --head-lr 1e-3 --wd 0.05 --aug crop --train-sampling uniform --ema 0
  --workers "$(( $(nproc) > 12 ? 12 : $(nproc) ))" --loader-timeout 600 --log-every 50
  --resume --no-final-predict --i-have-user-authorisation-to-finetune)

if [ "$STEP" = all ] || [ "$STEP" = check ]; then
  echo "== targets / argv check (CPU)"
  CUDA_VISIBLE_DEVICES="" "$PY" -u "$S6/train_s6f0_all.py" "${ARGV[@]}" --check-targets 2>&1 | tail -20
fi
if [ "$STEP" = all ] || [ "$STEP" = train ]; then
  echo "== train $EXP (log: $OUT/train.log)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -u "$S6/train_s6f0_all.py" "${ARGV[@]}" 2>&1 | tee -a "$OUT/train.log"
fi
