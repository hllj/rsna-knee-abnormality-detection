#!/usr/bin/env bash
# ITER-14 on a vast box: S6 recipe (nartaa, discussion 749359), all data, from the Kaggle-built px cache (no DICOMs downloaded).
# MODEL=convnext (default, owner 10-18): nartaa's ConvNeXt-S argv, seed 43. MODEL=dinov2: ViT-B variant (not run).
# Run in a named tmux session from the repo root:
#   bash scripts/vast/iter14_s6.sh                      # data + unpack + check + full run (resumes if re-run)
#   STEP=data|check|train bash scripts/vast/iter14_s6.sh
# The run itself is the speed probe: it logs windows/s and the ETA every 50 steps (runs/EXP-128/train.log).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PYTHON:-python}
K=${KAGGLE_BIN:-kaggle}
MODEL=${MODEL:-convnext}
EXP=${EXP:-EXP-128}
NSHARDS=${NSHARDS:-5}
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
if [ "$MODEL" = convnext ]; then
  # nartaa's production argv (run_all_alldata.sh / E_prior_s42_fall.run_config.json), only the seed differs (43: a second member
  # next to their public seed-42 weights). Init = the exact timm file they used (init_sha256 2457488b...).
  INIT=$("$PY" - <<'PY'
import hashlib
from huggingface_hub import hf_hub_download
p = hf_hub_download("timm/convnext_small.fb_in22k_ft_in1k", "model.safetensors")
h = hashlib.sha256(open(p, "rb").read()).hexdigest()
assert h == "2457488bec2ffcb2061b9e3fdd63360114549615e3e35ef6dc5c5efe549cbbc9", "init differs from nartaa's: " + h
print(p)
PY
)
  MARGS=(--backbone convnext_small --init "$INIT" --seed-base "${SEED:-43}" --crop 288 --lr 1e-4 --head-lr 1e-3)
else   # DINOv2 ViT-B (ITER-14 option B, not run): 14-px patches -> crop 280, ViT fine-tune LR with layer decay (untuned guess)
  MARGS=(--backbone vit_base_patch14_dinov2.lvd142m --init timm --seed-base "${SEED:-42}" --crop 280 --lr 5e-5 --llrd 0.75 --head-lr 1e-3)
fi
ARGV=(--px "$PX" --targets "$S6/data/targets_s6f0.csv" --out "$OUT" --arms E_prior "${MARGS[@]}"
  --seeds 1 --folds 5 --only-folds 0 --expect-train "$EXPECT" --expect-holdout 0 --all-data
  --epochs 16 --batch 2 --train-windows 64 --size 320 --views v4 --slices 3 --phys-mm 3.5
  --spatial-attn --grad-ckpt --wd 0.05 --aug crop --train-sampling uniform --ema 0
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
