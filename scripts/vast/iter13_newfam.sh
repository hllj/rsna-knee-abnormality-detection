#!/usr/bin/env bash
# ITER-13 on one 2-GPU box: two new families for the final blends, trained in parallel on the 96-slice cache (all data,
# best-3 epochs by gold-58), each uploaded as a Kaggle Model. Logs to logs_chain.txt; ends with ITER13_DONE.
#   GPU 0: dinov2_s_d96_alldata        (EXP-120) -> rsna-knee-dinov2-s-d96
#   GPU 1: radimagenet_r50_d96_alldata (EXP-121) -> rsna-knee-radimagenet-r50-d96
#   tmux new -d -s chain 'bash scripts/vast/iter13_newfam.sh > logs_chain_full.txt 2>&1'
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs models/oof data/labels data/pretrained
source scripts/vast/iter07_lib.sh
NW=$(( $(nproc) / 2 > 16 ? 16 : $(nproc) / 2 ))   # two trainings share the CPU

[ -f data/labels/HYBRID_labels_scores.csv ] || {
  kaggle datasets download riadmohamed42/jev-knee-labels -f HYBRID_labels_scores.csv -p data/labels -q
  [ -f data/labels/HYBRID_labels_scores.csv.zip ] && (cd data/labels && unzip -o -q HYBRID_labels_scores.csv.zip \
    && rm HYBRID_labels_scores.csv.zip)
}
[ -f data/pretrained/radimagenet_resnet50.pt ] || {
  kaggle datasets download shigechan/radimagenet-resnet50-official -f ResNet50.pt -p data/pretrained -q
  [ -f data/pretrained/ResNet50.pt.zip ] && (cd data/pretrained && unzip -o -q ResNet50.pt.zip && rm ResNet50.pt.zip)
  mv data/pretrained/ResNet50.pt data/pretrained/radimagenet_resnet50.pt
  echo "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734  data/pretrained/radimagenet_resnet50.pt" \
    | sha256sum -c - || log "RadImageNet sha256 MISMATCH"
}

[ "${RUN_ONLY:-0}" = 1 ] || fetch cache_384_d96 rsna-knee-cache-r96-p0 rsna-knee-cache-r96-p1 rsna-knee-cache-r96-p2 rsna-knee-cache-r96-p3 \
  rsna-knee-cache-r96-p4

run() {  # $1 GPU, $2 config, $3 Kaggle model slug, $4 EXP id
  local gpu=$1 cfg=$2 slug=$3 exp=$4 name
  name=$(python -c "from src.config import get_config; print(get_config(['exp=$cfg']).exp_name)")
  if [ ! -f "models/$name/fold-1.pt" ]; then
    CUDA_VISIBLE_DEVICES=$gpu python -m src.models.train_pl "exp=$cfg" fold=-1 paths.data_dir=data \
      "resources.num_workers=$NW" > "logs/train_$name.txt" 2>&1
  fi
  if [ ! -f "models/$name/fold-1.pt" ]; then  # OOM fallback, same effective batch 16
    log "$name: retry at batch 4 x accumulate 4"; rm -rf "logs/$name/fold-1"
    CUDA_VISIBLE_DEVICES=$gpu python -m src.models.train_pl "exp=$cfg" fold=-1 paths.data_dir=data \
      "resources.num_workers=$NW" training.batch_size=4 training.accumulate_grad_batches=4 > "logs/train_$name.txt" 2>&1
  fi
  if [ -f "models/$name/fold-1.pt" ]; then
    log "$name: $(python -c "import json; d=json.load(open('models/$name/fold-1_metrics.json')); print('gold', d.get('gold_auc'), 'min', d.get('minutes'), 'best-3 by gold', d.get('topk_scores'))" 2>&1 | tail -1)"
    python scripts/upload_models.py --model "$slug" --exp "$exp" "models/$name/fold-1.pt" >> "logs/upload_$name.txt" 2>&1 \
      && log "$name UPLOADED ($slug)" || log "$name UPLOAD FAILED"
  else
    log "$name FAILED"
  fi
}

[ "${RUN_ONLY:-0}" = 1 ] && return 0   # sourced by iter13b_dinov2_full.sh for run()

run 0 dinov2_s_d96_alldata rsna-knee-dinov2-s-d96 EXP-120 &
sleep 120   # stagger the two dataloader warm-ups
run 1 radimagenet_r50_d96_alldata rsna-knee-radimagenet-r50-d96 EXP-121 &
wait
log ITER13_DONE
