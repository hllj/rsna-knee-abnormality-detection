#!/usr/bin/env bash
# Download everything training needs (~15 GB): competition csvs, public label sets, and the preprocessed
# cache built on Kaggle (EXP-001, re-packed into tar shards by the rsna-knee-cache-pack notebook).
# No DICOMs are downloaded.
#
#   bash scripts/vast/get_data.sh            # -> data/raw, data/labels, data/processed/cache_256
#   CACHE_PACK=rsna-knee-cache-dense-pack CACHE_NAME=cache_256_dense bash scripts/vast/get_data.sh   # dense cache
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PYTHON:-python}
USER_NAME=$("$PY" -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")
DATA=${DATA_DIR:-data}
COMP=rsna-knee-abnormality-detection
mkdir -p "$DATA/raw" "$DATA/labels" "$DATA/processed" "$DATA/_pack"

echo "== competition csvs"
for f in train.csv train_series.csv test.csv test_series.csv sample_submission.csv; do
  [ -f "$DATA/raw/$f" ] || kaggle competitions download -c "$COMP" -f "$f" -p "$DATA/raw" -q
  [ -f "$DATA/raw/$f.zip" ] && (cd "$DATA/raw" && unzip -o -q "$f.zip" && rm "$f.zip")
done

echo "== label sets"
kaggle datasets download stevenleehans/rsna-knee-llm-report-labels -p "$DATA/labels" --unzip -q
kaggle datasets download pilkwang/rsna-knee-llm-labels -p "$DATA/labels" --unzip -q
# reader_repro's 4-table mean (goodpjw): + lixin gpt56sol and laymond qwen3-8b
kaggle datasets download lixin73/rsna-knee-llm-report-labels-sol56 -p "$DATA/labels" --unzip -q
kaggle datasets download laymond/rsna-knee-abnormality-qwen3-8b-weak-labels -p "$DATA/labels" --unzip -q
rm -f "$DATA/labels"/*.py

# CACHE_PACKS = space-separated part notebooks filling one cache dir (ITER-06), e.g.
#   CACHE_PACKS="rsna-knee-cache-raptor-p0 rsna-knee-cache-raptor-p1 rsna-knee-cache-raptor-p2" CACHE_NAME=cache_336_raptor
CACHE_NAME=${CACHE_NAME:-cache_256}
CACHE_PACKS=${CACHE_PACKS:-${CACHE_PACK:-rsna-knee-cache-pack}}
if [ "$(ls "$DATA/processed/$CACHE_NAME" 2>/dev/null | wc -l)" -lt 4407 ]; then
  for P in $CACHE_PACKS; do
    echo "== cache shards from $USER_NAME/$P -> $DATA/processed/$CACHE_NAME"
    # one shard per request, checksum-verified, resumable (re-run the script if interrupted)
    "$PY" scripts/vast/fetch_cache.py --data "$DATA" --user "$USER_NAME" --pack "$P"
    rm -rf "$DATA/_pack_$P"
  done
fi
echo "cache studies: $(ls "$DATA/processed/$CACHE_NAME" | wc -l)"
"$PY" scripts/eval_labels.py "paths.data_dir=$DATA"
echo "next: bash scripts/vast/train.sh effnet_224 0"
