#!/usr/bin/env bash
# Push the S6 px-cache builder (submission/s6_cache/s6_cache.py) as its own Kaggle kernel.
#   bash scripts/s6_cache_push.sh pilot              -> hlly34/rsna-knee-s6-cache-pilot (50 studies, lossless size + JPEG error)
#   bash scripts/s6_cache_push.sh <shard> <nshards>  -> hlly34/rsna-knee-s6-cache-<shard> (that slice of data/pool_ok.json)
# Needs the code dataset refreshed first (scripts/upload_code.sh) so submission/external/s6_train/ is on Kaggle.
set -euo pipefail
cd "$(dirname "$0")/.."
K="${KAGGLE_BIN:-kaggle}"
USER_NAME=$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")
if [ "$1" = pilot ]; then SLUG=rsna-knee-s6-cache-pilot; P="0, 1, 50, True, 95"
else SLUG=rsna-knee-s6-cache-$1; P="$1, $2, 0, False, 95"; fi
D=$(mktemp -d)
python3 - "$D" "$P" <<'PY'
import sys
d, p = sys.argv[1:]
src = open("submission/s6_cache/s6_cache.py").read()
head = "SHARD, NSHARDS, LIMIT, PILOT, QUALITY = "
line = next(l for l in src.splitlines() if l.startswith(head))
open(d + "/s6_cache.py", "w").write(src.replace(line, head + p))
PY
cat > "$D/kernel-metadata.json" <<META
{"id": "$USER_NAME/$SLUG", "title": "$SLUG", "code_file": "s6_cache.py", "language": "python", "kernel_type": "script",
 "is_private": true, "enable_gpu": true, "enable_tpu": false, "enable_internet": false, "machine_shape": "NvidiaTeslaT4",
 "dataset_sources": ["$USER_NAME/rsna-knee-code", "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"],
 "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [], "model_sources": []}
META
"$K" kernels push -p "$D"
rm -rf "$D"
