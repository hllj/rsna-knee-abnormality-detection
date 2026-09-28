#!/usr/bin/env bash
# Publish src/ configs/ scripts/ as a new version of the private dataset <user>/rsna-knee-code,
# which every Kaggle notebook of the pipeline attaches.
#   bash scripts/upload_code.sh "message"
set -euo pipefail
cd "$(dirname "$0")/.."
USER_NAME=$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")
STAGE=$(mktemp -d)
cp -R src configs scripts "$STAGE/"
find "$STAGE" -name "__pycache__" -type d -prune -exec rm -rf {} +
cat > "$STAGE/dataset-metadata.json" <<META
{"title": "rsna-knee-code", "id": "$USER_NAME/rsna-knee-code", "licenses": [{"name": "CC-BY-NC-4.0"}]}
META
MSG="${1:-code $(date +%Y%m%d-%H%M)}"
K="${KAGGLE_BIN:-kaggle}"
if "$K" datasets files "$USER_NAME/rsna-knee-code" >/dev/null 2>&1; then
  "$K" datasets version -p "$STAGE" -r zip -m "$MSG"
else
  "$K" datasets create -p "$STAGE" -r zip
fi
rm -rf "$STAGE"
