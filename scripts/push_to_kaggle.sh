#!/usr/bin/env bash
# Upload code + weights as private Kaggle datasets, then push the inference kernel.
#   bash scripts/push_to_kaggle.sh models/coatnet2_384/fold*.pt models/effnet_b0_224/fold*.pt
# Then open the kernel on kaggle.com, "Save Version" and "Submit" (code competition).
set -euo pipefail
USER=$(python -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")
STAGE=$(mktemp -d)

# --- code dataset -----------------------------------------------------------
mkdir -p "$STAGE/code"
cp -R src configs scripts "$STAGE/code/"
find "$STAGE/code" -name "__pycache__" -prune -exec rm -rf {} +
cat > "$STAGE/code/dataset-metadata.json" <<META
{"title": "rsna-knee-code", "id": "$USER/rsna-knee-code", "licenses": [{"name": "CC-BY-NC-4.0"}]}
META
if kaggle datasets status "$USER/rsna-knee-code" >/dev/null 2>&1; then
  kaggle datasets version -p "$STAGE/code" -m "code $(git rev-parse --short HEAD 2>/dev/null || date +%s)" -r zip -q
else
  kaggle datasets create -p "$STAGE/code" -r zip -q
fi

# --- weights dataset (pass .pt files as args) --------------------------------
if [ "$#" -gt 0 ]; then
  mkdir -p "$STAGE/weights"
  for f in "$@"; do
    d="$STAGE/weights/$(basename "$(dirname "$f")")"; mkdir -p "$d"; cp "$f" "$d/"
  done
  cat > "$STAGE/weights/dataset-metadata.json" <<META
{"title": "rsna-knee-weights", "id": "$USER/rsna-knee-weights", "licenses": [{"name": "CC-BY-NC-4.0"}]}
META
  if kaggle datasets status "$USER/rsna-knee-weights" >/dev/null 2>&1; then
    kaggle datasets version -p "$STAGE/weights" -m "weights $(date +%Y%m%d-%H%M)" -r zip -q
  else
    kaggle datasets create -p "$STAGE/weights" -r zip -q
  fi
fi

# --- kernel ------------------------------------------------------------------
sed "s/hlly34/$USER/g" submission/kernel/kernel-metadata.json > "$STAGE/kernel-metadata.json"
cp submission/kernel/rsna-knee-inference.ipynb "$STAGE/"
echo "Waiting for datasets to process before pushing the kernel..."; sleep 60
kaggle kernels push -p "$STAGE"
