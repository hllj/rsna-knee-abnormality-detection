#!/usr/bin/env bash
# ITER-14 post-training on the box: wait for the DONE marker, score gold-58 (ours, nartaa's public seed 42, their rank mean),
# upload the model + logs as the private Kaggle dataset hlly34/rsna-knee-s6-cnx-s43, then write runs/EXP-128/POST_DONE.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PYTHON:-python}
OUT=runs/${EXP:-EXP-128}
SEED=${SEED:-43}
M=$OUT/models/E_prior_s${SEED}_fall.pt
until [ -f "$OUT/DONE_E_prior_s${SEED}_fall.json" ]; do sleep 60; done
echo "$(date) training done"
"$PY" scripts/vast/s6_gold.py --px data/s6_px320 --labels data/raw/train.csv --model ours_s$SEED="$M" \
  --model nartaa_s42=data/s6fall/E_prior_s42_fall.pt --out "$OUT/gold58.json" 2>&1 | grep -v Warning | tee "$OUT/gold58.txt"
UP=$(mktemp -d)
cp "$M" "$OUT/DONE_E_prior_s${SEED}_fall.json" "$OUT/run_config.json" "$OUT/gold58.json" "$OUT/gold58.txt" "$UP/"
grep -v "windows/s" "$OUT/train.log" > "$UP/train_epochs.log" 2>/dev/null || true
cat > "$UP/dataset-metadata.json" <<META
{"title": "rsna-knee-s6-cnx-s43", "id": "hlly34/rsna-knee-s6-cnx-s43", "licenses": [{"name": "CC-BY-NC-4.0"}]}
META
kaggle datasets create -p "$UP" -q 2>&1 | tail -2 || kaggle datasets version -p "$UP" -m "EXP-128" -q 2>&1 | tail -2
echo "$(date) uploaded" | tee "$OUT/POST_DONE"
