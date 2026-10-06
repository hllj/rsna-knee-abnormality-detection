#!/usr/bin/env bash
# Ship local checkpoints to Kaggle and submit them (code competition = submission via a notebook):
#   1. upload *.pt + their metrics + offline pylibjpeg wheels as a new version of <user>/rsna-knee-weights
#   2. regenerate submission/inference to load from that dataset, tagged with the experiment id
#   3. push the notebook (GPU, internet off), wait for it to finish, submit its submission.csv
#   4. register the experiment in experiments/registry.csv (LB score: `python scripts/track.py lb` later)
#
#   bash scripts/vast/submit.sh EXP-006 "effnet_b0_224 fold0 on vast 3090" models/effnet_b0_224/fold0.pt
#   bash scripts/vast/submit.sh EXP-007 "teacher 5 folds" models/effnet_b0_224/fold*.pt
#   NO_SUBMIT=1 bash scripts/vast/submit.sh ...     # stop after the notebook run (check before spending a submission)
set -euo pipefail
cd "$(dirname "$0")/../.."
EXP_ID=${1:?experiment id, e.g. EXP-006}
MSG=${2:?short description}
shift 2
CKPTS=("$@")
[ ${#CKPTS[@]} -gt 0 ] || CKPTS=(models/*/fold*.pt)
PY=${PYTHON:-python}
USER_NAME=$("$PY" -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")
DS="$USER_NAME/rsna-knee-weights"
KERNEL="$USER_NAME/rsna-knee-inference"
COMP=rsna-knee-abnormality-detection

echo "== 1. weights dataset $DS (${#CKPTS[@]} checkpoints)"
STAGE=$(mktemp -d)
for c in "${CKPTS[@]}"; do
  d="$STAGE/$(basename "$(dirname "$c")")"; mkdir -p "$d"
  cp "$c" "$d/"; cp "${c%.pt}_metrics.json" "$d/" 2>/dev/null || true
done
cp models/oof/*_cv.json "$STAGE/" 2>/dev/null || true
# wheels for Kaggle's Python 3.12 image, installed offline with --no-deps by the notebook
"$PY" -m pip download -q -d "$STAGE/wheels" --only-binary=:all: --no-deps --python-version 3.12 \
  --platform manylinux2014_x86_64 --platform manylinux_2_28_x86_64 pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg
printf '{"title": "rsna-knee-weights", "id": "%s", "licenses": [{"name": "CC-BY-NC-4.0"}]}' "$DS" > "$STAGE/dataset-metadata.json"
# unique marker: the wait below must see THIS version, not an older one that has a same-named fold0.pt
# (EXP-017 was scored on the previous version's weights because of exactly that race)
MARKER="UPLOAD_${EXP_ID}_$(date +%s).txt"
echo "$EXP_ID $MSG" > "$STAGE/$MARKER"
EXPECTED=$(for c in "${CKPTS[@]}"; do echo "$(basename "$(dirname "$c")")/$(basename "$c")"; done | sort | tr '\n' ' ')
if kaggle datasets files "$DS" >/dev/null 2>&1; then
  kaggle datasets version -p "$STAGE" -r zip -m "$EXP_ID $MSG" --delete-old-versions
else
  kaggle datasets create -p "$STAGE" -r zip
fi
rm -rf "$STAGE"
echo "waiting for the dataset version to be processed..."
sleep 60
until kaggle datasets files "$DS" --page-size 200 2>/dev/null | grep -q "$MARKER"; do sleep 20; done
sleep 30

echo "== 2. inference notebook for $EXP_ID"
"$PY" submission/build_kernels.py --weights-dataset "$DS" --inference-id "$EXP_ID" \
  --inference-what "rank-average of ${#CKPTS[@]} checkpoint(s): $MSG"

echo "== 3. push + run on Kaggle"
OUT=$(kaggle kernels push -p submission/inference)
echo "$OUT"
VER=$(echo "$OUT" | grep -oE "version [0-9]+" | grep -oE "[0-9]+")
# the status endpoint can still report the previous version's COMPLETE right after a push
for i in $(seq 1 30); do kaggle kernels status "$KERNEL" 2>&1 | grep -qE "RUNNING|QUEUED" && break; sleep 10; done
until s=$(kaggle kernels status "$KERNEL" 2>&1); echo "$s" | grep -qE "COMPLETE|ERROR|CANCEL"; do sleep 60; done
echo "$s"
echo "$s" | grep -q COMPLETE || { kaggle kernels output "$KERNEL" -p /tmp/infer_log >/dev/null; tail -50 /tmp/infer_log/*.log; exit 1; }

# verify the run used exactly the uploaded checkpoints from this upload before spending a submission
rm -rf /tmp/infer_check && kaggle kernels output "$KERNEL" -p /tmp/infer_check --file-pattern "experiment.json" >/dev/null
GOT=$("$PY" -c "import json;e=json.load(open('/tmp/infer_check/experiment.json'));print(' '.join(sorted(e.get('ckpts',[]))),'|',' '.join(e.get('upload_ids',[])))")
echo "notebook used: $GOT"
[ "${GOT%% |*} " = "$EXPECTED" ] || { echo "ABORT: notebook checkpoints != uploaded ($EXPECTED)"; exit 1; }
echo "$GOT" | grep -q "$MARKER" || { echo "ABORT: notebook did not see upload marker $MARKER (stale dataset version)"; exit 1; }

"$PY" scripts/track.py add "$EXP_ID" --kind submit --kernel "$KERNEL" --version "$VER" \
  --notes "$MSG | $(printf '%s ' "${CKPTS[@]}")"
if [ -n "${NO_SUBMIT:-}" ]; then echo "NO_SUBMIT set: notebook v$VER ran fine, not submitted"; exit 0; fi

echo "== 4. submit version $VER"
kaggle competitions submit "$COMP" -k "$KERNEL" -v "$VER" -f submission.csv -m "$EXP_ID $MSG"
echo "submitted; scoring takes ~30-60 min. Then: python scripts/track.py lb"
