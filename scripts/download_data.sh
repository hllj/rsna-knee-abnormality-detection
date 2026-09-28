#!/usr/bin/env bash
# Small files only. The DICOMs are ~570 GB: preprocess on a machine/notebook that has them
# (see README "Data"), or start from a public preprocessed cache.
set -euo pipefail
C=rsna-knee-abnormality-detection
mkdir -p data/raw data/labels
for f in train.csv train_series.csv test.csv test_series.csv sample_submission.csv; do
  kaggle competitions download -c $C -f "$f" -p data/raw -q
done
# Public report-derived label sets (CC0). Scores vs the 58 gold studies in docs/COMPETITION_NOTES.md
for d in stevenleehans/rsna-knee-llm-report-labels pilkwang/rsna-knee-llm-labels \
         lixin73/rsna-knee-llm-report-labels-sol56 dreaddevelopment/rsna-knee-labels; do
  kaggle datasets download "$d" -p data/labels --unzip -q
done
rm -f data/labels/*.py
ls -la data/raw data/labels
# Full DICOMs (only where you have the disk):
#   kaggle competitions download -c $C -p data/raw && unzip -q data/raw/$C.zip -d data/raw
