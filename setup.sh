#!/bin/bash
# Environment + data bootstrap for the RSNA knee competition.
set -e
conda env create -f environment.yml || true
echo "Run: conda activate rsna-knee && pip install -e . && pre-commit install"
mkdir -p logs data/raw data/processed data/labels models
echo "Then: bash scripts/download_data.sh   (small csvs + public label sets)"
