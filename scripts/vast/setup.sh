#!/usr/bin/env bash
# One-time setup of a rented GPU box (vast.ai, RTX 3090 24 GB; any CUDA PyTorch image).
#
#   Kaggle credentials first (either works):
#     scp ~/.kaggle/kaggle.json root@<vast-host>:~/.kaggle/kaggle.json
#     export KAGGLE_USERNAME=... KAGGLE_KEY=...
#
#   Get the code (no GitHub needed — it is published as the private Kaggle dataset <user>/rsna-knee-code):
#     pip install -q kaggle && kaggle datasets download <user>/rsna-knee-code -p rsna-knee --unzip
#     cd rsna-knee && bash scripts/vast/setup.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

if [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ] && [ ! -f ~/.kaggle/kaggle.json ]; then
  mkdir -p ~/.kaggle
  printf '{"username":"%s","key":"%s"}' "$KAGGLE_USERNAME" "$KAGGLE_KEY" > ~/.kaggle/kaggle.json
fi
[ -f ~/.kaggle/kaggle.json ] || { echo "missing ~/.kaggle/kaggle.json (see header)"; exit 1; }
chmod 600 ~/.kaggle/kaggle.json

# torch comes with the vast PyTorch image; install everything else the pipeline needs
python -m pip install -q --upgrade pip
python -m pip install -q "timm>=1.0.9" "lightning>=2.2" omegaconf pandas pyarrow scikit-learn \
  iterative-stratification opencv-python-headless pydicom pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg \
  tqdm psutil kaggle wandb
python - <<'PY'
import torch, timm, lightning
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("timm", timm.__version__, "| lightning", lightning.__version__)
PY
nproc; free -g | head -2; df -h . | tail -1
echo "next: bash scripts/vast/get_data.sh"
