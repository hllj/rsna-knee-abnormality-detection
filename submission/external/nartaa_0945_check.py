# Authorized scoring run: verify the selected checkpoint before inference.
RUN_INFERENCE = True
if not RUN_INFERENCE:
    raise SystemExit("Static notebook only. Attach the exact checkpoint and competition, enable a T4 GPU, and set RUN_INFERENCE = True to run inference.")

import hashlib
from pathlib import Path
import torch
if not torch.cuda.is_available():
    raise RuntimeError("This scored pipeline requires CUDA; CPU inference is not enabled.")
_weight_name = 'raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt'
_candidates = [
    Path('/kaggle/input/rsna-knee-publication-swa-weights-20261007') / _weight_name,
    Path('/kaggle/input/datasets/nartaa/rsna-knee-publication-swa-weights-20261007') / _weight_name,
    Path('/kaggle/input/rsna-knee-raptor-alldata-d96-blendjev-oai-c08r224') / _weight_name,
    Path('/kaggle/input/datasets/nartaa/rsna-knee-raptor-alldata-d96-blendjev-oai-c08r224') / _weight_name,
]
_weights = next((p for p in _candidates if p.is_file()), None)
if _weights is None:
    raise FileNotFoundError("Attach the exact selected weights; no alternative model is substituted.")
with _weights.open('rb') as _f:
    assert hashlib.file_digest(_f, 'sha256').hexdigest() == '3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6', "Checkpoint hash differs"

print('[nartaa-224] checkpoint verified')
