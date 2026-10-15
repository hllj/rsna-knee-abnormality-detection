# External stage: nartaa's 0.949 CoAtNet (option B, OAI fallback)

- `nartaa_0949_inference.py`: **verbatim** copy of the scored pipeline cell from
  https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror (author: Danial Zakaria / nartaa; built on
  dreaddevelopment's Raptor recipe). Kaggle notebook code, Apache-2.0 by default. Not modified.
- `nartaa_0949_check.py`: the notebook's checkpoint-verification cell (SHA-256 of the weights).
- Weights: Kaggle dataset `nartaa/rsna-knee-publication-swa-weights-20261007`
  (`raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt`). Model-use terms allow research/educational use incl. this
  competition. Trained with **OAI** external data (NIMH Data Archive, controlled access).
- **Rule status:** whether OAI-derived models are allowed is unanswered by the host (discussion 743416). This stage
  exists only as a fallback final pick (STRATEGY_2026-10-09.md, option B, chosen by the owner on 2026-10-09).

Required acknowledgement (OAI): Data and/or research tools used in the preparation of this work were obtained and
analyzed from the controlled access datasets distributed from the Osteoarthritis Initiative (OAI), a data repository
housed within the NIMH Data Archive (NDA). Dataset identifier(s): 10.15154/0hcg-f676.

## Second external stage: nartaa's 0.945 224-crop model (ITER-12)
- `nartaa_0945_inference.py`: **verbatim** scored pipeline cell of https://www.kaggle.com/code/nartaa/rsna-knee-0945-efficient-224crop (same author, same weights dataset, checkpoint `raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt`). `nartaa_0945_check.py`: its checkpoint check. Not modified.
- OAI status: allowed per the host ruling in discussion 741819 (2026-09-21) for data accessible without institutional sign-off; NDA lists OAI as institutional approval not required.

## heliosli R6 and R7 + flip TTA (2026-10-15)
- `heliosli_r6_stage.py`: cells 6-7 of heliosli's public 0.9573 notebook (R6 bundle `fe055406fad6a12f`, flip TTA).
  **Not usable:** it needs version 1 of `heliosli/rsna-knee-r6-frozen-fe055406`, and version 2 of that dataset now holds
  the R7 bundle under the same file names. Kaggle notebooks attach the latest version (a `/1` pin in the metadata is
  dropped on push), so the stage's bundle check fails. Kept for reference.
- `heliosli_r7flip_stage.py`: `heliosli_r7_stage.py` plus heliosli's own flip-TTA switch from that notebook
  (`FLIP_TTA = True`, `if manifest.get('flip_tta') or FLIP_TTA:`); the R7 worker (v4c) implements `--flip-tta`.
