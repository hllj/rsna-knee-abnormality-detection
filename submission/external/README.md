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
