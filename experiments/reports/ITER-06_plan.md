# ITER-06 plan — reproduce the two best public single-model pipelines, then merge (drafted 2026-10-07)

**Goal:** one single-fold model with the best CV–LB, then merge the best data-pipeline and model pieces of three
pipelines into it:
- ours (EXP-024/027);
- Raptor (dreaddevelopment);
- goodpjw's ConvNeXt-T reader.

Background: `ITER-05_plan.md` sections 5–6.

## Rules for the reproductions
- **Fold:** same `configs/folds_5.csv` fold 0 for all three (train folds 1–4, hold out fold 0, gold-58 never
  trained on).
- **Scoring:** OOF scored against `models/oof/report_targets.csv`, plus gold-58. This is the only way the CV
  numbers are comparable.
- **Submissions:** the fold-0 model of each is submitted alone (one submission each), giving one CV–LB point per
  pipeline.
- **Published scores are not fold-0 scores.** Raptor's published 0.932–0.935 is a model trained on **all** data.
  A fold-0 model sees 80% of it, so expect about 0.926–0.930. goodpjw's 1-fold score is 0.922.

## Phase A — reproductions (fold 0)

### A1 · EXP-041 — Raptor repro (data + model)
| Part | Setting (from Raptor's trainer and dataset pages) |
|---|---|
| Cache `cache_336_raptor` | 80 slices per study over **2–98%** of each series; slots sag-fluid 22 / sag-2nd 18 / cor-fluid 15 / cor-2nd 10 / axial 15; **140 mm crop stored at 336 px (0.417 mm/px)**; per-series 2–98 percentile window; slices sorted by IPP·normal; no laterality canonicalisation |
| Model | `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` @384 (bilinear upsample from 336); 3 adjacent slices as RGB, windows may cross slot borders; head LayerNorm → Linear(F,256)–Tanh–Dropout 0.2–Linear(256,12), softmax over windows per finding; no slot embedding |
| Training | soft BCE with pos_weight = clip((1−p)/p, 1, 10), p clipped to [0.03, 0.7]; AdamW backbone 3e-5 / head 1e-3, wd 0.02; OneCycle with 15% warm-up; 16 epochs; batch 8 (gradient checkpointing); k_train 12 random centres; **augmentation: ±10% gain only**; bf16; gradient clip 3; top-3 epoch weight average |
| Eval | **every window position (78)** |
| Labels | (a) ours; (b) Dread's public soft labels `dreaddevelopment/rsna-knee-labels`. Run (a) first: it isolates the pipeline from the labels |

**Cost:**
- Cache: about 40 GB uint8, so 2 Kaggle CPU notebooks (studies split in half), then a pack.
- Training: about 4 h on a 3090 (our CoAtNet took 177 min for 12 epochs at k 12; this is 16 epochs).
- Inference: CoAtNet at 78 windows is about 3× EXP-040, roughly 30–35 min for 1,300 studies (OOM-safe now).

### A2 · EXP-042 — goodpjw reader repro (their code, our fold)
| Part | Setting |
|---|---|
| Code | their `preprocess.py`, `knee.py`, `train.py`, `make_labels.py` (dataset `goodpjw2008/rsna-knee-2-5d-convnext-reader`), vendored under `third_party/goodpjw/` with attribution. Only paths and the fold column change (our `folds_5.csv` instead of report-grouped GroupKFold) |
| Cache | **per series**, every native slice (≤ 64), IOP-canonical in-plane orientation, 1–99.5 percentile. Their store is 0.4 mm/px at 384 px (≈ 90 GB for 24k series), too big for Kaggle outputs. Store at **0.55 mm/px, 280 px** (the same pixels the model sees after their 0.92 crop → 256) and use fov 1.0, so the model input is identical. ≈ 2–3 Kaggle CPU notebooks |
| Model | ConvNeXt-T, slot + slice-position embeddings, 2-layer transformer (dim 384), per-finding attention; k 8 windows per series in training (stratified bins), 12 at eval |
| Training | BCE on the mean of 4 public label tables (steven v4, pilkwang v2, lixin, qwen); AdamW 1e-4 / 5e-4, wd 0.02, cosine with 1 warm-up epoch; 14 epochs; batch 8; EMA 0.998 (evaluated with EMA); clip 3; GPU augmentation (rotation ±12°, scale ±12%, shift ±6%, gamma, gain) |
| Cost | caches on free Kaggle CPU; training about 2–3 h on a 3090 |

**Licence:** goodpjw's dataset is **Apache-2.0** (checked 2026-10-07), so vendoring is fine with the licence file
and an attribution header. Raptor is CC0.

## Phase B — merge (after A1/A2 are scored)
1. Pick the base: the best fold-0 CV with its LB point.
2. Port pieces from the other two **one at a time**, on the base's fold 0, keeping each only if it passes the gate
   (> +0.003 OOF):

| Area | Pieces |
|---|---|
| Data | 0.417 mm/px detail (D1), best-series choice + random series in training (D2), no duplicate slices (D3), stratified windows (D4), full in-plane canonicalisation (D5) |
| Model | slice-position embedding + transformer (goodpjw), ConvNeXt vs CoAtNet backbone |
| Training | EMA (D6), pos_weight BCE, minimal augmentation vs our affine stack |
| Labels | 4–5 sources (D7) |

3. Winner: train 5 folds **plus** one all-data model (the Raptor style), and submit both.

## Budget and timeline
- **vast credit:** $3.60. A1 + A2 take about 7 GPU-hours ≈ $1.5, so Phase B needs a top-up.
- **Kaggle:** cache notebooks are CPU-only (free). Submissions: 3 for Phase A; the rest of the week for Phase B.
- **Deadlines:** team merge 2026-10-15; final 2026-10-22.
