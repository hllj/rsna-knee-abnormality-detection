# ITER-05 plan — what to try after EXP-027 (research drill, 2026-10-07)

**Starting point:** EXP-027, ConvNeXt-S @288 on the dense 88-slice cache, 5 folds rank-averaged.
Public LB **0.930**, pooled OOF 0.8781, gold-58 ensemble 0.9104.

**Fold-0 gate for every experiment below:** EXP-024, OOF 0.8745 / gold 0.9037. Keep a change only if fold-0
OOF improves by more than 0.003 (the noise floor) or gold rises clearly with OOF flat.

**Where we stand:**
- Public LB on 2026-10-07, 5,388 teams: 0.930 is about rank 2,840.
- Top 10% needs about 0.944. Rank 100 needs 0.952. Rank 10 needs 0.961.
- About 1,600 teams sit at ≥ 0.943, mostly forks of the public blend.

**Sources:**
- Competition discussions since 2026-09-26.
- The six newest or highest-scoring public notebooks.
- About 45 top write-ups from past RSNA competitions, found via kaggle.farid.one.

---

## 1. What the research says

### 1.1 The public 0.943–0.946 is a blend of other people's checkpoints, not a better single recipe
- **What the notebooks contain:** every top public notebook is inference-only. They blend about 40 checkpoints
  that other participants trained:
  - 20 DINOv2-S slot models (pilkwang)
  - DINOv3 "A5" × 5
  - RadImageNet ResNet-50 heads
  - Raptor CoAtNet × 4 views
  - 4 more CoAt readers
- **Run time:** 6.5–8 h of the 9 h limit.
- **Scores of independently trained families:**
  - Raptor CoAtNet baseline: 0.924.
  - goodpjw's own ConvNeXt-T reader (`goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944`):
    1 fold 0.922, 5 folds **0.930**.
  - Our 0.930 is level with these.
- **The only public single model above 0.94:** 0.942, a CoAtNet-RMLP-2 @336 *fine-tuned from Dread's public
  Raptor weights*. Its recipe isn't shared.
- **Prize risk:** prize-winning code must be reproducible. Training code for most of those weights is unpublished
  and their licences are unclear (thread 744056 has no host answer yet).
- **Decision:** take ideas only; we train every member ourselves.

### 1.2 But strong solo pipelines do reach 0.94–0.955 (forum, not verifiable)
- **Reported scores:**
  - Tucker Arrants: ResNet / EffNet @224–288, simple pooling, ~0.95.
  - Scott Willis: small ResNet, single fold, 0.949, 5-min inference.
  - CoolinLai: ResNet-50 @224, 5 folds, 0.954.
  - SpeedSci: EffNet-B4 / ResNet-50 / DINOv2, each 5 folds, 0.940–0.942.
  - Archit Konde: CoAtNet @224, single fold, 0.950.
- **What they say worked:**
  - Own single-prompt LLM labels (Qwen / Gemma, under $5); "better labels didn't move the LB".
  - Report-label CV with a 0.003 noise threshold.
  - **Ablating a standard + MRI-specific augmentation stack one change at a time.**
  - Then backbone and resolution.
  - "No attention mechanisms needed."
- **Label mixing with out-of-fold predictions (Archit Konde):** mixing truly out-of-fold image predictions into the
  labels, 50/50 or heavier toward the model, "moved right away".
- **Counter-evidence on label mixing:**
  - goodpjw: 5-fold LB 0.930 → 0.929 while OOF inflated 0.888 → 0.903.
  - Nicolai Karcher: CV lift, no LB lift.
  - Our own EXP-010 was worse.
  - So self-distillation is **not** a reliable lever for us.

### 1.3 Past RSNA winners: recurring patterns (Lumbar 2024, Abdominal 2023, Cervical 2022, ICH 2019, PE 2020, Mammo 2023, Aneurysm 2025)
| Pattern | Where it won | Reported gain |
|---|---|---|
| CNN per slice, then **sequence model (BiGRU / LSTM / transformer) over ordered slices**, then attention + max pooling; often a 2nd stage on frozen embeddings | ICH 1st/2nd, PE 1st, Cervical 1st–3rd, Lumbar 1st/4th, Aneurysm 6th | Lumbar 1st: plain 2.5D 0.37 → MIL 0.35 → +biLSTM/aux 0.33; Aneurysm 6th: max-pool 0.87 → sequence 0.88 |
| **CutMix / MixUp** | Lumbar 5th ("strongest regulariser", p=1.0), Abdominal 2nd, Aneurysm 6th | ≈ +0.01 |
| **OOF-disagreement noise removal**: drop cells where \|label − ensemble OOF\| ≥ 0.8 | Lumbar 2nd | +1% public and private |
| **Flip with left↔right target swap** (aug + TTA) | Lumbar 3rd "split LR", Aneurysm 2nd/6th | ≈ +0.01 |
| Auxiliary losses (per-view / pre-transformer heads) | Lumbar 1st/4th, Abdominal 1st | +0.01–0.03 (Abdominal, segmentation aux) |
| Model EMA + soft-positive cap / label smoothing | Mammography 1st | stability |
| ROI crops for small structures | all top Lumbar, Abdominal 2nd, Aneurysm 6th | large on Lumbar |
| Trust pooled OOF over many seeds; the best private score is often the strongest single pipeline, not the biggest blend | Brain-tumour 1st, Abdominal 2nd | robustness |

### 1.4 Small but concrete details from current notebooks
- **Per-finding window pooling at test time:** max over windows for Fracture, Contusion, both menisci and
  Baker's; mean of the top 2 for ACL and MCL; mixed with softmax pooling (β 6–10) at α 0.15–0.25.
  Also a reversed-slice-order view at weight 0.10.
- **Pixel spacing:**
  - Public members: 130 mm crop at 336 px (≈ 0.39 mm/px).
  - goodpjw: 0.40 mm/px.
  - Ours: 140 mm stored at 256 px (0.55 mm/px), upsampled to 288.
  - Meniscus, fracture and contusion are the findings most likely to lose from this.
- **Labels (DINOsaur v5):**
  - **Median** of the public label sources (not the mean), sharpened with T = 0.85.
  - Per-cell confidence weight.
  - Loss = 0.86 × soft BCE + 0.14 × pairwise ranking loss.
- **Augmentation (goodpjw reader):** rotate ±12°, scale ±12%, shift ±6%, gamma exp(±0.25), gain ±10%.
  Ours is already close.
- **Synovitis / Effusion (thread 745949):**
  - Experts call fluid positive more often than reports do: 78% of "small/equivocal effusion" reports are
    gold-positive.
  - Image models trained on the soft labels already beat the labels by a wide margin on Effusion (0.986 gold).
    Synovitis stays the weakest finding for everyone (~0.80–0.86).

## 2. What we already have vs what is missing

| Already in our pipeline | Missing |
|---|---|
| attention-MIL per finding | ordered sequence model over windows; per-finding max pooling at test |
| slot embedding | auxiliary per-slot heads |
| window dropout | CutMix / bag MixUp |
| soft blended labels + silence priors | label-noise filtering by OOF disagreement; 4th label source; ranking loss |
| affine + brightness/contrast + gamma | flip with medial↔lateral label swap; EMA (configured but 0) |
| top-3 epoch averaging | multi-family blend (CoAtNet @384 running) |
| 256 px store, 140 mm crop | ≥ 336 px store at 130 mm (≈ 0.39 mm/px) |

## 3. Ranked experiment queue (fold 0 first, one change each)

Costs are on one RTX 3090: about 75 min per fold for ConvNeXt-S @288 dense, roughly $0.25.

| # | EXP | Change | Why / evidence | Cost | Notes |
|---|---|---|---|---|---|
| 1 | EXP-030 | **EMA 0.999** + positive targets capped at 0.95 | Mammo 1st; knob exists | 1 fold | config only |
| 2 | EXP-031 | **Bag MixUp / CutMix** (mix two studies' windows per slot, labels mixed by fraction; p 0.5) | Lumbar 5th, Aneurysm 6th ≈ +0.01 | 1 fold + ~40 lines | `datasets.py` collate |
| 3 | EXP-032 | **Flip with label swap**: mirror coronal/axial images, reverse sagittal slice order, swap Medial↔Lateral Meniscus and OA labels, **mask MCL** (no lateral partner) on flipped samples; also as TTA | Lumbar 3rd, Aneurysm 2nd/6th ≈ +0.01; ~doubles data for 4 findings | 1 fold + ~60 lines | Must not break canonical right-knee geometry; unit test on one study |
| 4 | EXP-033 | **Noise filtering**: down-weight label cells where \|y − EXP-027 OOF\| ≥ 0.7 (weight 0.2), per finding | Lumbar 2nd +1% LB; *removing* noise instead of mixing in predictions (mixing failed for us and goodpjw) | 1 fold + per-cell loss weights | Score OOF only against the **unfiltered** `report_targets.csv` |
| 5 | EXP-034 | **Stage-2 sequence head on frozen embeddings**: per fold, dump ordered window embeddings for all 88 slices from the EXP-026/024 backbones; train a BiGRU (or 2-layer transformer) per slot plus neighbour deltas, then per-finding attention + max pooling | ICH / PE / Cervical / Lumbar winners | Dump ~15 min/fold; head trains in minutes | No backbone retraining; train each head only on its fold's training studies |
| 6 | EXP-035 | **Labels v2**: add `yunus_llm_labels` as a 4th source, median + T = 0.85 sharpening, + 0.14 pairwise ranking loss | DINOsaur v5, goodpjw (4 tables) | 1 fold | Check label-vs-gold AUC first (`scripts/eval_labels.py`) |
| 7 | EXP-036 | **Per-finding test-time pooling** (max for focal findings, top-2 for ligaments) + reversed-order view | public stack | inference only | Score on fold-0 OOF before any submission |
| 8 | — | **Second family** for the blend: CoAtNet-2 @384 dense (EXP-028, running) → if it passes the gate, 5 folds (~4 h/fold) | Blending different families is the documented lever (+0.004 to +0.013 in public version tables) | ~20 h | Cheaper alternatives: DINOv2-S slot model, RadImageNet ResNet-50 @224 |
| 9 | — | **Finer spacing**: rebuild dense cache at 336 px / 130 mm (Kaggle CPU kernel), train ConvNeXt @336 | public members' geometry | cache ~2 h CPU + ~1.4× train | Do after the cheap ablations |

**How to run the queue:**
1. Run #1–#4 back-to-back on one rented 3090. That is about 5 h, about $1.
2. Stack the winners and train 5 folds. Submit fold 0 and then the 5-fold, following the iteration loop.
3. #5 and #7 reuse the existing EXP-027 checkpoints and can run in parallel locally or on Kaggle.

**Not worth repeating:**
- Self-distillation (own OOF mixed 50/50 into the labels): EXP-010, plus goodpjw's 0.930 → 0.929.
- Bigger encoders of the same family (thread 735154: +0.001).
- More eval windows (EXP-016: neutral).
- Intensity-gain TTA (prvsiyan: 0.940 → 0.922).
- Training on gold-58 or selecting checkpoints on it (12 gold studies per fold is too noisy, and it breaks our
  "gold never trained on" rule).

## 4. Deadlines and budget
- **Deadlines:** team merge / entry 2026-10-15; final 2026-10-22. 15 days left, 5 submissions a day.
- **vast credit:** about $4.3 before CoAtNet finishes. The cheap queue (#1–#4) costs about $1; a second-family
  5-fold costs about $4.
- **Submissions:** keep one per day spare. Final picks:
  - (a) the best ConvNeXt 5-fold (or ConvNeXt + CoAtNet blend);
  - (b) the most diverse blend of our own families.
