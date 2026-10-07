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

---

## 5. Single-model survey: public notebooks and forum claims (2026-10-07)

**Question:** is there a public notebook with a single model at ≥ 0.94, and how does it work?

**Answer: no.** 17 notebooks, selected from 657 by score rank and title, were read in full.
- Every public notebook at 0.941–0.946 runs the same community bundle of about 45 checkpoints:
  - DINOv2 ×20 and A5 ×5
  - RadImageNet heads
  - Raptor ×4 views and about 14 CoAtNets
- Titles mislead. "Pure CoAtNet 0946" is that bundle; its author's correction says it scores 0.943, and 0.946
  needed a private ConvNeXt mixed in at 10%.

### Best documented single-family public models
| Model | LB | Recipe | Code |
|---|---|---|---|
| **Raptor** (dreaddevelopment) — one CoAtNet-RMLP-2 run, **no folds**, trained on all 4,349 non-gold studies, epoch chosen on gold-58 | 0.924 (64 slices, 42 windows) → 0.927 (62 windows) → **0.932** ("fine spacing": 80 slices over 2–98%, 140 mm crop, 336 px, 78 windows) → **0.935** (v5 "maxspan", per sankurero's log) | `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` @384 from a 336 store; 3-slice windows; tanh-attention MIL per finding; soft BCE with pos_weight clip((1−p)/p, 1, 10); AdamW backbone 3e-5 / head 1e-3, wd 0.02, OneCycle (15% warm-up), 16 epochs, batch 8, k_train 12; **no augmentation except ±10% gain**; bf16, gradient clip 3; top-3 epoch weight average | trainer public (`dreaddevelopment/knee-mri-training-the-twelve-finding-model`); labels private |
| **goodpjw2008 ConvNeXt-T reader** | 1 fold 0.922; **3 folds 0.929**; 5 folds 0.930 | 0.4 mm/px, 384 → 256 px, ≤64 slices, 6 slots, 12 windows per series over 4–96%; ConvNeXt-T, then a 2-layer transformer over slice tokens (slot + position embeddings), then per-finding attention; mean of 4 public label tables, plain BCE, 14 epochs, folds grouped by report text; aug: rotation ±12°, scale ±12%, shift, gamma, gain, LR flip on coronal/axial | `train.py` and `make_labels.py` inside dataset `goodpjw2008/rsna-knee-2-5d-convnext-reader` |
| Ours, EXP-027 | **0.930** (5 folds) | see README | — |

Raptor's own model panel on gold (45 studies):

| Model | Gold AUC |
|---|---|
| CoAtNet @384 | 0.903 |
| CNN @336 | 0.883 |
| Swin-B @384 | 0.883 |
| ConvNeXt-B @384 | 0.875 |
| ConvNeXt-L @384 | 0.875 |
| EffNetV2-L @480 | 0.872 |
| MaxViT @384 | 0.844 |

Blending them gained only about +0.001 over CoAtNet alone on the LB.

### Forum-only claims of single models ≥ 0.94 (no code)
| Who | Claim | Hints given |
|---|---|---|
| Tanuki_boosting | 0.955 | single model, 5 folds; nothing else |
| CoolinLai | 0.954 | ResNet-50 @224, 5 folds |
| Musa Tur Farazi | 0.952 | 224 px single model |
| Tucker Arrants | low 0.95s | ResNet / EffNet @224–288, simple pooling, own Qwen labels, augmentation ablations one at a time, CV on report labels (0.003 threshold) |
| Archit Konde | 0.950 | CoAtNet @224 single fold; out-of-fold predictions mixed into labels (≥ 50/50 toward the model) |
| NguyenThanhNhan | 0.950 | Qwen3.5-2B VLM, LoRA + vision encoder @384, single fold, 3.5 h |
| Scott Willis | 0.949 | "smaller ResNet", single fold, own Gemma-4 labels (~0.89 vs gold), 5-min inference |
| Raymond Yuen | 0.949 | CoAtNet @288, full train; teachers on several weak-label sources, then pseudo-labels, then a student |
| SpeedSci | 0.940–0.942 | DINOv2 / EffNet-B4 / ResNet-50, 5 folds each; changing label version moved DINOv2 0.931 → 0.942 |
| Ali Ozan Memetoglu | 0.942 | single vision model on Claude-extracted labels |

**Common thread in the claims:** own LLM labels plus a simple, well-tuned CNN. Several credit pseudo-label rounds
from teachers trained on *different* label sources. That is not the same as one model distilling into itself,
which failed for us (EXP-010) and for goodpjw. None of these claims can be checked.

### What this adds to the queue
1. **Raptor-style full-data model (EXP-037).** Train on all non-gold studies for a fixed number of epochs
   (the fold-0 best epoch) with top-3 averaging.
   - Each 5-fold member sees 80% of the data; Raptor credits more data with +0.013 on gold.
   - Cost: 1 run (~90 min ConvNeXt / ~4 h CoAtNet). Check it on the LB only, since there is no held-out set.
   - Blend it with the 5-fold set.
2. **CoAtNet @384 dense (EXP-028, running)** already mostly follows Raptor's recipe: backbone 3e-5, head 1e-3,
   wd 0.02, k_train 12. Remaining differences:
   - We store at 256 px vs their 336.
   - We use affine + brightness/contrast augmentation vs their gain only.
   - We use plain soft BCE vs their pos_weight BCE.
   - Epoch 9: held-out 0.868 / gold 0.908, on track against the ConvNeXt dense bar (0.8745 / 0.9037).
3. **Teacher-ensemble pseudo-labels (EXP-038), not self-distillation.** Use the out-of-fold rank-average of
   *different* families (ConvNeXt dense + CoAtNet) as the teacher, mixed 50/50 into the soft labels; train a fresh
   student. This is Raymond Yuen's and Archit Konde's route. Score it only against unmixed `report_targets.csv`.

---

## 6. Data-pipeline deep dive: Raptor and goodpjw's reader vs ours (2026-10-07)

**Sources:**
- Raptor: inference and trainer notebooks, plus the dataset pages of its 10 corpus variants. Each page states the
  geometry and the public LB of a one-variable change.
- goodpjw's reader: the full code in `goodpjw2008/rsna-knee-2-5d-convnext-reader`
  (`preprocess.py`, `knee.py`, `train.py`, `make_labels.py`).
- Our pipeline: `src/dataset/dicom.py`, `src/dataset/datasets.py`.

### 6.1 Side by side
| | **Raptor** (CoAtNet-2 @384, single model 0.924–0.935) | **goodpjw reader** (ConvNeXt-T, 5 folds 0.930) | **Ours** (ConvNeXt-S, 5 folds 0.930) |
|---|---|---|---|
| Caching unit | per study, fixed 44 / 64 / 80 slots | **per series**, every native slice (≤ 64, else linspace) | per study, fixed 72 / 88 slots |
| Slots | 5: sag-fluid, sag-2nd, cor-fluid, cor-2nd, axial (finespacing 22/18/15/10/15) | 6 (plane × fat-sat) | 6 (plane × fluid; dense 26/14/20/10/14/4) |
| Series choice for a slot | preference order (fluid first) | **largest slice count at eval; random candidate in training** | **first row in CSV order** (`choose_series`) |
| Slice order | IPP · normal | IPP · normal, canonical direction per plane | IPP along dominant axis; sagittal reversed for left knees |
| In-plane orientation | none | **full: transpose / flip rows / flip columns to a fixed LPS frame (from IOP)** | left–right mirror for cor/ax only (laterality); no transpose or vertical flip |
| Laterality | none (raw) | none explicit (LPS frame) | canonical right knee |
| Span | 2–98% | 4–96% (window centres) | 4–96% |
| In-plane scale | 140 mm crop → **336 px stored (0.417 mm/px)**, upsampled to 384 | resampled to 0.4 mm/px in a 153.6 mm frame; 0.92 crop → 256 px (**0.55 mm/px**) | 140 mm crop → **256 px stored (0.547 mm/px)**, upsampled to 288 |
| Intensity | per-series 2–98 percentile | per-series 1–99.5 percentile | per-series 1–99 percentile |
| Train windows | random centres with replacement, k 12 | **one random centre in each of k equal bins** per series, k 8 | uniform random choice from valid centres, k 16 |
| Eval windows | every position (62 / 78) | evenly spaced, 12 per series (72 total) | evenly spaced, 48 of 86 |
| Augmentation | ±10% gain only | rotation ±12°, scale ±12%, shift ±6%, gamma exp(±0.25), gain ±10%, all shared per series, on GPU | affine (shift 0.06, scale 0.1, rotation 10°), brightness/contrast 0.2, gamma 0.8–1.2, shared per study; window dropout 0.1 |
| Model extras | tanh attention-MIL | slice-position embedding + 2-layer transformer over all windows, then per-finding attention | slot embedding + gated attention-MIL |
| Optimisation | backbone 3e-5 / head 1e-3, OneCycle, 16 epochs, batch 8, bf16, clip 3, pos_weight BCE | backbone **1e-4** / head 5e-4, cosine + 1 warm-up epoch, 14 epochs, batch 8, clip 3, **EMA 0.998 (evaluated with EMA)** | backbone 5e-5 / head 5e-4, 15 epochs, batch 8, clip 1, no EMA, top-3 average |
| Labels | own LLM soft labels (public: `dreaddevelopment/rsna-knee-labels`, no gold rows) | mean of 4 public tables (steven v4, pilkwang v2, lixin, qwen), clipped 0.01–0.99 | blend of 3 + silence priors |
| Folds | none (all data); epoch chosen on gold | GroupKFold **by normalised report text** | fixed `folds_5.csv` |

### 6.2 Measured evidence (single-variable, public LB, from Raptor's dataset pages)
| Change | LB | Takeaway |
|---|---|---|
| 140 → 170 mm crop at 336 px (0.417 → 0.506 mm/px) | 0.926 → **0.917** | in-plane detail beats coverage |
| Store at 384 instead of 336 (0.417 → 0.365 mm/px) | 0.926 → 0.926 | detail saturates around 0.4 mm/px |
| 44 slices, 6–94% → 2–98% (thinner) | 0.926 → 0.917 | wider span only helps with more slices |
| 64 slices over 2–98%, 62 windows | 0.928 | |
| 80 slices over 2–98%, 78 windows | 0.932 | 1.2% of the stack per slice is the best measured |
| Same weights, 42 → 62 eval windows | 0.924 → 0.927 | score every window position |

Other teams' measurements:

| Pipeline | Change | Result | Takeaway |
|---|---|---|---|
| Ziad (pooled attention head, like ours) | denser slices | −0.0045 | density pays for full-coverage readers, not for heads that already see the span |
| Ziad | 140 → 110 mm crop | ±0 | |
| Thread 745759 (public baseline) | sort slices by IPP instead of filename | +0.028 | we already sort by IPP |

### 6.3 Our own measurements today
- **Duplicated slices from fixed-count sampling.**
  - Sag-FS (26 slots): 33% of studies contain repeated consecutive slices, because their series is shorter than
    26 within the span. **10% of sag-FS training windows have a repeated channel.**
  - Cor-FS: 14% of studies, 4% of windows. Other slots: about 0.
  - Measured on 300 cached studies of `cache_256_dense`.
- **Several candidate series per slot.**
  - 13.6% of (study, plane, fluid) slots have ≥ 2 series, and **44% of studies** have at least one such slot.
  - By slot: axial non-fat-sat 35%, sagittal non-fat-sat 20%, coronal non-fat-sat 17%, sagittal fat-sat 12%.
  - We cache one series per slot, picked by CSV order, so the extra series are never seen.
- **Identical template reports.** 223 studies (5%, 52 groups) share an identical report, and 48 groups are split
  across our folds. Same labels, different images, so there is no image leakage. Low priority.

### 6.4 What to adopt, ranked
| # | Change | Evidence | Cost |
|---|---|---|---|
| D1 | **Finer in-plane scale: rebuild the dense cache at 336 px, 140 mm (0.417 mm/px)**; train ConvNeXt-S @336 and CoAtNet @384 from it | Raptor 170 mm (0.506 mm/px) cost −0.009; we are at 0.547 mm/px | Cache ×1.72 → ~31 GB, over Kaggle's 20 GB output cap, so build in 2 parts (studies split in half, like Raptor's corpus + ext). ConvNeXt @336 ≈ 1.35× train time (~100 min/fold on a 3090) |
| D2 | **Series choice: the most slices among exact matches** (not CSV order); also cache a 2nd candidate where one exists and pick at random in training (TTA option at inference) | goodpjw; 44% of studies affected | `choose_series` + a cache rebuild (do it together with D1) |
| D3 | **Per-slot sample count capped at the native slice count** (no duplicate slices); mask the unused slots | 33% of studies hit on sag-FS | small change in `sample_indices` / cache build; together with D1 |
| D4 | **Stratified random window centres** (one random centre per equal bin, per slot) instead of uniform random choice; eval at every centre (k_eval = all) where time allows | goodpjw; Raptor window-count table | ~20 lines in `datasets.py`; no cache change; 1 fold to test |
| D5 | **Full in-plane canonicalisation** (transpose / vertical flip to a fixed LPS frame from IOP) | goodpjw. Prevalence of non-standard in-plane orientation is unknown: add a header-only scan of IOP for all 24k series to the cache-build notebook first | small, in the cache build |
| D6 | Training recipe from the reader: **EMA 0.998 with eval on EMA weights**, backbone LR 1e-4 + cosine, clip 3 | goodpjw | = EXP-030 (config) |
| D7 | Labels: add a 4th public source (qwen or lixin, as goodpjw does) and consider Dread's public labels as a 5th. Dread's labels have no gold rows, so judge them by OOF and LB only | goodpjw, Raptor | = EXP-035 |
| — | Not worth it: wider crop (Raptor −0.009); storing above 0.4 mm/px (level); more slices for our pooled head without more coverage (Ziad −0.0045) | | |

**Suggested order:**
1. D4 + D6 on the current cache, one fold each. They are cheap and need no rebuild.
2. One Kaggle cache rebuild that bundles D1 + D2 + D3 (+ D5 after the IOP scan): `cache_336_dense_v2`, 2 parts.
3. ConvNeXt-S @336 fold 0 on it against the 0.8745 / 0.9037 bar, then CoAtNet @384 on it.
