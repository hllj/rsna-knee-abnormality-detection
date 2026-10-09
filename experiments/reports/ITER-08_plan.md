# ITER-08 plan — all-data reader + two new families (DINOv2, RadImageNet), drafted 2026-10-09

**Starting point:**
- Best public LB **0.936** (EXP-049 / EXP-055).
- EXP-059 (adds the all-data Raptor, family weights 0.3/0.15/0.15/0.3/0.1) is pending.
- Evidence and reasoning: `STRATEGY_2026-10-09.md`.
  - New model families move the blend (+0.002 to +0.006).
  - More folds or seeds of a family already in the blend barely do (Raptor f0 → 5f: 0.936 → 0.936).
  - An all-data model equals a 5-fold ensemble on its own (all-data Raptor 0.930).

## 1. All-data reader (EXP-061)
- **What:** reader v1 recipe exactly, trained on all 4,349 non-gold studies (`fold=-1`):
  - goodpjw's 4-table label mean;
  - 14 epochs, EMA 0.998, export the final EMA (as goodpjw's `train.py --fold -1`).
- **Config:** `configs/exp/reader_alldata.yaml` = `reader_repro.yaml` + `exp_name: reader_alldata`. Run with
  `fold=-1`.
- **Why not reader folds 1–4:** about ⅓ of the cost (~3 h vs ~10 h), and all-data matched 5 folds for Raptor.
  Loss: no pooled reader CV.
- **Use:** reader family = v1 f0 + v3 f0 + all-data. Two runs averaged gave +0.003 CV on fold 0.
- **Cost:** ~3 h + 35 GB reader cache ≈ $2.5.

## 2. New families, fold 0 first (EXP-062, EXP-063)
**Fold 0 first (not all-data) because:**
- a held-out CV comparable with our other fold-0 models (its ranking has matched the LB so far);
- rank correlation with ConvNeXt / Raptor / reader on the same studies (the diversity check);
- a fold-0 blend check.

Then, only if it earns it (rank corr ≲ 0.9 and CV within ~0.01 of ConvNeXt), train its **all-data** model as the
production member.

| EXP | Family | Setup | Fold 0 time |
|---|---|---|---|
| EXP-062 | **DINOv2 ViT-S/14** (`vit_small_patch14_dinov2.lvd142m`, timm) | Raptor 336 px cache (24×24 patches); attention-MIL head; partial fine-tune (last ~6 blocks, backbone LR ~1e-5, head 1e-3), as pilkwang's public members | ~1.5 h |
| EXP-063 | **RadImageNet ResNet-50** (medical pretraining; RadImageNet helped Synovitis 0.62 → 0.78 for one team) | 224 px from the dense cache; weights from a public Kaggle dataset (check format / licence first); same MIL head | ~1 h |

**Code needed:**
- a `pretrained_path` / weight-loading hook for RadImageNet;
- backbone-specific LR and freezing (`freeze_blocks`);
- tests;
- a memory probe on the box.

## 3. Conditional on EXP-059
If EXP-059 > 0.936: all-data ConvNeXt dense (~1.5 h, ~$1.4) + all-data Raptor seed 2 (~4.4 h, ~$2.8).
Otherwise: skip both.

## 4. Free submissions (no GPU)
- **Reader-seed family blend:** EXP-055 with the reader family = v1 + v3 (+ v2). Same preprocessing, so the data is
  read once; ~+10 min runtime.
- **Final pick 2 (fixed now, robust):** an all-5-fold-type blend with no LB-tuned weights.

## 5. Budget, order, deadlines
- **One box (1× RTX 3090), order:**
  1. reader cache: all-data reader;
  2. Raptor cache: DINOv2 fold 0;
  3. dense cache: RadImageNet fold 0.
- **Cost:** ~$5–6 incl. storage and downloads. Size the disk to the caches. Destroy at the end.
- **Submissions:** each new model alone (fold 0), then blends.
- **Deadlines:** team merge **10-15** (decide on teaming / public stack by then), final **10-22**.

## 6. Lessons from `evgendvorkin/rsna-versia-5` (read 2026-10-09)
**What the public 0.950 actually is** (changelog in the notebook):
- 0.937 → 0.943: more CoAt "arms" and members (incl. a reverse-flip arm), routing one finding (Lateral Meniscus) fully
  to the strongest branch, k_eval 94, and family reduction in probability space.
- 0.950 replaces the ~45-checkpoint stack with **two readers**:
  - **nartaa's single CoAtNet** `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt` (0.949 alone);
  - goodpjw's ConvNeXt-T reader at per-finding weights 0.03–0.35 (fusion adds about +0.001).

**The CoAtNet is our Raptor reproduction's architecture** (identical `RaptorClassifier` tanh-MIL head, CoAtNet-RMLP-2,
ImageNet norm, cross-slot windows). The differences are the recipe and data:

| | nartaa 0.949 | our all-data Raptor (EXP-054, LB 0.930) |
|---|---|---|
| Training data | all 4,349 + **2,399 OAI knees** (external) | all 4,349 |
| Cache | **96 slices** (26/22/18/12/18, axial any), **native 384 px** | 80 slices (22/18/15/10/15), 336 → 384 |
| Labels | own + steven + pilkwang + **JEV** | default blend + JEV (w 2) |
| Optimiser | backbone 4.0e-5, head 1.95e-3, wd 0.059, **effective batch 16** (1 × 16) | 3e-5, 1e-3, 0.02, batch 8 |
| Inference | central **320 px crop** of 384 (no resize), **94 windows**, **mirror TTA** | 78 windows, full frame |

Per the 746792 post, their **pre-OAI** all-data model was already 0.939–0.942, about +0.01 above ours. That gap
comes from **labels, optimiser settings and the denser 96-slice native-384 cache**, all usable without OAI. This is
the biggest actionable lever found.

**Measured on our predictions:**
- probability-space vs rank-space reduction within a family: no difference (gold 0.9080 vs 0.9078 Raptor,
  0.9097 vs 0.9104 ConvNeXt; corr 0.997);
- per-finding fusion weights: +0.0001 cross-validated.

The 0.950 per-finding weights are commented with gold-58 AUCs, i.e. tuned on 58 studies.

**Speed engineering worth copying** (for runtime and the efficiency track):
- a multithreaded DICOM reader (16 header + 8 pixel threads per prep process);
- uint8 windows built on the GPU through a 256-entry LUT;
- backbone forward in chunks of 16 windows;
- no DataParallel (it caused host-RAM OOM on the hidden test).

**Not adopted:**
- **OAI:** host rule "generally accessible to all participants"; participants in China cannot access it.
- **nartaa's weights directly:** the public-stack question (S2 in STRATEGY_2026-10-09.md), a user decision.
- **Gold-tuned per-finding weights.**

## 7. Revised priority (supersedes section 3)
1. **Raptor v2 all-data (EXP-064):** nartaa's non-OAI recipe on our pipeline.
   - Optimiser: backbone 4e-5, head 2e-3, wd 0.06, effective batch 16 (batch 8 × accumulate 2).
   - Labels: blend + JEV.
   - Step A: on the existing 80-slice cache (no new cache).
   - Step B: if it helps, a 96-slice native-384 cache (Kaggle CPU, ~45 GB in 5 parts) + 320 crop + 94 windows.
   - This replaces the plain "2nd seed" idea.
2. All-data reader (section 1).
3. DINOv2-S / RadImageNet fold 0 (section 2).
4. Mirror TTA for the Raptor members if runtime allows (5 folds + all-data at 2× is ~8 h, too slow; all-data only
   is fine).

## 8. ITER-08 experiments (registry ids; updated as they run)
| EXP | Type | What | Config / notebook | Status |
|---|---|---|---|---|
| EXP-059 | submit | (from ITER-07) final blend v2: ConvNeXt 5f + Raptor 5f + all-data Raptor + reader v1 + CoAtNet (0.3/0.15/0.15/0.3/0.1) | inference v35 | scoring (~09:00 UTC) |
| **EXP-064** | train | **Step A — Raptor v2 all-data:** nartaa recipe (backbone 4.01e-5, head 1.95e-3, wd 0.0588, effective batch 16), best-3 epochs **by gold-58**, 80-slice cache, blend + JEV x2 | `raptor_v2_alldata`, `fold=-1` | **next on the 3090** |
| **EXP-066** | cache | **96-slice native-384 cache** (26/22/18/12/18, 2–98%, 140 mm, 384 px), 5 Kaggle CPU parts, ~44 GB | `rsna-knee-cache-r96-p0..p4` | **building now** (free) |
| **EXP-065** | train | **Step B — Raptor v2 on the 96-slice cache**, every window (94) at test + mirror TTA | `raptor_v2_d96_alldata`, `fold=-1` | after step A, if the cache is ready |
| EXP-067 | submit | step A alone (vs all-data Raptor EXP-058 = 0.930) | tbd | after EXP-064 |
| EXP-068 | submit | step B alone, with mirror TTA | tbd | after EXP-065 |
| EXP-061 | train | all-data reader (v1 recipe) | `reader_alldata` (to add) | planned, after A/B |
| EXP-062 | train | DINOv2-S fold 0 (new family) | to implement | planned |
| EXP-063 | train | RadImageNet ResNet-50 fold 0 (new family) | to implement | planned |

**Selection note for A and B:**
- **Choice:** the user chose nartaa's protocol, i.e. the 3 epochs with the best gold-58 AUC averaged (`top_k_mode: best`,
  monitor `val_auc_gold`, 24 windows per epoch).
- **Caveat:** gold-58 has a bootstrap SD of ±0.014, so the choice among late epochs is partly noise.
- **Check:** the trainer logs every epoch's gold, so the picked epochs can be compared with the last 3 afterwards.

**Update 2026-10-09 08:31:** EXP-059 = **0.936** (no change vs EXP-055). Per section 3 / the strategy rule:
**skip** plain all-data ConvNeXt and a plain 2nd Raptor seed. Step A (EXP-064) stays: it tests a different *recipe*,
judged alone vs EXP-058 (0.930), not as an extra copy.
