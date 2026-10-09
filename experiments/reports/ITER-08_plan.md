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
