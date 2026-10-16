# DINOv2 / RadImageNet: public code, weights and recipes (2026-10-17)

Question (owner): can we train DINOv2 or RadImageNet and blend the best into both final picks?

## Public weights trained on competition data (no training needed)
| Asset | What | Used in |
|---|---|---|
| `pilkwang/rsna-knee-weights` (20 × 89 MB) | 20 DINOv2-S slot-attention members (`m_<hash>.pt`), competition data + LLM labels | every 0.941–0.943 community notebook |
| `antoinegg1/rsna-knee-e9-radimagenet-heads-v15`, `-e11-diverse-heads-v20`, `prvsiyan/rsna-knee-v52-radimagenet-heads-20260812`, `mattiaangeli/rsna-knee-radimagenet-foldsv1-heads` (~58 MB each) | attention heads on a frozen RadImageNet ResNet-50 encoder | same |
| `tonylica/rsna-knee-bend-dinov3-0917-repro-assets` (4 GB) | one-dataset bundle: DINOv2 ×20 + RadImageNet encoder + 10 heads + DINOv3 A5 + Raptor; notebook `tonylica/rsna-knee-dino-radimagenet-rank-ensemble` (source recipe verified 0.941 public) | — |
| `girishbose/dinov2-vits14-rsna-knee`, `-vitb14-` | other fine-tuned DINOv2 | — |
| `sadamtorres/rsna-ft-a` / `-b` | 5-fold DINOv2 ViT-B/14-reg4 @224 / 336, LB 0.866 / 0.883 alone | — |

## Training recipes
- **pilkwang `rsna-knee-baseline-v1`** (DINOv2-S, the source of the 20 members; copied verbatim by mattiaangeli and
  sofiaanjenje): HF DINOv2-small (`metaresearch/dinov2/PyTorch/small/1`), 6 slots (one series per plane × weighting ×
  fat-sat), 3 adjacent slices as the 3 channels, CROP_MM 130, CACHE_IMG 336, slot attention + presence mask,
  `EPOCHS 10`, `LR_HEAD 1e-3`, `LR_BACKBONE 8e-6`, `UNFREEZE_LAST 6` (last 6 blocks trained), batch 8 studies.
- **sofiaanjenje `rsna-knee-e11-train`**: RadImageNet ResNet-50 (`marwanmath/resnet-50-radimagenet-marwan`) frozen encoder
  + attention heads, same slot pipeline; heads blended into the DINO parent at alpha 0.35–0.60 (chosen on OOF).
- **wguesdon `rsna-knee-dinov2-at-meniscus-resolution`**: IMG_SIZE 336 (24 × 24 patches); ≥ 300 px needed for 1–3 mm tears.
- **sadamtorres**: canonical right knee, slice-position + plane embeddings, CLS + patch-mean features.
- **Pre-trained backbones**: DINOv2-S HF copy `bijaychdas/rsna-knee-2026-dinov2-small-hf`; RadImageNet ResNet-50
  `shigechan/radimagenet-resnet50-official` (HF Lab-Rasool/RadImageNet, MIT; keys `backbone.{0,1,4..7}`).

## Caveats
- Several DINO trainers upweight gold in the loss (×2 DINOsaur v5, ×3 read-the-report, ×8 wguesdon) → gold-58 is not
  a held-out ruler for those members (GOLD58_REVIEW_2026-10-09.md). pilkwang's baseline status unchecked.
- The community bundle alone is 0.941–0.943, below both picks; heliosli's blend-gold notebooks dropped it for R6/R7.

## Options
- **A (free, fastest):** run pilkwang DINOv2 ×20 + RadImageNet heads (tonylica's one-dataset bundle) as an external
  stage and blend into pick 2 (EXP-104, 0.943) at ~0.25; one test in pick 1 at ~0.1. Smoke tests are free.
- **B (train our own):** port pilkwang's recipe (partial unfreeze of the last 6 blocks, backbone LR 8e-6) into
  `KneeMILModel`, fold-0 gate then all-data; GPU rental ~$8–10, ~2 days to a blendable model.
