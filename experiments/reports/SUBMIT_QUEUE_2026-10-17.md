# Submission queue — 2026-10-17

Picks now: **EXP-117** (R6 + flip TTA, 0.65/0.35 routing; Kaggle page order EXP-117 > EXP-118 > EXP-113 > EXP-105,
all 0.957) and **EXP-104** (rule-safe, 0.943).

## Public-notebook scan (10-16 23:40 local)
- Top of public LB 0.968 (CentralyDS); 0.966–0.967 for the next six teams.
- goodpjw2008's notebook is now titled **LB 0.958** (v26): R6 + flip 0.6 + R7 0.4 in the compact slot, goodpjw reader 0.1,
  and **heliosli's per-finding Raptor weights** from heliosli's private-LB blend notebook (10-15/16).
- heliosli's recipe (blend-mine-in notebook): nartaa 384 + 224 (A224 0.35) vs R6 + flip TTA only; per-finding nartaa
  weight Synovitis 0.8, Lateral OA 0.2, PF OA 0.6, Baker's 0.35, the other 8 findings 0.5 (R6 = 1 − it).
- Nothing else new: all other new notebooks use the same nartaa / heliosli / OrthoFoundation datasets.

| EXP | Notebook | Change vs EXP-117 | Status |
|---|---|---|---|
| EXP-119 | v78 | heliosli's 0.958 per-finding weights (above) in place of our 0.65/0.35 routing; R6 + flip TTA | smoke-tested; runtime ≈ EXP-117 (5 h 15 min); **awaiting owner OK to submit** |

## DINOv2 / RadImageNet (owner question 10-16)
- Both load offline: inference builds the model with `pretrained=False` and loads our checkpoint.
- **DINOv2 ViT-S/14** (`vit_small_patch14_dinov2.lvd142m`, Apache 2.0, natural-image pretraining): no code change,
  `configs/exp/dinov2_s_224.yaml` exists (planned as EXP-062 in ITER-08, never run). Public 5-fold DINOv2 reported 0.937–0.942.
- **RadImageNet ResNet-50** (`shigechan/radimagenet-resnet50-official` = HF Lab-Rasool/RadImageNet, MIT): keys are
  `backbone.{0,1,4,5,6,7}` (torchvision children) → needs a ~20-line `pretrained_path` hook mapped onto timm `resnet50`.
  Medical pretraining (includes MSK MRI); reported to help Synovitis for one team.
- Recommendation: DINOv2-S first (no code, most different from our CoAtNet/ConvNeXt members, clearly rule-safe) → fold 0
  gate, then an all-data model for pick 2. Our own models have not lifted the OAI/R6 stack (EXP-094..096), so expect the
  gain in pick 2, not pick 1. Needs a GPU rental (owner OK).
