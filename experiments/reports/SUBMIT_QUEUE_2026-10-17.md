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

## Public-notebook research (10-17 ~01:00 local)
- 426 notebooks scanned; best public claim still goodpjw LB 0.958 (heliosli per-finding weights + R6 flip). lavinwins "RSNA Gold Medal" = heliosli 0.958 copy (= EXP-119). goodpjw 10-17 update: R6 flip + S6 at 0.3 (second compact model; R7 as second model did not help us, EXP-118). farhanabid: old DINOv2/RadImageNet community bundle.
- Unused public model: bliverrigpdoge knee-centred fine-spacing reader `rsna-knee-leg-r9all` (loop_R9_all.pt, 10-04); its notebooks score 0.956-0.957.
- leoprovorov analysis: public half ~640 studies -> score SD ~0.003 from sampling alone, larger than the spread of the top 400 teams (0.0022); 940 teams print 0.956 from the same models; averaging stops helping as model errors correlate -> independent families (our DINOv2 / RadImageNet, EXP-120/121) are the remaining lever; public 0.957 vs 0.958 is mostly noise.

## ITER-13 new families (box 56274019, 2x RTX 3090, 2.16 h, ~$2.3; destroyed 10-16 19:12 UTC)
All three trained on all data (fold -1) on cache_384_d96 and uploaded as Kaggle Models; none passes the gold-58 gate (EXP-065 0.914).

| EXP | Model | Gold-58 | Rank corr with base | Gold blend with EXP-065+072 (base 0.9202), w = 0.05 / 0.1 / 0.2 |
|---|---|---|---|---|
| EXP-120 | DINOv2-S, last 6 blocks, 10 ep | 0.841 | 0.85 | 0.9187 / 0.9173 / 0.9145 |
| EXP-121 | RadImageNet R50, full FT, 16 ep | 0.842 | 0.82 | 0.9195 / 0.9190 / 0.9143 |
| EXP-122 | DINOv2-S, whole encoder, 20 ep | 0.864 | 0.90 | 0.9190 / 0.9178 / 0.9136 |
| all three | equal mix | — | — | 0.9192 / 0.9178 / 0.9148 |

Reading: every weight lowers the gold blend; the new families are ~0.05-0.07 weaker and not decorrelated enough to pay for it.
Recommendation: keep picks EXP-117 / EXP-104 unchanged; no submission slots spent on these blends.

## Public-notebook scan (10-17 ~10:00 local)
- **heliosli V2** (`this-is-best-public-dataset-new-version-0-9576`, 10-16 19:05): same nartaa 384 + 224 (A224 0.35) vs R6 + flip TTA, but
  sharper per-finding nartaa weights (Synovitis 0.85, Lateral OA 0.15, PF OA 0.65, Baker's 0.30, Medial OA and the other 7 0.5) and
  goodpjw's 2.5D reader (3 public folds) rank-blended on the final submission at goodpjw's v11 per-finding weights
  (Baker's 0.035, Contusion 0.03, Medial OA / ACL 0.025, Medial Meniscus / MCL 0.02, Lateral Meniscus / Fracture / Lateral OA / PF OA 0.008,
  Effusion 0.004, Synovitis 0.003). → **EXP-123** (our fusion: nartaa = wr(1−g), R6 = (1−wr)(1−g), goodpjw = g).
- lavinwins `top-1-gold` (10-17) = heliosli's 0.958 recipe (= EXP-119).
- goodpjw R6 + S6 (10-17): 0.957 vs 0.958 without S6 → negative, not applied.
- lavinwins `rsna-gold-medal` now a "v6 combination layer" on the 0.943 public pipeline (rank-fix for A5 subpopulation blend,
  per-target routing); claims 0.943 → 0.945, different model stack from our pick 2 (EXP-104) → not applied.

| EXP | Notebook | Change vs EXP-117 | Status |
|---|---|---|---|
| EXP-119 | v78 | heliosli 0.958 per-finding weights | submitted 10-17 10:43 local |
| EXP-123 | v79 | heliosli V2 (sharper weights + goodpjw reader v11) | submitted 10-17 10:43 local |

## Our routing on top of heliosli's weights (owner request 10-17 ~11:00 local)
Union recipe: heliosli's weights where they leave 0.5 (Lateral OA, PF OA, Synovitis, Baker's), our EXP-117 routing on the
findings heliosli keeps at 0.5 (nartaa 0.35 on ACL / Lateral Meniscus, 0.65 on Medial Meniscus / Effusion / Fracture).
On PF OA and Baker's our routing and heliosli's point in opposite directions; heliosli's (LB-derived) are kept.

Gold-58 sanity check (`gold_hybrid.py`; nartaa in-sample, R7 + flip stands in for R6, our routing chosen on these studies):

| Recipe | Gold-58 macro | Note |
|---|---|---|
| all findings 0.5 | 0.935 | |
| EXP-117 (ours) | 0.940 | routing fitted here |
| EXP-119 / EXP-123 | 0.937 / 0.937 | |
| EXP-124 = 123 + ours | 0.939 | +Lateral Meniscus, Effusion, Fracture, ACL |
| EXP-125 = 119 + ours | 0.939 | same |

| EXP | Notebook | Parent | Status |
|---|---|---|---|
| EXP-124 | v80 | EXP-123 + our routing | smoke-tested (COMPLETE, weights verified); awaiting owner OK |
| EXP-125 | v81 | EXP-119 + our routing | smoke-tested (COMPLETE, weights verified); awaiting owner OK |
