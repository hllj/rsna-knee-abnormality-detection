# Iteration 02 (plan) — resolution / backbone for the weak meniscus & OA labels

Current best EXP-009 (EfficientNet-B0 @224 from a 256 px / 140 mm / 72-slice cache, 5 folds, LB 0.914).
Weakest labels, pooled OOF vs report labels / gold-58: Lateral Meniscus 0.841 / 0.812, MCL 0.823 / 0.927,
PF OA 0.845 / 0.857, Lateral OA 0.856 / 0.832, Synovitis 0.857 / 0.755.

## Evidence from Kaggle (discussions + public notebooks / weights)

| Source | Setup | Result | What it says |
|---|---|---|---|
| Raptor `raptor-knee-native384` page | CoAtNet-2 @384, same 44 slices / 6–94 % / 140 mm, corpus stored at **384 vs 336** | LB 0.926 vs 0.926 | storing more pixels than the model's input already interpolates to: **no gain** |
| Raptor `raptor-knee-maxspan` page | CoAtNet-2 @384, **64 slices over 2–98 %**, 336 px store, 62 eval windows | LB **0.928** (single model) vs 0.926 (44 slices); 44 slices spread over 2–98 % → 0.917 | **slice density** is the lever, not span or stored resolution |
| Raptor fine-spacing (disc. 737696) | 80 slices: sag-FS 22, sag 18, cor-FS 15, cor 10, ax 15 | LB 0.932 at 78 windows | denser sagittal/coronal sampling keeps paying |
| Disc. 737696 (Raptor author) | eval windows 48 → 72 | LB → 0.944, +8 min runtime | more eval windows helps until ~30–60, then flat |
| Disc. 737696 (other team) | windows k = 16 / 24 / 31 / 48 / 62 | 0.8867 / 0.8909 / 0.8932 / 0.8939 / 0.8948 (weak-label macro) | saturates around 31–48 windows |
| Disc. 735304 | CoAtNet single fold @224 / @288 / @384 | LB 0.950 / 0.938→0.949 (labels) / 0.937 | resolution is **not** what separates top single models |
| Disc. 735304 | ResNet 5-fold @224; small ResNet single fold @224 | LB 0.954; 0.936 | 224 is enough with good labels |
| Disc. 735304 (0.94+ competitors) | "began with 384 px and downgraded to 288, barely any effect"; "224/288 is the sweet spot so far" | — | 288 is a safe upper bound for CNNs |
| Disc. 737597 (controlled fold-0 ablations) | EffNet-B0, 288 px, 140 mm: 192 px/no crop 0.7695 → 288/140 mm 0.7777; centre-9 adjacent vs 9 spread +0.018 | Lateral Meniscus their weakest (0.740) | **slice geometry > resolution**; meniscus tears are 1–3 mm and global pooling dilutes them |
| `wguesdon/rsna-knee-dinov2-at-meniscus-resolution` | sampling argument: 1 mm tear needs ≤ 0.5 mm/px → ≥ 300 px over 150 mm; 448 never beat 336 | — | theory says ≥ 300 px for tears (for ViT patch tokens; CNN stride differs) |
| `stevenleehans/…cpu-pixel-cache` | crop 130 mm normalises 98.5 % of series vs 39 % at 160 mm | — | our 140 mm is fine; 130 mm is the other common choice (72 of the public configs use 130) |
| Disc. 740610 (0.954 single-model author) | "Start with 224/288 px and a simple CNN (ResNet34 / EffNet-B0)… then try different backbones and resolutions to see if the extra training time is worth it" | — | resolution/backbone is a late, small lever |
| Public ensemble routing (renta, mattiaangeli) | Lateral Meniscus taken **100 % from the CoAtNet branch** (0.939 → 0.940) | +0.001 LB | the hybrid CoAtNet reads lateral meniscus better than DINO/RadImageNet |

**Reading:** raising the *stored* resolution alone is roughly neutral; what moved scores was **denser slice
sampling (especially sagittal/coronal fluid series), more eval windows, and a hybrid CNN-transformer
(CoAtNet) backbone**. Our 256 px cache at 140 mm is 0.55 mm/px — already close to the ≥ 300 px / 0.5 mm/px bound —
so upsampling it to 288–384 at the model input is what the Raptor 336→384 models do, at no re-cache cost.

## Planned runs (each follows the loop: fold 0 → gate vs EXP-009 fold 0: OOF 0.865, gold 0.876)

| # | Config | Change | Cost (3090, fold 0) | Why |
|---|---|---|---|---|
| 1 | *(no training)* `windows.k_eval=64` at inference | all ~66 window positions instead of 48 | 0 GPU; re-score OOF | Raptor +0.003–0.006 from denser eval windows |
| 2 | `effnet_288` | EfficientNet-B0 @288 upsampled from the 256 cache | ~1.3× EXP-006 (~50 min) | cheapest resolution test; 288 is the "sweet spot" upper bound |
| 3 | `coatnet_384` | CoAtNet-RMLP-2 @384 from the 256 cache, grad checkpointing, bs 6 | ~3–4 h/fold | the backbone behind every 0.92–0.95 single model + lateral-meniscus routing |
| 4 | `convnext_288` | ConvNeXt-S @288 | ~1.5 h/fold | a different CNN for ensemble diversity |
| 5 | new cache `dense` (Kaggle CPU notebook) | sag-FS 24, sag 16, cor-FS 20, cor 10, ax-FS 16, ax 4 = 90 slices | ~2 h CPU re-cache | strongest single lever in the evidence (density) |

Order: 1 → 2 → 3 (CoAtNet only if 2 shows the upsampled input helps or if fold-0 budget allows) → 5.
Submissions per the loop: fold 0 of the winner, then its 5 folds, then a rank blend with EXP-009.
