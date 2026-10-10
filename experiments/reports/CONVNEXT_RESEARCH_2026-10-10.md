# Improving ConvNeXt and EXP-055: research from discussions and public code (2026-10-10)

Sources: 147 discussion topics and about 60 pulled public notebooks, plus our gold-58 evaluation of every model
(`runs/GOLD-EVAL/`).

## What the evidence says
| Source | Finding | Size |
|---|---|---|
| Raptor weights (737696, Dread) | Same weights, more windows scored at test: 42 → 62 → 78 windows = 0.924 → 0.927 → **0.932**; 24 windows ≈ 0.92 | **+0.003 to +0.008, inference only** |
| goodpjw stack notebook | Training targets = 50/50 report labels + held-out reader predictions: gold 0.896 → **0.921** as labels; single model **0.922 → 0.927 LB**, but the 5-fold ensemble 0.930 → 0.929 (smoother labels make the folds alike) | +0.005 single; use a teacher of a **different architecture** |
| Best single model (735304) | Raymond Yuen: all-data CoAtNet 288 0.938 → **0.949** from labels only (pseudo-labelling + blended label sets). tennogh: OOF pseudo-labels correlate with LB (0.943) | labels are the main lever for others |
| nartaa (746792) | 96 slices / native 384 / 94 windows +0.001; **label blend + JEV +0.002**; their non-OAI recipe = **0.942** vs our replica EXP-068 = 0.938 | 0.004 gap on the same recipe |
| Tucker Arrants (740610, low-0.95 models) | A simple CNN at 224–288 px with plain pooling reaches 0.94+; then ablate augmentations one at a time (MRI-specific) against report-label CV | method |
| 735304 | 140 mm crop "helped a ton" vs 100 mm; 160 mm similar | we already use 140 mm |
| Ziad Ahmed (737696) | More slices per slot can hurt in another pipeline (−0.0045) | geometry gains do not transfer blindly |
| Bee's knees, EXP-069 | nartaa 0.949 + a ConvNeXt reader at 0.3 → 0.948; a weaker member at 30% lowers a much stronger one | blend weights must follow strength |
| Dread (742926) | Models that disagree make better ensembles | diversity over copies |

## Our ConvNeXt dense vs the best recipes
| | ConvNeXt dense (EXP-024/026, 5f LB 0.930) | Raptor / nartaa |
|---|---|---|
| Cache | 88 slices, **stored 256, upsampled to 288** | 80–96 slices, native 336/384 |
| Span | 4–96 % | 2–98 % |
| Laterality | canonical (left → right): no mirror TTA | native: mirror TTA (+0.001) |
| Windows at test | **48** within slots | **all** (78–94) |
| Labels | our older blend | 4 tables + JEV ×2 |

## Ranked directions
1. **All windows at inference for ConvNeXt dense 5f and CoAtNet f0** (k_eval 48 → all, about 76 windows).
   - No training: add a per-family `k_eval` override to `src/models/inference.py`.
   - Check on gold-58 (one Kaggle run), then on the LB in an exploration slot.
   - Cost: about +60% GPU time for those two families.
2. **Pseudo-label ConvNeXt student.** Targets = 0.5 labels + 0.5 pooled OOF of Raptor repro 5f (a CoAtNet teacher,
   i.e. a different architecture).
   - Fold 0 first; judge on gold + LB, not on OOF (OOF inflates once targets contain predictions).
   - Expect +0.005 single; may not add in a blend.
   - EXP-010 (2026-08) was negative, but it used a weak EffNet-B0 teacher and was judged on report-label OOF.
3. **ConvNeXt with OUR recipe on native-resolution data.**
   - cache_336_raptor (80 slices, 336 native, 2–98 %, non-canonical) at 288 or 336.
   - Keep lr 5e-4 / backbone ×0.1, attn_mil, our augmentations, + JEV labels.
   - EXP-073 changed data, LRs and head at once (gold 0.897); this isolates the data change.
   - Non-canonical also makes the model mirror-eligible. Fold 0 gate vs EXP-024 (OOF 0.8745, gold 0.9037).
4. **Close the 0.942 vs 0.938 gap on the 96-slice recipe** (the strongest family):
   - match nartaa's label blend exactly;
   - try a third seed only if EXP-074 shows seed averaging helps.
5. **Augmentation ablation** (Tucker's method) on ConvNeXt fold 0. One change at a time: gain-only vs ours vs + zoom.
   Lower prior, more runs.
6. **A different family for diversity**: DINOv2-S fold 0 (ITER-08 EXP-062, not run yet). Others report 0.937–0.942 with DINOv2.

## Results
- **Direction 1 (all windows at inference) — NEGATIVE.** Run: gold-eval v2, `--k_eval_override`, 58 gold studies.
  - Predictions changed (mean |Δp| 0.011–0.018, corr 0.995–0.998), but gold AUC did not:
    CoAtNet f0 0.9088 → 0.9088, ConvNeXt dense 5f 0.9100 → 0.9101.
  - In the blends: EXP-075 0.9226 → 0.9220, EXP-076 0.9259 → 0.9254.
  - Likely reason: our within-slot windows already sample 48 of ~76 positions evenly per slot, and adjacent windows
    overlap. Raptor's gain came from cross-slot windows over a 2–98 % span.
  - Not adopted: it would add ~60 % runtime for those families.
- **Forced mirror TTA on the canonicalised families — NEGATIVE, as expected.** Run: gold-eval v3, `--mirror_force`.
  - The mirror pass alone reads like an out-of-distribution input: CoAtNet f0 0.9088 → 0.8763, ConvNeXt dense
    5f 0.9100 → 0.8892.
  - Averaged with the plain pass it still loses: CoAtNet f0 0.9023 (−0.0065), ConvNeXt dense 5f 0.9062 (−0.0038).
  - In the blends: EXP-075 0.9226 → 0.9219, EXP-076 0.9259 → 0.9234.
  - Keep mirror off for every laterality-canonicalised model. Training on native-laterality data (EXP-080) is the
    way to make ConvNeXt mirror-eligible.
- **Direction 2 (EXP-079, pseudo-label student) — POSITIVE on CV.**
  - Fold 0 OOF vs report labels: 0.8825, vs EXP-024 0.8745 (+0.008) and the teacher's architecture (Raptor f0) 0.8770.
  - Gold 0.9044 vs 0.9037 (flat).
  - LB check: EXP-078 (fold 0 alone) vs EXP-025 0.926, on 2026-10-11.
  - If it gains, train folds 1-4 (~1.5 h each on a 3090).
- **Direction 3 (EXP-080, ConvNeXt on the native 336 cache, our recipe) — NEGATIVE.**
  - OOF 0.8644 (−0.010), gold 0.8897 (−0.014).
  - With our recipe ConvNeXt prefers the canonicalised dense cache, so it stays mirror-ineligible.

Order: 1 is free and ready for tomorrow's exploration slot. 2 and 3 are one 3090 fold-0 run each (~2.5 h).
