# Submission queue and strategy — 2026-10-11 (5 per day; reset 00:00 UTC = 07:00 UTC+7)

## Gold-58 evaluation of every model (notebook `hlly34/rsna-knee-gold-eval`, 2026-10-10)
All 18 checkpoints run on the 58 gold studies exactly as on the test set (plain + mirror, rank blend);
scored by `$CLAUDE_JOB_DIR/tmp/gold_blend.py`.

| Family | Gold (mirror where eligible) | Gold used for selection |
|---|---|---|
| Raptor v2 96-slice seed 2 (EXP-072) | **0.9203** | yes |
| Raptor v2 80-slice (EXP-064) | 0.9153 | yes |
| Raptor v2 96-slice seed 1 (EXP-065) | 0.9147 | yes |
| Raptor repro 5f | 0.9104 | no |
| ConvNeXt dense 5f | 0.9100 | no |
| CoAtNet dense f0 | 0.9088 | no |
| Reader v1 f0 | 0.9084 | no |
| Raptor v1 all-data | 0.9080 | no |
| Reader v3 f0 | 0.9045 | yes |
| ConvNeXt 96-slice (EXP-073) | 0.8969 | yes |

**Calibration (blends already on the LB):**

| Blend | Gold-58 | LB |
|---|---|---|
| EXP-071 | 0.9199 | **0.940** |
| EXP-055 | 0.9202 | 0.936 |
| EXP-070 | 0.9203 | 0.936 |
| EXP-068 (96-slice s1 alone) | 0.9147 | 0.938 |
| EXP-067 | 0.9153 | 0.933 |
| EXP-060 | 0.9133 | 0.934 |
| EXP-058 | 0.9080 | 0.930 |

At our level, gold-58 does **not** rank blends the way the LB does. EXP-055 has the highest gold but only 0.936 on
the LB. The 96-slice model alone has a low gold (0.915) but scores 0.938, and EXP-071 is the best on the LB at the
same gold as EXP-055. Gold-tuned weights are therefore a hypothesis to test, not a pick.

**Grid search (family weights, step 0.1):** best gold **0.9259** = 96-slice seeds 0.5 + reader v1 0.2 + CoAtNet f0 0.2
+ ConvNeXt dense 5f 0.1. The in-sample bootstrap says 98% vs EXP-071's weights, but the weights were chosen on the
same 58 studies, so that is inflated.

## Queue
| # | Local | EXP | Notebook | What | Gold-58 | Question |
|---|---|---|---|---|---|---|
| 1 | 07:05 ✓ submitted | EXP-074 | v41 | 96-slice seeds 1+2 averaged, mirror | 0.9206 | Does a 2nd seed beat 0.938? |
| 2 | 07:06 | EXP-075 | v44 | EXP-071 recipe + seed 2: EXP-055 families ×0.6 + seeds ×0.4, mirror on seeds + reader | 0.9226 | Prior-weighted best: > 0.940? |
| 3 | 07:07 | EXP-076 | v43 | **Gold-58-picked:** seeds 0.5 + reader v1 0.2 + CoAtNet f0 0.2 + ConvNeXt 5f 0.1 | **0.9259** | Does gold-58 weight tuning transfer to the LB? |
| 4 | after 2/3 score | EXP-077 | tbd | **OAI:** nartaa 0.8 + winner of EXP-075/076 0.2, per-finding rank fusion | — | Fallback pick: > 0.949? |
| 5 | 07:08 | EXP-078 | v45 | **Pseudo-label ConvNeXt student fold 0 alone** (EXP-079: OOF 0.8825 vs 0.8745, gold 0.9044) | 0.9044 | Do pseudo-label CV gains reach the LB? vs EXP-025 0.926 |

- EXP-069 showed that our blend at 35% *lowers* nartaa (0.949 → 0.948), hence 0.2 for ours.
- Runtime: EXP-071 (7 mirrored members) took ~8.75 h of the 9 h limit. Mirror is now limited to the strongest
  families (`--mirror_families`). EXP-069 (nartaa + EXP-055) took ~4.2 h, so the OAI blend must not carry more than
  2 mirrored members.

## Strategy to the deadline (final 2 picks by 2026-10-22, team merge 10-15)
1. The LB decides close calls; gold-58 is only a coarse filter (±0.004 macro noise on 58 studies, and it misranks
   our blends above).
2. Training lever: the 96-slice cache with the v2 recipe (+0.005 LB alone). Next round only if EXP-074 shows seed
   averaging helps:
   - a third seed (~4.5 h on one 3090); or
   - a 96-slice 5-fold run for honest OOF weights (~21 GPU-h).
   ConvNeXt on this recipe is weak (gold 0.897); do not repeat.
3. Final picks: #1 = best rule-safe LB blend; #2 = nartaa-based OAI blend if the host allows OAI (743416),
   otherwise the second-best rule-safe blend from a different family mix.

## Mirror TTA: which models (gold-58, `gold_mirror.py`)
- Mirror improved every eligible family (9/9): Raptor repro 5f +0.0032, 96-slice seed 2 +0.0028, reader v3 +0.0023,
  Raptor v1 all-data +0.0014, Raptor v2 80-slice +0.0012, reader v1 +0.0009, ConvNeXt-96 +0.0009, 96-slice seed 1 +0.0006.
- Not eligible: ConvNeXt dense 5f and CoAtNet dense f0. They were trained on laterality-canonicalised inputs, so a
  mirrored knee is out of distribution.
- Effect in the blends:
  - EXP-075: none 0.9218, seeds 0.9219, **seeds + reader 0.9226**, seeds + repro5 0.9222, all eligible 0.9226.
  - EXP-076: none 0.9243, seeds 0.9253, **seeds + reader 0.9259**.
- **Rule:** mirror the 96-slice seeds and the reader only. Raptor repro 5f costs 5 extra passes for no blend gain;
  it is what pushed EXP-071 to ~8.75 h of the 9 h limit. EXP-075 was rebuilt as v44 with this rule.
- Full grid with all 9 families, mirror ON vs OFF (`gold_grid_mirror.py`): best ON 0.9259 vs OFF 0.9246 (+0.0013).
  The top blend with every family mirrored is still EXP-076's formula. Adding mirrored reader v3 / Raptor v2 80 /
  Raptor v1 / ConvNeXt-96 does not beat it, so there is no further mirror gain available.
- nartaa (746792): their 0.949 already averages plain + anatomical mirror (+0.001 public: 0.948 → 0.949); the 320 px
  crop at inference gave 0.000. Their non-OAI recipe (96 slices, label blend + JEV) scored **0.942** without mirror,
  vs our replica EXP-068 0.938 with mirror. That 0.004 gap is a lead for the next training iteration
  (labels / training details).
