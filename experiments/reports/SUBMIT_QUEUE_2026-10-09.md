# Submission queue — 2026-10-09 (5 per day; reset 00:00 UTC = 07:00 UTC+7)

| # | EXP | Notebook version | What | Question | Status |
|---|---|---|---|---|---|
| 1 | EXP-055 | v29 | ConvNeXt dense 5f + Raptor repro 5f + reader v1 f0 + CoAtNet f0 (0.3/0.3/0.3/0.1) | What do 5 Raptor folds add over EXP-049 (0.936)? | verified (12 ckpts, ~4.1 h); cron 07:05 |
| 2 | EXP-056 | v30 | reader v2 fold 0 alone (EXP-050) | Our labels on the reader vs v1 (0.922)? | verified; cron 07:06 |
| 3 | **EXP-060** | v32 | pair blend: ConvNeXt dense 5f + Raptor repro 5f, 50/50 | Best all-5-fold blend (pooled OOF 0.8852, gold 0.9130): a final-pick candidate | verified (10 ckpts); cron 07:07 |
| – | ~~EXP-057~~ | v31 | reader v3 fold 0 alone | dropped: v3 = v1 on CV (0.8691 vs 0.8690), corr 0.95, so little to learn | not submitted |
| 4 | EXP-058 | v33 | all-data Raptor alone (EXP-054, gold 0.9064) | All-data vs single fold (Raptor f0 0.928)? | verified; cron 07:08 |
| 5 | EXP-059 | tbd | final blend v2: EXP-055 + all-data Raptor and/or the better reader | Best blend from current members | build after #1–#4 score |

Mirror TTA stays off for blends with 5 Raptor folds: runtime would roughly double (~7 h) for ~+0.0002 CV.

## Results (2026-10-09)
| EXP | What | Public LB |
|---|---|---|
| EXP-055 | ConvNeXt 5f + Raptor 5f + reader v1 f0 + CoAtNet f0 (0.3/0.3/0.3/0.1) | **0.936** (= EXP-049 with Raptor f0) |
| EXP-056 | reader v2 fold 0 alone | 0.920 (v1 0.922: higher CV from our labels did not transfer) |
| EXP-060 | ConvNeXt 5f + Raptor 5f, 50/50 | 0.934 (pooled CV 0.8852, but no reader / CoAtNet) |
| EXP-058 | all-data Raptor alone | **0.930** (best single model; Raptor f0 0.928) |
| EXP-059 | EXP-055 with the Raptor share split 5f / all-data (0.3/0.15/0.15/0.3/0.1) | pending (v35, 13 ckpts, ~4.8 h) |

- **Family diversity beats folds.** Dropping the reader and CoAtNet costs −0.002 (EXP-060). Raptor 5f vs Raptor f0
  inside the blend: ±0 at three-decimal LB resolution.
- **Reader:** v1 stays. Training it on our labels (v2) raised CV against those same labels but lowered LB.
- **All-data model:** on par with a 5-fold ensemble (0.930), so worth more all-data models (ConvNeXt, a 2nd
  Raptor seed).
