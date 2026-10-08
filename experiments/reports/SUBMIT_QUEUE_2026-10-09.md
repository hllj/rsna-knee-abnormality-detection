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
