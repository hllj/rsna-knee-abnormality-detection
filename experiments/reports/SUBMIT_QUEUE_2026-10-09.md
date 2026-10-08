# Submission queue — 2026-10-09 (5 per day; reset 00:00 UTC = 07:00 UTC+7)

| # | EXP | Notebook version | What | Question | Status |
|---|---|---|---|---|---|
| 1 | EXP-055 | v29 | ConvNeXt dense 5f + Raptor repro 5f + reader v1 f0 + CoAtNet f0 (0.3/0.3/0.3/0.1) | What do 5 Raptor folds add over EXP-049 (0.936)? | verified (12 ckpts, ~4.1 h); cron 07:05 |
| 2 | EXP-056 | v30 | reader v2 fold 0 alone (EXP-050) | Our labels on the reader vs v1 (0.922)? | verified; cron 07:06 |
| 3 | EXP-057 | tbd | reader v3 fold 0 alone (EXP-053) | Does top-3 EMA alone fix the reader (vs 0.922)? | build after upload (~19:10 UTC) |
| 4 | EXP-058 | tbd | all-data Raptor alone (EXP-054) | All-data vs single fold (Raptor f0 0.928)? | build after upload (~21:00 UTC) |
| 5 | EXP-059 | tbd | final blend v2: EXP-055 + all-data Raptor and/or the better reader | Best blend from current members | build after #1–#4 score |

Mirror TTA stays off for blends with 5 Raptor folds: runtime would roughly double (~7 h) for ~+0.0002 CV.
