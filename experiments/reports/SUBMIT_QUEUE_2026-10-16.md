# Submission queue — 2026-10-16

Base: EXP-105 (public 0.957, rank ~241/6,014): nartaa 0.65 x 384 + 0.35 x 224 vs heliosli R7, routed per finding
(R7-side Lateral OA/PF OA/Lateral Meniscus/ACL; nartaa-side Effusion/Fracture/Medial Meniscus/Baker's; others 0.5).
Final picks now: **EXP-105** (top score) and **EXP-104** (rule-safe, 0.943).

| EXP | Notebook | Change vs EXP-105 | Gold-58 | Status |
|---|---|---|---|---|
| EXP-113 | v72 | R7 + flip TTA | 0.9396 (EXP-105 0.9404) | submitted 10-15 14:43 (spare slot); score pending |
| EXP-114 | v73 | lighter routing 0.6 / 0.4 | 0.9397 | smoke-tested; **submit 10-16** |
| EXP-115 | v74 | 0.6 / 0.4 + R7 flip TTA | 0.9391 | smoke-tested; submit only if EXP-113 beats EXP-105 |
| EXP-116 | v75 | flat 0.5 + R7 flip TTA (flip-only test vs EXP-103) | 0.9355 (EXP-103 0.9365) | smoke-tested; submit only if EXP-113 beats EXP-105 |

Scheduled 07:04 local; the run checks EXP-113 first. At least one slot stays spare.

## Notes (2026-10-15)
- **R6 dropped.** heliosli's public 0.9573 notebook uses R6 (bundle fe055406) + flip TTA, but
  `heliosli/rsna-knee-r6-frozen-fe055406` version 2 now holds the R7 bundle under the same names; Kaggle attaches the
  latest version (a `/1` pin is dropped on push). Re-hosting version 1 privately was not done (third-party weights).
- **Flip TTA on R7:** gold-58 neutral (R7 alone 0.9312 → 0.9317; blends −0.001, driven by Lateral OA −0.015 on 11
  positives). Runtime ×2 for R7: 21 s vs 10 s per study per worker → R7 ≈ 3.8 h, whole notebook ≈ 5–5.5 h of 9 h.
- Kaggle scores the notebook once on the full hidden test (public 30 % + private 70 %), so the submit→score time
  is the full-test runtime.
- Rule-safe: third 96-slice seed (EXP-112, `configs/exp/raptor_v2_d96_alldata_s3.yaml`) ready; renting needs owner OK
  (~$3–4).
