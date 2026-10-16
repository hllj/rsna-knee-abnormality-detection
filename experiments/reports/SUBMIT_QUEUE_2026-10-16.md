# Submission queue — 2026-10-16

Base: EXP-105 (public 0.957, rank ~241/6,014): nartaa 0.65 x 384 + 0.35 x 224 vs heliosli R7, routed per finding
(R7-side Lateral OA/PF OA/Lateral Meniscus/ACL; nartaa-side Effusion/Fracture/Medial Meniscus/Baker's; others 0.5).
Final picks now: **EXP-113** (top score) and **EXP-104** (rule-safe, 0.943).

| EXP | Notebook | Change vs EXP-105 | Gold-58 | Status |
|---|---|---|---|---|
| EXP-113 | v72 | R7 + flip TTA | 0.9396 (EXP-105 0.9404) | 0.957 (10-15); Kaggle page ranks it above EXP-105 |
| EXP-114 | v73 | lighter routing 0.6 / 0.4 | 0.9397 | **0.957** (scored 12:39 local, ~3 h 20 min) |
| EXP-115 | v74 | 0.6 / 0.4 + R7 flip TTA | 0.9391 | submitted 10-16 09:18 local; score pending (~5 h, flip) |
| EXP-116 | v75 | flat 0.5 + R7 flip TTA (flip-only test vs EXP-103) | 0.9355 (EXP-103 0.9365) | dropped |

The 07:04 scheduled run did not submit; re-run at 09:18 local submitted EXP-114 + EXP-115, 3 slots spare.

**EXP-113 result (10-15 19:57 local): 0.957**, the same displayed score as EXP-105. Submit → score took 5 h 15 min (full
test, including queue), matching the flip-TTA estimate. Our rank went 240 → 266 while other teams moved; a
snapshot comparison could not show whether EXP-113's unrounded score is above EXP-105's. Flip TTA gives no visible gain
→ per the rule, 10-16 submits **EXP-114 only**; EXP-115/116 stay unsubmitted. Pick 1 stays EXP-105 (faster, same score).

**Update 10-15 21:30 (owner, Kaggle submissions page sorted by unrounded score): EXP-113 > EXP-105 > EXP-106.**
Flip TTA gives a small real gain → **pick 1 = EXP-113** (full-test run 5 h 15 min, within 9 h). 10-16 now submits
**EXP-114 and EXP-115** (0.6/0.4 without / with flip), 3 slots spare; EXP-116 dropped.

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
