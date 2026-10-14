# Submission queue — 2026-10-14 (variants on the EXP-093 base)

## 2026-10-13 results (decided the base)
| EXP | Blend | Gold-58 | LB |
|---|---|---|---|
| **EXP-093** | nartaa 0.7 + heliosli public compact ConvNeXt-S 0.3 | 0.9313 | **0.954** |
| EXP-094 | + pseudo-label ConvNeXt 5f 0.1 | 0.9306 | 0.953 |
| EXP-095 | + reader 5f 0.1 | 0.9318 | 0.953 |
| EXP-096 | + our 96-slice seeds 0.2 | — | 0.953 |

- EXP-093 beats EXP-077 (0.949) by +0.005 and is the best submission so far.
- Adding our models to every finding costs 0.001 (one LB rounding step) at 10–20 %.
- Rule (owner-approved): EXP-093 ≥ 0.950, so the prebuilt variants on its base are scheduled.

## Schedule
| # | Local | EXP | Notebook | Change vs EXP-093 | Gold-58 | Lateral OA (gold) |
|---|---|---|---|---|---|---|
| 1 | submitted 10-13 ~14:00 local | EXP-097 | v58 | Lateral OA only: nartaa 0.2 + heliosli 0.3 + reader 5f 0.5 | 0.9340 | 0.828 |
| 2 | 07:06 | EXP-099 | v60 | EXP-097 + probit fusion | 0.9342 | 0.830 |
| 3 | 07:07 | EXP-100 | v61 | nartaa 0.6 + heliosli 0.25 + goodpjw public reader 0.15 | — | — |
| 4 | 07:08 | EXP-098 | v59 | probit fusion only | 0.9313 | 0.795 |
| 5 | — | spare | — | — | — | — |

Final pick 2 stays the EXP-093 recipe (fixed public weights, not tuned on our LB).

## Changes approved 2026-10-13 evening
- **EXP-101 = notebook v62**: nartaa 0.6 + heliosli 0.4 (smoke test OK, external stage 222 s).
- 07:06 is conditional: EXP-099 (v60) if EXP-097 >= 0.954, otherwise EXP-101 (v62).
- 07:09 submits EXP-101 (v62) if it was not used at 07:06; otherwise spare.
- EXP-100 rebuilt: goodpjw public reader 0.3 on Lateral OA + Contusion only (nartaa 0.4 there), base 0.7/0.3 elsewhere.
  goodpjw on gold-58: Lateral OA 0.855 vs nartaa 0.776, Contusion 0.957 vs 0.937, Effusion 0.882 vs 0.973 (avoid).
  Gold: restricted 0.9341 vs all-findings 0.9329 vs base 0.9313. New notebook version: see below once built.
- **EXP-100 = notebook v63** (restricted; smoke test OK; goodpjw 0.3 on Lateral OA + Contusion, 0 elsewhere). 07:07 now submits v63 (v61 retired).
| 5 | 07:10 | EXP-103 | v65 | heliosli R7 recipe: nartaa 0.5 (384 + 224 at 0.35) + R7 0.5 (fastMRI-trained; owner accepted the risk) | — | — |

## Submitted 2026-10-14 ~07:06 local (all accepted, quota used)
- EXP-097 = 0.954 (>= 0.954), so the conditional 07:06 slot submitted **EXP-099 (v60)**; 07:09 submitted **EXP-101 (v62)**.
- Also EXP-100 (v63, restricted goodpjw), EXP-098 (v59, probit), EXP-103 (v65, R7 recipe).
