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
| 1 | 07:05 | EXP-097 | v58 | Lateral OA only: nartaa 0.2 + heliosli 0.3 + reader 5f 0.5 | 0.9340 | 0.828 |
| 2 | 07:06 | EXP-099 | v60 | EXP-097 + probit fusion | 0.9342 | 0.830 |
| 3 | 07:07 | EXP-100 | v61 | nartaa 0.6 + heliosli 0.25 + goodpjw public reader 0.15 | — | — |
| 4 | 07:08 | EXP-098 | v59 | probit fusion only | 0.9313 | 0.795 |
| 5 | — | spare | — | — | — | — |

Final pick 2 stays the EXP-093 recipe (fixed public weights, not tuned on our LB).
