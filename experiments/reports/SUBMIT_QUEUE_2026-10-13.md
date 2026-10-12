# Submission queue — 2026-10-13 (nartaa + heliosli anchored blends)

| # | Local | EXP | Notebook | Blend (rank fusion) | Gold-58 |
|---|---|---|---|---|---|
| 1 | 07:05 | EXP-093 | v54 | nartaa 0.7 + heliosli public compact ConvNeXt-S 0.3 (public 0.951 recipe) | 0.9313 |
| 2 | 07:06 | EXP-094 | v55 | nartaa 0.65 + heliosli 0.25 + pseudo-label ConvNeXt 5f 0.1 | 0.9306 |
| 3 | 07:07 | EXP-095 | v56 | nartaa 0.65 + heliosli 0.25 + reader 5f 0.1 (mirror) | 0.9318 |
| 4 | 07:08 | EXP-096 | v57 | nartaa 0.6 + heliosli 0.2 + our 96-slice seeds 0.2 (mirror) | — |
| 5 | — | spare | — | — | — |

- Context: EXP-077 (nartaa 0.8 + seeds) = 0.949 equals nartaa alone; our ConvNeXt / reader / CoAtNet partners cost
  0.001–0.002 (EXP-083..086).
- Gold-58 cannot rank nartaa-anchored blends (they sit 0.005–0.008 above the gold line, in reversed order);
  the LB decides.
- Runtime ≈ nartaa 0.5 h + heliosli 1–1.5 h + ours, under 9 h.
