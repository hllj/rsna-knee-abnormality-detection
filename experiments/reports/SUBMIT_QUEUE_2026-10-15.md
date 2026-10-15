# Submission queue — 2026-10-15 (draft; finalize after the 10-14 scores)

| EXP | Notebook | Blend | Status |
|---|---|---|---|
| EXP-102 | v64 | EXP-093 recipe + nartaa 224-crop checkpoint as 35 % of the nartaa slot (heliosli compact-v1 0.3) | smoke-tested; owner: submit after 10-14 |
| EXP-103 | v65 | heliosli R7 recipe: nartaa 0.5 (0.65 x 384 + 0.35 x 224) + R7 0.5 | smoke-tested (external stages 442 s); R7 trained with fastMRI (rule risk) — owner decision pending |

Notes (2026-10-13 notebook scan):
- New 0.956–0.957 public notebooks all use heliosli's R7 compact model (fastMRI auxiliary training).
- Ties on the displayed 3-decimal score are ranked by the unrounded score; adding nartaa's 224 checkpoint at 35 % of
  the Raptor slot moved a 0.954 notebook from rank 437 to 320 (varshithakolupuri).

- **2026-10-14 00:40 local:** owner approved R7. EXP-103 (v65) moved to today's spare slot, 07:10 local. EXP-102 (v64) stays for 10-15.

## Final-pick rule (owner, 2026-10-14)
- **Pick 1 — top score:** highest public LB, any public weights (OAI- or fastMRI-trained allowed).
- **Pick 2 — rule-safe:** only models trained on competition data (+ public label tables, ImageNet backbones):
  our models and goodpjw's public reader. No nartaa (OAI), heliosli compact-v1 (training data not stated) or R7 (fastMRI).
  Default **EXP-074 (0.941)**; **EXP-104** replaces it only if it scores higher on the LB.

| EXP | Notebook | Blend | Gold-58 | Status |
|---|---|---|---|---|
| EXP-104 | v66 | rule-safe: seeds 0.35+0.35 + reader 5f 0.15 + goodpjw reader 0.15 | 0.9243 (seeds alone 0.9206) | smoke-tested (no nartaa stage, nartaa weight 0); for 10-15 |

Caution: gold gains from partners on top of the seeds did not transfer before (EXP-076 gold 0.9259 → LB 0.940).

## Final queue 2026-10-15 (owner approved routing 10-14 22:40; submits 07:06 local)
Per-finding nartaa/R7 routing on the EXP-103 base: R7-side = Lateral OA, PF OA, Lateral Meniscus, ACL;
nartaa-side = Effusion, Fracture, Medial Meniscus, Baker's; other findings 0.5/0.5. Builder: `--ext-hel-map`.

| EXP | Notebook | R7 weight (R7-side / nartaa-side) | Gold-58 (in-sample) | Status |
|---|---|---|---|---|
| EXP-104 | v66 | rule-safe (see above) | 0.9243 | smoke-tested |
| EXP-105 | v67 | 0.65 / 0.35 | 0.9404 | smoke-tested (external stages 461 s) |
| EXP-106 | v68 | 0.7 / 0.3 | 0.9418 | smoke-tested (465 s) |
| EXP-107 | v69 | 0.8 / 0.2 | 0.9439 | smoke-tested (430 s) |

EXP-103 gold 0.9365. Routing groups were chosen on gold, so the gold gains are optimistic; the LB decides.
Slot 5 kept spare (EXP-102 v64 can use it).

## Results (2026-10-15)
| EXP | Public LB | Note |
|---|---|---|
| EXP-104 | **0.943** | new rule-safe best (EXP-074 0.941) → pick 2 |
| EXP-105 | **0.957** | new best → pick 1; public rank 240/6,013 (silver line rank 300 at 0.956) |
| EXP-106 | 0.957 | rank did not move when it scored → unrounded ≤ EXP-105 |
| EXP-107 | 0.956 | routing too strong |

Routing gains on gold (0.9404 → 0.9439) did not keep growing on the LB: the mild 0.65/0.35 split is best.
Submitted late (07:39–09:23) after the 07:06 run stalled; slot 5 unused.
