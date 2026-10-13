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
