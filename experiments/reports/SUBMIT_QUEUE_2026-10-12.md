# Submission queue — 2026-10-12 (ITER-12, Track A: nartaa-anchored blends)

| # | Local | EXP | Notebook | Blend (rank fusion) | Question |
|---|---|---|---|---|---|
| 1 | 07:05 | EXP-077 | v47 | nartaa 0.8 + EXP-074 (96-slice seeds 1+2, mirror) 0.2 | Control: does our nartaa-like family add anything? |
| 2 | 07:06 | EXP-083 | v48 | nartaa 0.7 + ConvNeXt dense 5f 0.3 | heliosli recipe with our ConvNeXt (target ≥ 0.951) |
| 3 | 07:07 | EXP-084 | v49 | nartaa 0.7 + (ConvNeXt 5f 0.5 + reader v1 f0 0.5, mirror on reader) 0.3 | Does the reader add on top? |
| 4 | 07:08 | EXP-085 | v50 | Routing — nartaa 0.6 on Medial/Lateral/PF OA, Fracture, Contusion; 0.75 on Effusion/Synovitis; 0.85 elsewhere. Ours = ConvNeXt 5f 0.4 + reader 0.4 + CoAtNet f0 0.2 | Do pre-set finding groups beat a single weight? |
| 5 | 07:09 | EXP-086 | v51 | EXP-085 + nartaa 0.945 224-crop as 0.1 of the anchor | Second view of the anchor (rabari) |

All verified on the public test: external stage(s) ran, nartaa checkpoint hashes verified, fused 3 × 12.
Runtime ≈ 0.5 h (nartaa) + 1–2.5 h (ours). Partners trained today (EXP-087 pseudo ConvNeXt folds 1–4,
EXP-088 reader folds 1–4) replace the single-fold / old 5-fold partners in the next round.
