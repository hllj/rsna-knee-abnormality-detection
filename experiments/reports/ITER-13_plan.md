# ITER-13 — status and plan to the final picks (2026-10-14)

## 1. Where we are
| Date | Best | What moved it |
|---|---|---|
| 10-09 | 0.936 | our blend (ConvNeXt 5f + Raptor 5f + reader + CoAtNet) |
| 10-11 | 0.941 | our 96-slice Raptor v2, 2 seeds (rule-safe best) |
| 10-12 | 0.949 | nartaa OAI CoAtNet (public) + our seeds |
| 10-13 | 0.954 | nartaa 0.7 + heliosli compact-v1 0.3 |
| 10-14 | **0.956** | **nartaa 0.5 (with 224 px) + heliosli R7 0.5 (EXP-103)** |

- Public LB 10-14: **rank 282 of 5,953**. Silver line (top 5 %) = rank 297 at 0.956; bronze (top 10 %) = rank 595 at 0.956.
- 719 teams show 0.956 and 184 are above. Inside the block, rank follows the unrounded score.
- The medal lines moved 0.950 → 0.955 → 0.956 in three days. Expect about 0.957 by 10-22.

## 2. What the results say
1. **Public partner models moved the score; ours did not, once nartaa was in.**
   - Adding our models (ConvNeXt, readers, CoAtNet, 96-slice seeds) to nartaa blends cost 0.001 or was neutral (EXP-083..086, 094..096).
   - Per-finding routing, probit fusion, goodpjw on two findings and heliosli 0.4 all held 0.954 (EXP-097..101).
2. **Gold-58 cannot rank nartaa-anchored blends.** It predicts our own models well (r 0.94), so it still guides the
   rule-safe track.
3. **Displayed scores are rounded.** Gains of +0.0005 are invisible, but they move rank inside a tied block.
   Our leaderboard rank only shows our best submission, so a new variant visibly helps only if it beats EXP-103's
   unrounded score.
4. **Private-LB risk:** a 900-team block within 0.002 will reshuffle. Robust choices matter more than public-LB tuning.

## 3. Final picks (rule set 10-14)
- **Pick 1 — top score:** EXP-103 (0.956), or a variant if it ranks higher.
- **Pick 2 — rule-safe:** EXP-074 (0.941, competition-data models only), or EXP-104 if it beats it.

## 4. Plan to 10-22 (5 submissions/day, about 40 left)
**A. Push pick 1 inside the 0.956 block (no training, low cost)** — small, principled changes on the EXP-103 base:
| EXP | Change | Why |
|---|---|---|
| EXP-105 | Lateral OA: our reader 5f 0.5 (nartaa/R7 share the rest) | nartaa's weakest finding; readers lead on gold |
| EXP-106 | goodpjw reader 0.3 on Lateral OA + Contusion | its only two strong findings on gold |
| EXP-107 | nartaa 0.5 + R7 0.35 + heliosli compact-v1 0.15 | two different heliosli models |
| EXP-108 | 224-crop share 0.35 → 0.5 in the nartaa slot | the cheapest documented in-block gain |

Judge by our rank after each score. Stop tuning after about 2 rounds to avoid fitting the public LB.

**B. Rule-safe pick 2**
- 10-15: EXP-104 (seeds + both readers).
- Optional training (1 × 3090, about $1–1.5 each): a 3rd 96-slice seed or a 96-slice 5-fold for pick 2.
  Seed averaging gave +0.003 at 2 seeds.

**C. Watch the public code every 1–2 days** for new partner models (heliosli R8, new readers) and rule clarifications.

**D. Team merge (deadline 10-15):** only worth it with a team whose models are not the same public blend.

**E. 10-21/22:** lock the picks. Pick 1 = best rank among recipes fixed in advance; pick 2 = best rule-safe LB.
