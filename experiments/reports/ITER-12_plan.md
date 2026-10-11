# ITER-12 plan — from 0.941 to a medal (2026-10-11)

## 1. Where we stand
- Best rule-safe: **0.941** (EXP-074: 96-slice Raptor v2 seeds 1+2; EXP-082 adds Raptor v3 and ties).
- Public LB, 5,726 teams (2026-10-11 07:37 UTC):

  | Position | Public LB |
  |---|---|
  | #1 | 0.965 |
  | top 1 % (#57) | 0.958 |
  | top 5 % (#286, silver) | 0.954 |
  | **top 10 % (#572, bronze)** | **0.950** |
  | us | 0.941, about #2,300 |

  1,079 teams score ≥ 0.950, so bronze needs **≥ 0.951** to be safe; silver ~0.954.

## 2. New findings
1. **OAI is allowed under the host's ruling.**
   - Host (Po-Hao Chen, 741819, 2026-09-21): OAI "does not violate the challenge rules if it is generally accessible to all
     participants without an escalated (i.e. institutional) review and sign-off … and is free of charge".
   - Evidence 2026-10-08/09 (743416): the NDA lists OAI with "Institutional Approval Required: No"; individuals are approved
     immediately via login.gov; NDA says access is open to everyone.
   - The host's Aug 27 principle also applies: click-through agreements OK, institutional approvals not.
   - nartaa's OAI-trained public weights therefore look usable. Residual risk: no explicit host "yes" for OAI itself.
2. **All 50 top public notebooks (0.950–0.954) attach nartaa's OAI weights.**
   - 41 of them also use goodpjw's 2.5D ConvNeXt reader; 10–11 use heliosli's ConvNeXt-S "compact" model.
   - heliosli: nartaa 0.7 + its own 4-fold ConvNeXt-S 0.3, weights fixed in advance → 0.951.
   - rabari (0.954): nartaa-384 + nartaa-224-crop + goodpjw reader, per-finding routing (ConvNeXt ~0.5 on Lateral OA,
     ~0.02 on Synovitis), probit fusion.
3. **Our 96-slice family is nartaa's recipe without OAI**, so it adds little to nartaa. EXP-069 (nartaa 0.65 + our blend
   with Raptor members) dropped to 0.948. Partners that help nartaa are *different* architectures (ConvNeXt, readers).
4. **Per-finding complementarity on our gold run** (`gold_perfinding.py`):
   - The 96-slice seeds lead on ACL, MCL, menisci and Medial OA.
   - The reader leads on Lateral OA +0.027, PF OA +0.023, Fracture +0.019 and Contusion +0.010.
   - CoAtNet-dense f0 leads on Effusion +0.034 and Synovitis +0.022.
   - This matches rabari's routing.
5. **Calibration (GOLD58_CALIBRATION_2026-10-11.html):**
   - Gold predicts the LB well within a family (r 0.94) but has family offsets, and gold-tuned weights overfit (EXP-076).
   - So use coarse, pre-set weights: finding groups, not 12 tuned weights.

## 3. Strategy: two tracks, two final picks
**Track A — OAI-anchored ensemble (medal track).** Anchor = nartaa 0.949. Add our *different-architecture* models, plus
public partners where they are clearly allowed (goodpjw reader, nartaa 224-crop).
**Track B — rule-safe insurance.** Our own 96-slice family, in case OAI is ruled out before the end.
**Final picks (2026-10-22):** #1 best Track A blend, #2 best Track B blend.

## 4. Experiments
### Track A (submissions; no GPU training needed for A1–A3)
| EXP | What | Weights (rank fusion) | Question |
|---|---|---|---|
| EXP-077 | nartaa + our 96-slice seeds (EXP-074) | 0.8 / 0.2 | Control: does our correlated family add anything? (built 14:37 today, submits 10-12 07:05) |
| EXP-083 | nartaa + **ConvNeXt dense 5f** | 0.7 / 0.3 | heliosli recipe with our ConvNeXt: target ≥ 0.951 |
| EXP-084 | nartaa + ConvNeXt dense 5f + reader v1 f0 | 0.7 / 0.15 / 0.15 | Does the reader (best on OA / fracture) add on top? |
| EXP-085 | EXP-084 with **finding-group routing** | partners 0.4 on Lateral OA, PF OA, Medial OA, Fracture, Contusion; 0.25 on Effusion / Synovitis (CoAtNet f0 in); 0.15 elsewhere | Coarse routing from structure, not gold-fitted: target 0.953+ |
| EXP-086 | best of 083–085 + nartaa 224-crop (0.1 of the anchor) | — | Second view of the anchor (rabari) |

- Runtime: nartaa ~0.5 h + our ConvNeXt 5f ~1 h + reader ~0.5 h, well inside 9 h.
- Mirror TTA only on members that allow it (reader yes; ConvNeXt / CoAtNet dense no).

### Track A training (one 3090)
| EXP | What | Cost |
|---|---|---|
| EXP-087 | Pseudo-label ConvNeXt student **folds 1–4** (EXP-079 recipe; fold 0 already +0.003 LB) → replaces ConvNeXt dense 5f as the main partner | ~6 h, ~$1.5 |
| EXP-088 | Reader v1 folds 1–4, if not already trained (EXP-052?), for a 5-fold reader partner | ~4 h, ~$1 |

### Track B (rule-safe)
| EXP | What | Cost |
|---|---|---|
| EXP-089 | 96-slice Raptor v2 **5-fold** (honest OOF + 5 members + teacher for the next pseudo-label round) | ~22 GPU-h, ~$6 (2 boxes) |
| EXP-090 | 96-slice 3rd seed | ~$1 |
| EXP-091 | Rule-safe blend: 96-slice seeds + pseudo ConvNeXt 5f + reader with the same finding-group routing | free |

### Stretch (decide by 10-13): our own OAI model
nartaa's table: OAI examples with masked labels **+0.005**, 384 full-field **+0.003**.
- Needs the owner to register at the NDA (login.gov + click-through), download OAI knee MRIs and map the
  semi-quantitative reads to our 12 findings (masked where unmapped). 4–6 days.
- Result: a second, decorrelated OAI-trained model. Highest ceiling, highest effort.

## 5. Schedule (5 submissions / day)
| Day | Submissions | Training |
|---|---|---|
| 10-12 | EXP-077 (scheduled), EXP-083, EXP-084, EXP-085, EXP-086 | EXP-087 (+ EXP-088) on one 3090 |
| 10-13 | best Track A + pseudo ConvNeXt 5f; Track B checks | EXP-089 on two 3090s; stretch decision |
| 10-14 → 10-15 | Track A refinements; **team merge deadline 10-15** | EXP-090 |
| 10-16 → 10-21 | consolidate both tracks; avoid weight churn (public-LB overfit) | — |
| 10-22 | final picks: best Track A + best Track B | — |

## 6. Rules for blending (from the calibration report)
- Pre-set weights only: anchor share 0.6–0.8, finding-group routing with 2–3 groups. No per-finding gold tuning.
- A change worth a slot should predict ≥ +0.002. One LB tick (0.001) is noise.
- Keep runtime < 6 h (safety margin for the 9 h limit).
