# ITER-09 plan — build on the 96-slice Raptor v2 (2026-10-10)

## 1. Where we are (public LB)
| EXP | What | LB |
|---|---|---|
| EXP-069 | OAI fallback: 0.65 × nartaa + 0.35 × EXP-055 | 0.948 (nartaa alone 0.949) |
| **EXP-068** | Raptor v2 all-data, 96-slice 384 cache, mirror TTA (EXP-065) | **0.938** best rule-safe |
| EXP-055 / 070 | ConvNeXt 5f + Raptor repro 5f + reader + CoAtNet (070: + reader v3) | 0.936 / 0.936 |
| EXP-067 | Raptor v2 all-data, 80-slice cache | 0.933 |
| EXP-058 | Raptor v1 all-data | 0.930 |
| EXP-071 | 0.6 × EXP-055 + 0.4 × EXP-065, mirror TTA | pending |

What moved the LB: recipe (EXP-058 → 067, +0.003) and the 96-slice native-384 cache (067 → 068, +0.005).
What did not: more runs of a family already in the blend (EXP-059, EXP-070), per-finding weights.

## 2. The two options
**A. Blend the OAI fallback (nartaa) with EXP-068 and EXP-055.** Low value.
- EXP-069 showed that our 0.936 blend at 35% *lowers* nartaa (0.949 → 0.948): our members are 0.01+ weaker.
- EXP-068 is stronger (0.938) but uses nartaa's own recipe and 96-slice geometry, minus OAI. It is likely highly
  correlated with nartaa, so it adds little diversity. Versia-5 got +0.001 from a reader at 3–35% per finding.
- The OAI rule is still unanswered (743416). Worth one submission at most: **nartaa 0.85 + EXP-068 0.15**, no EXP-055.

**B. Retrain on the best recipe and data.** The main track, but targeted. Do not retrain all of EXP-055.
- Retraining ConvNeXt 5f / reader / CoAtNet on the old caches repeats what EXP-059/070 showed: the same families saturate.
- Put the GPU hours where the gains came from: **the 96-slice cache + the v2 recipe**.
  1. **Seed averaging of the best model:** `raptor_v2_d96_alldata` seed 2 (then 3). A second seed gave +0.003 CV for
     a single model before; here it is the dominant member, so it should carry to LB.
  2. **A second backbone on the same data for diversity:** ConvNeXt-S (our strongest other family, 0.930 at 5f) on the
     96-slice cache with the v2 recipe, all-data, best-3-by-gold.
  3. Reader on the new data: skip (v2/v3 added nothing; costly).

## 3. Gate on EXP-071 (scores ~07:00–07:30 UTC 2026-10-10)
- **EXP-071 ≥ 0.940:** the old families still add diversity. The final rule-safe pick = EXP-055 families + the new
  96-slice members (EXP-068 + seeds + ConvNeXt-d96).
- **EXP-071 ≤ 0.938:** the old families are dead weight next to EXP-068. The final pick = the 96-slice family only
  (Raptor v2 seeds + ConvNeXt-d96).

## 4. Experiments
| EXP | Kind | What | Where / when |
|---|---|---|---|
| EXP-072 | train | `raptor_v2_d96_alldata` seed 2 (`seed=7`), ~255 min | 3090 box A |
| EXP-073 | train | `convnext_v2_d96_alldata`: ConvNeXt-S @384 on cache_384_d96, v2 recipe, best-3-by-gold | 3090 box B |
| EXP-074 | submit | EXP-065 + EXP-072 averaged (Raptor v2 d96 × 2 seeds), mirror TTA | 10-11 |
| EXP-075 | submit | gated: best rule-safe blend with the new 96-slice members (section 3) | 10-11 |
| EXP-076 | submit (optional) | OAI: nartaa 0.85 + EXP-068 0.15 | 10-11, if a slot is spare |
| EXP-077 | train | seed 3 / ConvNeXt fold models, only if EXP-074 gains | 10-12 |

## 5. Cost and time
- Two 3090s in parallel, ~4.5 h each incl. the 44 GB cache fetch. Prefer hosts with low download fees (Poland
  31250690: $0.0026/GB vs $0.039/GB in Estonia, about $0.11 vs $1.7 for the cache). Likely a few dollars per run
  including storage; check the live price before renting. Needs a credit top-up.
- Deadline 2026-10-22 (team merge 10-15): two training rounds fit comfortably; keep 2 final picks =
  best rule-safe blend + (if OAI is allowed) the nartaa-based fallback.
