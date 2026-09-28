# RSNA Knee Abnormality Detection 2026 — competition notes

Researched 2026-09-28 from the Kaggle overview/data/rules pages, ~40 discussion threads and the top
public notebooks. Links are relative to https://www.kaggle.com/competitions/rsna-knee-abnormality-detection.

## Facts

| | |
|---|---|
| Task | 12 binary findings per knee MRI study: ACL, MCL, Medial/Lateral Meniscus, Medial/Lateral/PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture |
| Metric | mean of the 12 per-label ROC AUCs (macro AUC) |
| Train | 4,407 studies, 24,371 series (~5.5/study, 20–45 slices, median 30), ~570 GB DICOM |
| Labels | **only 58 studies have radiologist labels**; every study has its free-text report (9+ languages) |
| Test | ~1,300 studies, images only (no report at test time) |
| Series meta | `Anatomical_Plane` (Sag/Cor/Ax), `Fluid_Sensitive`, `Fat_Suppression` (identical in train) |
| Format | code competition: GPU/CPU notebook ≤ 9 h, internet off, output `submission.csv` |
| Limits | 5 submissions/day, 2 final picks; team merge + entry deadline **2026-10-15**; final **2026-10-22** |
| Prizes | $77k: top-10 main LB + 3 efficiency prizes (score vs runtime; efficiency LB notebook by ryanholbrook) |
| Winner license | CC-BY-NC 4.0; must publish code + weights + a video |
| LB (2026-09-28) | #1 0.960, top-20 ≥ 0.955, ~200 teams sit at 0.941 by forking one public ensemble |

## Rulings from hosts (discussion)

* **Commercial LLM APIs are allowed** for extracting labels from reports (733965). Reports may go to OpenAI/Anthropic/Google etc.
* External data: non-commercial licences are fine; datasets behind a simple click-through registration are *generally* OK;
  datasets requiring institutional agreements are not. **KneeCoT is explicitly banned.** MRNet / SKM-TEA / fastMRI+ / OAI
  status is still being asked (743416) — check before relying on them.
* Label criteria (host overview 733343): borderline → negative. ACL/MCL = high-grade (>50% fibres) tears; menisci = surface-reaching
  tear on ≥2 images; OA = ≥1 cm area of >50% cartilage loss per compartment; Effusion & Baker's = **moderate or large** only;
  Contusion = marrow edema without fracture line; Fracture = acute cortical break.

## What the field has learned (the recipe)

1. **Labels are the bottleneck, not the backbone.** DINOv2-S → B: +0.001 (noise). Everyone trains on report-derived soft labels.
   * LLM extraction beats regex (0.878 vs 0.814 macro vs gold). Single-prompt LLM extraction ≈ as good as fancier pipelines.
   * Keep labels **soft** (hedged → ~0.7); rounding to 0/1 hurts.
   * "Not mentioned" must be a first-class answer, then filled **per finding**: silence ⇒ absent for Baker's/OA (3% / 0% positive when silent),
     but ⇒ unknown for Synovitis (34%) — fill synovitis from Effusion (P(syn|eff) 0.63 vs 0.22). Synovitis is everyone's worst label (0.6–0.8).
   * Fracture: 7% of reports vs 31% of gold positives — heavily under-reported.
2. **Pseudo-labelling with OUT-OF-FOLD teacher predictions** is the main lever of top single models (0.949–0.954):
   `y = 0.5·y_report + 0.5·y_oof` (some push further toward the model, esp. Synovitis). Multiple teachers from different label sources help.
   In-fold predictions just parrot labels back. Several people report CV gains that did *not* move LB — validate carefully.
3. **Model**: 2.5D — 3 physically adjacent slices as RGB, bag of K windows per study, **per-finding attention-MIL pooling**.
   Small backbones are enough: EffNet-B0 / ResNet @ 224–288 reach 0.94+ LB; CoAtNet-2 @ 352–384 ≈ 0.94 single fold;
   ConvNeXt, DINOv2/v3 and RadImageNet encoders add diversity (RadImageNet helped Synovitis 0.62 → 0.78 for one team).
   A LoRA-finetuned Qwen3.5-2B VLM @ 384 hit 0.950 single fold.
4. **Preprocessing details that matter**
   * Sort slices by `ImagePositionPatient · normal` — filenames are random UIDs; InstanceNumber is reversed in ~1/3 of series.
   * Physical crop 140–160 mm FOV around centre; 224–384 px; sample densely over the central 4–96 % of each series.
   * More eval windows helps until ~30–60 per study, then saturates.
   * **No left/right flips** (medial vs lateral labels; laterality missing in ~half the studies).
   * Mixed transfer syntaxes: install pylibjpeg + openjpeg (or gdcm).
5. **Validation**: CV on report labels of held-out folds (noise floor ~0.003) correlates well with LB; the 58 gold studies only
   resolve differences > ~0.02. Public LB is easier than local CV. Expect some private shake-up; don't tune blend weights on LB.
6. **Speed**: the best 0.944–0.949 solutions score in 5–13 min (threaded DICOM reads, fp16, one model per GPU) —
   worth it for the efficiency prize.

## Public assets worth using (CC0 unless noted)

| Asset | What |
|---|---|
| `stevenleehans/rsna-knee-llm-report-labels` | LLM labels; `llm_labels_v4_blend.csv` = **0.893** vs gold (best public), `llm_labels_full.csv` 0.878 |
| `pilkwang/rsna-knee-llm-labels` | LLM labels with `__verdict`/UNK, 0.866 |
| `lixin73/rsna-knee-llm-report-labels-sol56` | GPT-5.6-Sol labels, 0.835 |
| `dreaddevelopment/rsna-knee-labels` | soft labels (gold rows excluded) |
| `dreaddevelopment/knee-raptor-corpus(+ -ext)` | preprocessed 44×336×336 stacks for all 4,407 studies (16 GB) |
| `dreaddevelopment/raptor-knee-*` | CoAtNet "Raptor" weights (0.924–0.944) + other corpora |
| `barun2104/rsna-knee-mri-processed-3d-volumes`, `alenic/...-jpeg-224x224` | other preprocessed caches (17 GB / 11 GB) |
| RadImageNet weights (`shigechan/radimagenet-resnet50-official`) | medical-pretrained encoder |

Measured locally against the 58 gold studies (scripts/eval_labels.py): v4_blend 0.8927 · v2 0.8873 · full 0.8780 ·
pilkwang 0.8658 · GPT-5.6 0.8352 · this repo's default blend (v4_blend + full + pilkwang, per-finding silence priors) 0.8911.

## Plan to a competitive submission (≈3.5 weeks left)

1. **Day 1–2** — cache the data (Kaggle notebook or a cloud box with the DICOMs), train `exp=effnet_224` fold 0 on the default
   blended labels, push the inference kernel, get a first LB number (target ≥ 0.93).
2. **Labels** — run `src/labeling/llm_label.py` with 1–2 strong LLMs (cheap: < $5 per full pass reported), evaluate with
   `scripts/eval_labels.py`, add to `labels.sources`. Don't tune prompts on the 58 gold reports.
3. **Teacher → student** — 5-fold `exp=coatnet_384` (and/or effnet) teachers → `scripts/merge_oof.py` → `exp=teacher_pseudo`
   student. This is where 0.94 → 0.95+ has come from.
4. **Diversity** — ConvNeXt / DINOv2 / RadImageNet members, rank-average in the kernel (`--rank_average`).
5. **Final picks** — one strongest ensemble (main LB) + one fast single model (efficiency prize).
