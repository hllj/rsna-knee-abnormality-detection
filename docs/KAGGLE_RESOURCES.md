# Kaggle resources: labels, weights, recipes (drilled 2026-09-28)

Everything below is public on Kaggle and usable under the competition rules (external data/models
allowed if free and openly accessible). Gold AUC = macro AUC against the 58 radiologist-labelled
studies, measured locally with `scripts/eval_labels.py` (noise ≈ ±0.02 on 58 studies).

## 1. Report-derived label sets

| Dataset | File | Gold AUC | Notes |
|---|---|---|---|
| `stevenleehans/rsna-knee-llm-report-labels` | `llm_labels_v4_blend.csv` | **0.8927** | best public; synovitis imputed from effusion; 0.25 floor for silence |
| same | `llm_labels_v2.csv` | 0.8873 | v1 + synovitis-only imputation |
| same | `llm_labels_full.csv` | 0.8780 | raw LLM; 0.5 = "not addressed" (25% of cells) |
| `yunusgmsoy/rsna-knee-llm-report-labels` | `yunus_llm_labels.csv` | 0.8798 | independent LLM reading, useful for blending |
| `pilkwang/rsna-knee-llm-labels` | `report_labels_v2.csv` | 0.8658 | has `__conf` / `__verdict` (UNK); used by the whole public ensemble lineage |
| `laymond/rsna-knee-abnormality-qwen3-8b-weak-labels` | `qwen_knee_weak_labels.csv` | 0.843 (label×conf) | local Qwen3-8B; generator notebooks included (Qwen3-8B, Qwen2.5-32B) |
| `lixin73/rsna-knee-llm-report-labels-sol56` | `labels_llm_gpt56sol.csv` | 0.8352 | GPT-5.6-Sol |
| `dreaddevelopment/rsna-knee-labels` | `labels_llm_soft.parquet` | n/a | soft labels behind the Raptor weights; gold rows excluded by design |
| `barun2104/rsna-knee-stratified-folds-and-llm-soft-labels` | `train_folds.csv`, `train_folds_with_pseudo.csv` | n/a (gold copied in) | stratified folds + image pseudo-labels |
| `yehezkielhaganta/psuedo-labeling-knee` | `LLM_reasoning.csv` | n/a (gold copied in) | English translations + clinical reasoning per report — handy for prompt work |

Best practice reported by top single models (0.949–0.954): blend 2–3 label sources, keep soft, fill silence per
finding, then mix in **out-of-fold** image-teacher predictions (≈50/50, heavier for Synovitis). One strong team said
total LLM API spend was < $5, and cheap vs expensive LLMs made little difference.

## 2. Pretrained / competition weights

| Dataset | What | Reported |
|---|---|---|
| `dreaddevelopment/raptor-knee-maxspan`, `-native384`, `-native384dense`, `-finespacing`, `-widedense` | "Raptor" CoAtNet-RMLP-2 @336/384, per-finding attention-MIL, each with its preprocessing contract | 0.924–0.944 |
| `dreaddevelopment/knee-raptor-corpus` + `-ext` | all 4,407 studies preprocessed to 44×336×336 uint8 (16 GB) | — |
| `mattiaangeli/rsna-knee-coat-resgated-ep10-top3` | residual-gated CoAtNet (ep 4/6/8) + trainer package | in 0.941 |
| `mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2`, `-global96-top3` | CoAtNet 96-slice stack, depth zones / global gated head | in 0.943 |
| `mattiaangeli/knee-mri-fold-weights` | A5: 5-fold DINOv3 attention pooling | in 0.94x |
| `pilkwang/rsna-knee-weights` | 20 DINOv2 slot-attention members | in 0.94x |
| `marwanmath/resnet-50-radimagenet-marwan`, `shigechan/radimagenet-resnet50-official` | RadImageNet ResNet-50 (medical pretraining) | Synovitis 0.62 → 0.78 for one team |
| `antoinegg1/rsna-knee-e9-*`, `-e11-*`, `prvsiyan/rsna-knee-v52-radimagenet-heads-*` | RadImageNet heads | in 0.94x |
| `metaresearch/dinov2` (Kaggle Models) | DINOv2 S/B | — |
| `sadamtorres/rsna-ft-a/b` | 5-fold fine-tuned DINOv2 ViT-B @224/336 | 0.866 / 0.883 |

## 3. Public notebooks worth reading

**Inference ensembles (the public-notebook ceiling, 0.940–0.943; some titles claim up to 0.957)**
* `mattiaangeli/bend-the-knee-to-the-dinosaurs` — the base recipe: DINOv2×20 + DINOv3 A5 + RadImageNet +
  Raptor 4 views (60/10/10/20) + residual CoAtNet; CoAt branch 60% (Lateral Meniscus 100%), final percentile rank.
* `jiweiliu/rsna-knee-fast-2xt4-inference`, `pjmathematician/rsna-knee-d4-lite`, `mattiaangeli/bend-the-knee-to-speedy-raptors-the-original`
  — the 0.943 version: 4 CoAt members, rank of mean probabilities, 2×T4 in < 30 min (shared frozen DINO prefix).
* `evgendvorkin/rsna-versia-5` — best changelog of the lineage (0.937 → 0.943, which change did what).
* `prvsiyan/the-bee-s-knees-final-rsna-push` — readable audit of the 0.940 recipe + 5 failed variants
  (intensity-gain TTA regressed to 0.922).
* `starkhushi/rsna-knee-why-public-forks-stop-at-0-941` — why ~200 forks tie at 0.941.

**Training recipes**
* `dreaddevelopment/knee-mri-training-the-twelve-finding-model` — Raptor trainer: 3-slice windows, per-finding
  attention MIL, soft BCE, gold held out, top-3 epoch weight average. CoAtNet-2 @384, bs 8, 16 epochs ≈ 3 h on a 4090.
* `romantamrazov/rsna-knee-dinosaur-v5-train`, `sofiaanjenje/rsna-knee-e11-train` — DINOv2/v3 cross-series
  attention (12 finding queries over all series tokens + plane×FS type embedding) and RadImageNet head training.
* `sadamtorres/domain-adaptation-beats-resolution-dinov2-on-knee` — laterality from geometry, canonical right
  knee, slice-position + plane embeddings, CLS + patch-mean features; Qwen3-14B two-layer label extraction (0.881).
* `stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache` — build the cache on a **CPU** kernel (64 min, no GPU
  quota); crop 130 mm normalises 98.5% of series vs 39% at 160 mm; OOF understates ensemble gains (+0.024–0.031).
* `wguesdon/rsna-knee-dinov2-at-meniscus-resolution` — why ≥ 300 px matters for 1–3 mm tears; per-series window.
* `renta0426/rsna-knee-0-937-weak-label-dinov2-meniscus-resid` — meniscus-specialist residual.
* `barun2104/baseline-rsna-knee-mri-2-5d-resnet34`, `karnakbaevarthur/rsna-knee-eda-to-2-5d` — simple 2.5D baselines.
* `gchauhan/eda-rsna-knee-what-the-data-tells-us` — data traps (slice order, laterality, silent reports).

## 4. Key discussions

| Thread | Takeaway |
|---|---|
| 735304 Best single-model score | CoAtNet 224 single fold 0.950; ResNet 224 5-fold 0.954; Qwen3.5-2B VLM LoRA @384 0.950; Gemma-4 local labels → 0.949 single fold, 5-min inference; OOF-pseudo-label blending is the lever |
| 733932 "Not addressed" is a label | per-finding silence semantics; synovitis ← effusion (+0.009 macro) |
| 733965 / 733652 / 741819 / 743416 | commercial LLM APIs allowed; KneeCoT banned; OAI only without institutional sign-off (effectively no); external knee datasets mostly unusable |
| 737696 Raptor weights | window density matters (80 slices, 1.2% of stack per slice best); eval windows saturate ~30–60 |
| 743374 Fast submissions | < 10 min: threaded DICOM reads, raw PixelData decode, fp16, one model per GPU |
| 735154 Scaling encoder | DINOv2 S→B +0.001 (noise) |
| 737566 Synovitis | only ~16% of reports mention it; label AUC 0.79 vs gold; RadImageNet helps |
| 734004 / 733517 | scanner metadata ≈ 0.60–0.65 AUC alone; no shortcut |
| 735639 | ~7 studies contain both knees |
| 743148 | image→report→label round trip doesn't help; image teachers + 50/50 blend do |
| 743303 | workflow: preprocess cache on Kaggle, push as private dataset, train on Colab/cloud GPU |

## 5. Recommended approach (what this repo implements)

1. **Targets** — blend `llm_labels_v4_blend` + `llm_labels_full` + `report_labels_v2` (+ own LLM run) with
   per-finding silence priors → then OOF pseudo-label rounds.
2. **Input** — per-study stack from 6 slots (plane × fluid), canonical right knee, 130–140 mm crop, 224–384 px,
   dense slice sampling over 4–96% of each series.
3. **Model** — 2.5D: 3 adjacent slices → RGB; bag of K windows; timm backbone + slot embedding + per-finding gated
   attention MIL. Start EfficientNet-B0 @224 (fast, 0.94+ reported), then CoAtNet-2 @384, ConvNeXt, DINOv2, RadImageNet.
4. **Validate** — held-out report-label fold (noise ≈ 0.003) + gold 58 as a coarse sanity check.
5. **Submit** — rank-average diverse members; optionally blend with the public 0.943 ensemble for the main LB,
   and keep one fast single model for the efficiency prize.

## 6. Dry run on Kaggle (2026-09-28)

Kernels: `hlly34/rsna-knee-dryrun-train` (GPU T4, internet on) → `hlly34/rsna-knee-inference` (GPU, internet OFF).
Code is uploaded as the private dataset `hlly34/rsna-knee-code`.

| Step | Result |
|---|---|
| Kaggle mounts | `/kaggle/input/competitions/<slug>`, `/kaggle/input/datasets/<user>/<slug>`, `/kaggle/input/notebooks/<user>/<kernel>` — auto-detected |
| Preprocess 58 gold + 600 random studies (4 CPUs) | 693–971 s → ~1.3 s/study; full train set ≈ 1.5 h on a CPU kernel |
| Knee side from geometry | R 343 / L 301 / unknown 14; slot coverage 100% (same-plane fallback) |
| Train EfficientNet-B0 @224, fold 0, 485 train studies, 8 epochs | 394 s on one T4 |
| Held-out report-label AUC (115 studies) | 0.762 |
| Gold-58 AUC (image model / report labels) | **0.771** / 0.893 — model already beats labels on Effusion, Lateral OA, Synovitis |
| Test inference (3 public test studies) | 31 s in the dry-run kernel; 79 s offline incl. wheel install |

Not the model's ceiling: 11% of the data and 8 epochs. Next run: full cache, 5 folds, 15–20 epochs.
