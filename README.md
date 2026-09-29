# RSNA Knee Abnormality Detection 2026

Training / inference pipeline for [rsna-knee-abnormality-detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
(12 knee-MRI findings, macro AUC, code competition), built on
[hllj/kaggle-boilerplate](https://github.com/hllj/kaggle-boilerplate).

| Read | For |
|---|---|
| [docs/COMPETITION_NOTES.md](docs/COMPETITION_NOTES.md) | rules, host rulings, what works, plan |
| [docs/KAGGLE_RESOURCES.md](docs/KAGGLE_RESOURCES.md) | public labels / weights / notebooks / discussions, scored |
| [docs/VAST.md](docs/VAST.md) | details of the rented-GPU workflow below |
| [experiments/EXPERIMENTS.md](experiments/EXPERIMENTS.md) | every experiment: CV, gold-58, public LB |

**Current best: EXP-009 — EfficientNet-B0 @224, 5 folds, rank-averaged → public LB 0.914**
(OOF 0.867 vs report labels, gold-58 0.893). Full history in [experiments/EXPERIMENTS.md](experiments/EXPERIMENTS.md).

| Exp | What | OOF | Gold-58 | Public LB |
|---|---|---|---|---|
| EXP-000 | dry run: fold 0, 485 studies, 8 epochs | 0.762 | 0.771 | 0.791 |
| EXP-007 | fold 0 only, full data, 15 epochs | 0.865 | 0.876 | 0.907 |
| EXP-008 | fold 1 only (same setup) | 0.874 | 0.877 | 0.905 |
| EXP-009 | all 5 folds, rank-averaged | 0.867 | 0.893 | **0.914** |
| EXP-010 | student, 50/50 pseudo-labels (fold 0) | 0.845 | 0.851 | — (stopped: worse) |
| EXP-012 | EXP-009 checkpoints via fast streaming inference (~11 min instead of ~30) | 0.867 | 0.893 | **0.914** |

Iteration reports: [ITER-01 baseline](experiments/reports/ITER-01_2026-09-28.md) ·
[ITER-01b inference speed-up](experiments/reports/ITER-01b_inference_speedup.md) ·
[ITER-02 plan (resolution / backbone)](experiments/reports/ITER-02_plan.md) ·
[ITER-02 fold-0 results: ConvNeXt-S @288 wins](experiments/reports/ITER-02_2026-09-29.md).

---

## How every change is made (iteration loop)

Full commands in [docs/WORKFLOW.md](docs/WORKFLOW.md); the same rules are in [CLAUDE.md](CLAUDE.md).

| Step | Do | Command | Output |
|---|---|---|---|
| 1 | **Train 1 fold, one architecture**; gate vs current best fold 0 (OOF 0.865 / gold 0.876): continue only if OOF ≥ best − 0.003 and gold not down > 0.02 | `bash scripts/vast/train.sh <config> 0` | `models/<exp>/fold0.pt`, fold-0 OOF + gold |
| 2 | **Submit fold 0, then the 5-fold ensemble** (train folds 1-4 in between); check the [inference time budget](docs/WORKFLOW.md#inference-time-budget-check-at-step-2) — 5 × EfficientNet-B0 ≈ 30–45 min of the 9 h limit | `bash scripts/vast/submit.sh EXP-NNN "<msg>" <ckpts>` | 2 LB scores |
| 3 | **Update results + CV–LB** | `python scripts/track.py lb` · `python scripts/track.py add EXP-NNN --outputs runs/EXP-NNN` | `experiments/EXPERIMENTS.md` incl. *CV vs LB* |
| 4 | **Back up models** (one Kaggle Model per architecture, one variation per fold), then destroy the GPU box | `python scripts/upload_models.py --model <slug> --exp EXP-NNN runs/EXP-NNN/models/<exp>/fold*.pt` | Kaggle Model variations |
| 5 | **Report: result, CV–LB reading, keep/drop, next steps, cost** | copy `experiments/reports/TEMPLATE.md` → `ITER-NN_<date>.md` | iteration report |

A change that fails the step-1 gate is recorded as NEGATIVE and goes straight to step 5.

---

## 0. What already exists on Kaggle (account `hlly34`)

| Kaggle object | What it is | Status |
|---|---|---|
| dataset `hlly34/rsna-knee-code` | this repo's `src/ configs/ scripts/ submission/` | refresh with `scripts/upload_code.sh` |
| notebook `rsna-knee-cache` | all 4,407 training studies preprocessed (EXP-001) | done |
| notebook `rsna-knee-cache-pack` | same cache as 14 tar shards, 14.5 GiB (for downloading) | done |
| notebook `rsna-knee-dryrun-train` | subset run, EXP-000 → **public LB 0.791** | done |
| notebook `rsna-knee-train-teacher` | EfficientNet-B0 teacher (EXP-002) | see registry |
| notebook `rsna-knee-inference` | offline ensemble → `submission.csv` | used for every submission |

You never need the 570 GB of DICOMs: training uses the cache, and the hidden test set is read inside the
Kaggle inference notebook.

---

## Path A — train on a rented GPU (vast.ai RTX 3090), submit on Kaggle  ← recommended

Copy-paste in order. Every step prints what to run next.

### A1. Rent and log in
vast.ai → PyTorch (CUDA 12.x) template, 1× RTX 3090, **≥ 8 CPU cores, ≥ 32 GB RAM, ≥ 60 GB disk**.

```bash
# on your laptop: give the box your Kaggle credentials
scp -P <PORT> ~/.kaggle/kaggle.json root@<HOST>:~/.kaggle/kaggle.json
ssh -p <PORT> root@<HOST>
tmux new -s knee                     # keeps training alive if SSH drops (re-attach: tmux a -t knee)
```

### A2. Get the code and install
```bash
pip install -q kaggle
kaggle datasets download hlly34/rsna-knee-code -p rsna-knee --unzip
cd rsna-knee
bash scripts/vast/setup.sh           # installs deps, prints GPU / CPU / RAM / disk
```

### A3. Download data (~15 GB, cache checksum-verified)
```bash
bash scripts/vast/get_data.sh        # -> data/raw (csvs), data/labels, data/processed/cache_256 (4,407 .npz)
                                     # ends by printing each label source's AUC vs the 58 gold studies
```

### A4. Train the teacher — fold 0 first
```bash
bash scripts/vast/train.sh effnet_224 0
```
Look at the last lines: `oof_auc` (report labels, held-out fold) and `gold_auc_*` (58 radiologist labels).
Reference from the Kaggle runs at epoch 11: report ≈ 0.867, gold ≈ 0.88. If fold 0 looks right:

```bash
bash scripts/vast/train.sh effnet_224 "1 2 3 4"          # remaining folds (appends to the same CV summary)
```
Outputs: `models/effnet_b0_224/fold{0..4}.pt`, `models/oof/effnet_b0_224.csv` (teacher OOF),
`models/oof/effnet_b0_224_cv.json` (CV summary).

### A5. Submit the teacher
```bash
NO_SUBMIT=1 bash scripts/vast/submit.sh EXP-006 "teacher effnet fold0 (vast)" models/effnet_b0_224/fold0.pt   # optional dry check
bash scripts/vast/submit.sh EXP-007 "teacher effnet 5 folds (vast)" models/effnet_b0_224/fold*.pt
```
`submit.sh` uploads the weights as `hlly34/rsna-knee-weights`, points `rsna-knee-inference` at it, runs it on
Kaggle (GPU, internet off), waits for success, submits `submission.csv` with the message `EXP-007 …`, and adds
the row to `experiments/registry.csv`. Scoring takes 30–60 min, then:
```bash
python scripts/track.py lb           # pulls public LB scores into the registry / EXPERIMENTS.md
```

### A6. Student round (pseudo-labels from the teacher's out-of-fold predictions)
> **Measured (EXP-010): the default 50/50 mix is worse than the teacher** on fold 0
> (OOF 0.845 vs 0.865, gold 0.851 vs 0.876). Check fold 0 first and compare its
> `models/oof/student_pseudo_oof_fold0.csv` against the teacher on report labels (not the student's own
> `val_auc_report`, which is scored against the pseudo-blended targets). Try a smaller weight first:
> `labels.pseudo.weight=0.2` plus the three per-finding weights set to 0.2 (OmegaConf merges dicts, so `{}` would not clear them).
```bash
bash scripts/vast/train.sh teacher_pseudo 0 labels.pseudo.weight=0.2 labels.pseudo.per_label_weight.Synovitis=0.2 \
    "labels.pseudo.per_label_weight.PF OA=0.2" "labels.pseudo.per_label_weight.Lateral OA=0.2"          # fold 0 check
bash scripts/vast/train.sh teacher_pseudo "0 1 2 3 4"    # uses models/oof/effnet_b0_224.csv by default
bash scripts/vast/submit.sh EXP-008 "teacher+student 10-model ensemble" \
    models/effnet_b0_224/fold*.pt models/student_pseudo/fold*.pt
python scripts/track.py lb
```

### A7. More diversity (optional, same commands)
```bash
bash scripts/vast/train.sh convnext_288 "0 1 2 3 4" preprocess.res=256
bash scripts/vast/train.sh coatnet_384  "0 1 2 3 4" preprocess.res=256 training.batch_size=6
bash scripts/vast/submit.sh EXP-009 "effnet+student+convnext+coatnet" models/*/fold*.pt
```
The cache is 256 px, so configs whose `preprocess.res` differs must override it as shown (the model still sees
`model.img_size`). Add any other override the same way, e.g. `training.num_epochs=20 training.lr=5e-4`.

### A8. Bring results home
```bash
# on your laptop, inside the repo
scp -P <PORT> root@<HOST>:~/rsna-knee/experiments/registry.csv experiments/registry.csv
python scripts/track.py render && git add experiments && git commit -m "experiments: EXP-00x"
```

### A9. Back up every checkpoint to Kaggle Models (one variation per fold)
```bash
python scripts/upload_models.py --model rsna-knee-effnet-b0-224 --exp EXP-009 runs/EXP-009/models/effnet_b0_224/fold*.pt
```
Backed up so far (private): [hlly34/rsna-knee-effnet-b0-224](https://www.kaggle.com/models/hlly34/rsna-knee-effnet-b0-224)
→ PyTorch variations `exp-009-fold0` … `exp-009-fold4` (the 0.914 ensemble) and `exp-000-fold0` (dry run).
Attach a variation to any Kaggle notebook to reuse it; each carries `fold<k>.pt`, `fold<k>_metrics.json` and usage notes.

---

## Path B — everything inside Kaggle notebooks (T4, ~30 GPU-h/week, max 2 GPU sessions)

Notebooks are generated by `submission/build_kernels.py` — edit the `EXPERIMENTS` dict there (ids, folds,
epochs), never on kaggle.com.

```bash
KAGGLE_BIN=kaggle bash scripts/upload_code.sh "what changed"   # 1. publish code
python submission/build_kernels.py                              # 2. regenerate notebooks
kaggle kernels push -p submission/train-teacher                 # 3. teacher (fold list in EXPERIMENTS)
kaggle kernels status hlly34/rsna-knee-train-teacher            #    wait for COMPLETE
kaggle kernels push -p submission/train-student                 # 4. student (reads teacher OOF)
kaggle kernels push -p submission/inference                     # 5. ensemble of teacher + student outputs
kaggle competitions submit rsna-knee-abnormality-detection \
    -k hlly34/rsna-knee-inference -v <VERSION> -f submission.csv -m "EXP-004 <description>"   # 6. submit
python scripts/track.py lb                                      # 7. record LB
```

| Notebook | Hardware | Input | Output | Exp |
|---|---|---|---|---|
| `rsna-knee-dryrun-train` | T4, internet | competition, labels | 1 fold on 658 studies | EXP-000 |
| `rsna-knee-cache` | CPU | competition | `cache_256/*.npz` | EXP-001 |
| `rsna-knee-cache-pack` | CPU | cache | 14 tar shards | EXP-001 |
| `rsna-knee-train-teacher` | T4×2, internet | cache, labels | fold models, OOF, CV | EXP-002 |
| `rsna-knee-train-student` | T4×2, internet | cache, labels, teacher OOF | fold models, CV | EXP-003 |
| `rsna-knee-inference` | T4, internet **off** | teacher+student outputs *or* `hlly34/rsna-knee-weights` | `submission.csv` | EXP-004+ |

---

## Experiment workflow (both paths)

1. **Pick the next id** — see the last row of [experiments/EXPERIMENTS.md](experiments/EXPERIMENTS.md).
2. **Run** — the id goes in the submission message (`submit.sh EXP-NNN …`) or in `build_kernels.py → EXPERIMENTS`;
   notebooks print it and write `experiment.json`.
3. **Register** — `submit.sh` does it; for Kaggle notebooks:
   `kaggle kernels output hlly34/<notebook> -p runs/EXP-NNN && python scripts/track.py add EXP-NNN --outputs runs/EXP-NNN --kernel hlly34/<notebook> --version <v> --notes "..."`.
4. **Score** — `python scripts/track.py lb` after scoring.

How to read the numbers:
* `oof_auc` — out-of-fold predictions vs report-derived targets; compare experiments on this (noise ≈ 0.003).
* `gold_auc_*` — vs the 58 radiologist-labelled studies; a coarse sanity check (noise ≈ ±0.02).
* `lb_public` — Kaggle public leaderboard. Don't tune on it: 5 submissions/day, small public split.

---

## Local development

```bash
conda env create -f environment.yml && conda activate rsna-knee && pip install -e . && pre-commit install
bash scripts/download_data.sh          # csvs + public label sets
python scripts/eval_labels.py          # label sources and the configured blend vs gold-58
pytest -m "not slow"                   # unit tests
pytest -m slow                         # end-to-end on synthetic DICOMs: preprocess → train → CV merge → inference
```

Own LLM labels (commercial APIs are allowed by the host):
```bash
export OPENAI_API_KEY=...              # or OPENAI_BASE_URL=http://localhost:8000/v1 for a local vLLM server
python -m src.labeling.llm_label --model <model-id> --out data/labels/my_llm_labels.csv
python scripts/eval_labels.py data/labels/my_llm_labels.csv
# then add my_llm_labels.csv to labels.sources in configs/default.yaml
```

---

## Reference

### Approach
```
reports ──LLM──► soft labels (3 public sources) ──per-finding silence priors + blend──► targets ◄── teacher OOF (student round)
DICOM ──position sort, canonical right knee, 140 mm crop, 1–99% window──► per-study stack: 6 slots × 72 slices × 256²
bag of K windows (3 adjacent slices = RGB) ─► timm backbone ─► + slot embedding ─► per-finding attention pooling ─► 12 logits
folds (configs/folds_5.csv) ─► top-3 epoch weight average per fold ─► rank-average all fold models at inference
```

### Layout
```
configs/default.yaml         all settings; configs/exp/*.yaml override them (effnet_224, coatnet_384, convnext_288,
                             dinov2_s_224, teacher_pseudo). Any key can be overridden on the CLI: training.lr=2e-4
configs/folds_5.csv          the fold split every run uses (58 gold studies = fold -1, never trained on)
src/dataset/dicom.py         DICOM → cached stack (same code for the training cache and test inference)
src/dataset/datasets.py      bag-of-windows Dataset;  augmentations.py: study-consistent affine + intensity, no flips
src/labeling/blend.py        label sources → silence handling → blend → optional pseudo-label mix
src/labeling/llm_label.py    LLM report labeller (any OpenAI-compatible endpoint)
src/models/model.py          KneeMILModel;  train_pl.py: Lightning training + OOF/gold preds + metrics json
src/models/inference.py      test DICOM → cache → ensemble → submission.csv
scripts/vast/                setup.sh, get_data.sh, train.sh, submit.sh (Path A)
scripts/                     preprocess.py, eval_labels.py, merge_oof.py (CV summary), track.py (registry),
                             upload_code.sh (publish code dataset), download_data.sh
submission/build_kernels.py  generates every Kaggle notebook (Path B) and their kernel-metadata.json
experiments/                 registry.csv (source of truth) + EXPERIMENTS.md (generated table)
tests/                       unit tests + end-to-end smoke test
```

### Outputs of one training run
```
models/<exp_name>/fold<k>.pt              weight-averaged model + its full config (inference rebuilds from it)
models/<exp_name>/fold<k>_metrics.json    fold metrics (read by track.py)
models/oof/<exp_name>_oof_fold<k>.csv     out-of-fold predictions;  <exp_name>_gold_fold<k>.csv: gold-58 predictions
models/oof/<exp_name>.csv                 all folds' OOF merged = teacher file for pseudo-labels
models/oof/<exp_name>_cv.json             CV summary;  report_targets.csv: the targets OOF is scored against
logs/                                     Lightning CSV logs, per-fold stdout
```
