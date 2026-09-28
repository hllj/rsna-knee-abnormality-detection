# Iteration workflow (every future change follows this loop)

One iteration = one hypothesis (a new architecture, resolution, label set, loss, …). Steps and commands:

```
1. fold 0, one architecture ──gate──► 2. submit fold 0 + 5-fold ──► 3. results + CV–LB ──► 4. backup models ──► 5. report + next
                    └── fails gate: stop, record as negative, straight to 5
```

## 1. Train one fold first (one architecture)
```bash
bash scripts/vast/train.sh <config> 0 [overrides]           # e.g. convnext_288 0 preprocess.res=256
bash scripts/vast/status.sh                                 # per-epoch report / gold AUC while it runs
```
Compare fold 0 against the current best's **fold 0** (same held-out studies; `runs/EXP-009/.../fold0_metrics.json`:
OOF 0.865, gold 0.876):

* **OOF vs report labels** — compare only the merged/`_oof_fold0.csv` score against *report* targets
  (`merge_oof.py` output). A pseudo-label run's own `val_auc_report` is scored on blended targets — not comparable.
* **Gate**: go on only if fold-0 OOF ≥ best − 0.003 and gold-58 has not dropped by more than ~0.02.
  Fold-to-fold spread is ±0.008 OOF / ±0.011 gold, so a single fold only catches clear wins or losses.
* Failed gate → record the experiment as NEGATIVE in the registry, skip to step 5.

## 2. Submit 1 fold, then 5 folds
```bash
bash scripts/vast/submit.sh EXP-NNN "<config> fold0" models/<exp_name>/fold0.pt
bash scripts/vast/train.sh <config> "1 2 3 4" [same overrides]
bash scripts/vast/submit.sh EXP-NNN+1 "<config> 5-fold" models/<exp_name>/fold*.pt
# optional: blend with the current best
bash scripts/vast/submit.sh EXP-NNN+2 "<config> + EXP-009" models/<exp_name>/fold*.pt runs/EXP-009/models/effnet_b0_224/fold*.pt
```
Budget: 5 submissions/day — keep 1 spare. `NO_SUBMIT=1` runs the Kaggle notebook without submitting.

## 3. Update results and CV–LB
```bash
python scripts/track.py lb                                  # public LB for every EXP-… submission
python scripts/track.py add EXP-NNN --outputs runs/EXP-NNN  # after copying the box's models/ + logs to runs/EXP-NNN
```
`experiments/EXPERIMENTS.md → CV vs LB` then shows ΔOOF / Δgold / ΔLB vs the previous submission and the
running correlations. Update the "Current best" table at the top of `README.md` if the LB improved.

## 4. Upload models for backup
```bash
scp -P <port> -r root@<host>:~/rsna-knee/models runs/EXP-NNN/      # local copy first
python scripts/upload_models.py --model <kaggle-model-slug> --exp EXP-NNN runs/EXP-NNN/models/<exp_name>/fold*.pt
```
One Kaggle Model per architecture (e.g. `rsna-knee-effnet-b0-224`, `rsna-knee-convnext-s-288`), one PyTorch
variation per fold (`exp-nnn-fold<k>`). Then destroy the vast.ai instance (`vastai destroy instance <id>`).

## 5. Report and next steps
Copy `experiments/reports/TEMPLATE.md` → `experiments/reports/ITER-NN_<date>.md` and fill it: hypothesis, one-fold
check, submissions, CV–LB reading, backups, keep/drop, ranked next ideas, cost. Commit
`experiments/` + `README.md` together.

## Inference time budget (check at step 2)

Kaggle limit: 9 h per submission run (efficiency prize also rewards less). Measured on the hidden-test path
(Kaggle T4, 4 CPUs, ~1,300 test studies):

| Part | Cost | Source |
|---|---|---|
| DICOM → cache (once, shared by all models) | 1.05–1.5 s/study → **23–32 min** | dry run: 658 studies in 693–971 s |
| EfficientNet-B0 @224, 1 fold (48 windows/study) | ~1.5–2 min (data-loader bound) | 3090 val 928 studies ≈ 70 s; EXP-009 log +0.9 s/fold on 3 studies |
| **1 fold total** | **~25–35 min** | EXP-007 scored ≲ 32 min after submit |
| **5 folds total** | **~32–45 min** | EXP-009 scored ≲ 30 min after submit |

Rule of thumb: `total ≈ 30 min + Σ_models (1,300 studies / throughput)`. For a new architecture, time the fold-0
validation pass on the training box and scale by 1,300 / 928 × (T4 / 3090 slowdown ≈ 2–3× for GPU-bound models).
Keep the full submission under ~3 h to leave margin for slower test DICOMs. `NO_SUBMIT=1 submit.sh` runs the
notebook on the 3 public test studies only — it checks correctness, not runtime.

## Conventions
* Experiment ids `EXP-NNN` are global and never reused; they prefix every submission message.
* One fold split for everything: `configs/folds_5.csv` (58 gold studies = fold -1, never trained on).
* Current best and its fold-0 numbers are the bar for the next gate.
