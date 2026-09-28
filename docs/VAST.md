# Training on a rented GPU (vast.ai RTX 3090) and submitting to Kaggle

Kaggle gives ~30 GPU-h/week on T4s and at most 2 GPU sessions at once. A rented RTX 3090 (24 GB) trains the
same models faster, without those limits. Only the preprocessed cache (14.5 GiB in 14 tar shards) is downloaded — never the 570 GB
of DICOMs — and the submission itself still runs on Kaggle (code competition, internet off).

```
Kaggle                                          vast.ai box (RTX 3090)
──────                                          ──────────────────────
rsna-knee-cache (EXP-001, 4,407 npz)
   └─► rsna-knee-cache-pack (tar shards) ──────► scripts/vast/get_data.sh   data/processed/cache_256
hlly34/rsna-knee-code (dataset) ───────────────► the code                   scripts/vast/train.sh
                                                                             models/<exp>/fold*.pt
hlly34/rsna-knee-weights (dataset) ◄──────────── scripts/vast/submit.sh
   └─► rsna-knee-inference (GPU, offline) ─► submission.csv ─► leaderboard (EXP-NNN)
```

## 0. Rent the box

* Template: any **PyTorch (CUDA 12.x)** image; 1× RTX 3090; ≥ 8 CPU cores (data loading is CPU-bound —
  more cores ≈ faster epochs); ≥ 32 GB RAM; ≥ 60 GB disk (shards + extracted cache).

## 1. Setup (once per box)

```bash
# from your laptop
scp -P <port> ~/.kaggle/kaggle.json root@<host>:~/.kaggle/kaggle.json
# on the box
pip install -q kaggle
kaggle datasets download hlly34/rsna-knee-code -p rsna-knee --unzip    # the code (src/ configs/ scripts/ submission/)
cd rsna-knee
bash scripts/vast/setup.sh        # pip deps, GPU check
```

The code dataset is refreshed from the repo with `KAGGLE_BIN=kaggle bash scripts/upload_code.sh "msg"`.
(If you push the repo to GitHub, `git clone` instead works just as well.)

## 2. Data (~15 GB: 14 cache shards + csvs + labels)

```bash
bash scripts/vast/get_data.sh     # csvs + public label sets + cache shards (checksum-verified) + label AUC check
```

## 3. Train

```bash
bash scripts/vast/train.sh effnet_224 0                    # teacher, fold 0 first (compare with Kaggle EXP-002)
bash scripts/vast/train.sh effnet_224 "1 2 3 4"            # remaining folds once fold 0 looks right
bash scripts/vast/train.sh teacher_pseudo "0 1 2 3 4" "labels.pseudo.oof_files=[models/oof/effnet_b0_224.csv]"
```

* Any config key can be appended (`training.batch_size=32 training.num_epochs=20 model.backbone=...`).
* Each fold writes `models/<exp_name>/fold<k>.pt`, `fold<k>_metrics.json`, OOF + gold predictions; after the
  loop `merge_oof.py` prints the CV summary (pooled OOF AUC, gold AUC per fold and for the fold ensemble).
* Use `tmux` so a dropped SSH session doesn't kill training. Expect roughly 3–5 min/epoch for EfficientNet-B0 @224
  on a 3090 with 8+ cores (the Kaggle T4 run took ~8.9 min/epoch with two folds sharing 4 CPUs).
* The cache is 256 px. Configs that need a different `preprocess.res` must be overridden
  (`preprocess.res=256`) or re-cached.

## 4. Submit

```bash
NO_SUBMIT=1 bash scripts/vast/submit.sh EXP-006 "effnet fold0 vast" models/effnet_b0_224/fold0.pt   # dry: run notebook only
bash scripts/vast/submit.sh EXP-007 "teacher 5 folds" models/effnet_b0_224/fold*.pt                  # run + submit
python scripts/track.py lb       # after scoring (~30-60 min) pull the public LB into experiments/registry.csv
```

`submit.sh` uploads the checkpoints (+ metrics + offline pylibjpeg wheels) as a new version of
`hlly34/rsna-knee-weights`, regenerates `submission/inference` for that dataset with the experiment id, pushes it
(GPU, internet off), waits for the run to succeed, submits, and registers the experiment.

Bring `experiments/registry.csv` back to the repo (scp it, or rerun `scripts/track.py`) so the experiment table
stays in one place.
