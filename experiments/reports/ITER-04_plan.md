# Iteration 04 (plan) — denser-slice cache, then CoAtNet @384

Current best: **EXP-020** ConvNeXt-S @288, 5 folds, public LB **0.926** (fold-0 bar: OOF 0.872 / gold 0.896).
Weakest gold findings: Synovitis 0.72, Lateral Meniscus 0.82, Lateral OA 0.82.

## 1. What a denser-slice cache is
Every study is cached as one uint8 stack of slices taken from 6 slots (plane × fluid-sensitive). Today
(`cache_256`, EXP-001) it holds **72 slices**: sag-FS 20, sag 10, cor-FS 16, cor 8, ax-FS 14, ax 4, sampled evenly
over the central 4–96 % of each series. A series has ~30 slices (median), so 20 samples land ~1.4 real slices apart
and a 2.5D window ("3 neighbouring cached slices") is not truly adjacent anatomy — a 1–3 mm meniscal tear seen on
2 consecutive slices can fall between samples.

The dense cache (`cache_256_dense`, EXP-023) keeps **88 slices**, adding them where menisci and ligaments are read:

| Slot | now | dense |
|---|---|---|
| sagittal fluid | 20 | **26** |
| sagittal T1 | 10 | **14** |
| coronal fluid | 16 | **20** |
| coronal T1 | 8 | **10** |
| axial fluid | 14 | 14 |
| axial T1 | 4 | 4 |
| **total** | **72** | **88** |

Size: 72 slices = 14.48 GiB → 88 slices ≈ 17.7 GiB, under Kaggle's 20 GB notebook-output cap (100+ slices would not
fit at 256 px). Config: `configs/exp/convnext_288_dense.yaml` (also used by the cache notebook).

**Evidence (Kaggle, docs/KAGGLE_RESOURCES.md):** Raptor CoAtNet with the same 140 mm crop: 44 → 64 → 80 slices/study =
single-model LB 0.926 → 0.928 → 0.932, while spreading the *same* 44 slices over a wider span dropped to 0.917 —
density, not span, is the lever. Strongest single lever in the public evidence; untested here so far.

## 2. Steps
| # | What | Where | Cost | Exp |
|---|---|---|---|---|
| 1 | Build dense cache (CPU, ~2–2.5 h) — **done** | Kaggle `rsna-knee-cache-dense` | 0 GPU, 0 $ | EXP-023 |
| 2 | Pack into ~1 GB tar shards — **done: 4,407 studies, 17.83 GiB, 18 shards** | Kaggle `rsna-knee-cache-dense-pack` | 0 | EXP-023 |
| 3 | ConvNeXt-S @288 dense, **fold 0** — gate vs EXP-020 fold 0 (OOF 0.872 / gold 0.896) | vast.ai 3090 | ~1.6 h ≈ $0.4 | EXP-024 |
| 4 | If it passes: folds 1–4, submit fold 0 + 5-fold | vast.ai 3090 | ~6.5 h ≈ $1.6 | EXP-025/026 |
| 5 | CoAtNet-RMLP-2 @384 fold 0 (on dense cache if 3 passes) | vast.ai 3090 | ~3.5–4 h ≈ $1 | EXP-027 |

Commands on vast (after `setup.sh`):
```bash
CACHE_PACK=rsna-knee-cache-dense-pack CACHE_NAME=cache_256_dense bash scripts/vast/get_data.sh
bash scripts/vast/train.sh convnext_288_dense 0                       # batch 8 x accumulate 1 on a 24 GB card
```

## 3. Batch size on the 24 GB RTX 3090
EXP-020 used batch 4 × accumulate 2 (copied from the 14.5 GB Kaggle T4) and peaked at 11.6 GB. On a 3090 use
**batch 8 × accumulate 1** (same effective batch; ConvNeXt has LayerNorm, so results stay comparable; ~21–22 GB).
Expect only ~5–15 % speed-up — GPU utilisation was already 98–100 %. Spare memory is better spent on
`windows.k_train` 16 → 24 (more slices per study per step) as a separate, gated experiment.

## 4. CoAtNet-RMLP-2 @384 settings for a 24 GB card
ITER-02's T4 run (EXP-015) was cut by the 9 h cap mid-schedule and its top-3 average mixed in a weak epoch
(validated only every 2nd epoch). Raptor measured this backbone at batch 8 × 12 windows + grad checkpointing =
23.7 GB and used head LR 1e-3 / backbone 3e-5 / wd 0.02 (our config already matches).

```bash
bash scripts/vast/train.sh coatnet_384 0 \
  training.batch_size=4 training.accumulate_grad_batches=2 model.grad_checkpointing=true \
  windows.k_train=12 windows.k_eval=24 training.num_epochs=12 training.val_every=1 training.top_k_average=3
```
~14–16 GB (try batch 6 if the first epoch shows < 16 GB), ~3.5–4 h per fold on a 3090. Benchmark its inference
(`scripts/bench_inference.py`) before a 5-fold submission. For the dense cache add
`preprocess.slots.sag_fs.n=26 …` or use a dense CoAtNet config.

## 5. Results so far
### EXP-024 — ConvNeXt-S @288 on the dense cache, fold 0 (2026-10-08, vast 3090 Washington)
| Fold 0 | OOF vs report labels | Gold-58 | Lat. Meniscus OOF / gold | Synovitis gold | Train time |
|---|---|---|---|---|---|
| EXP-020 (72 slices, bs 4×2) | 0.8719 | 0.8958 | 0.860 / 0.808 | 0.741 | 87 min |
| **EXP-024 (dense 88 slices, bs 8×1, channels_last)** | **0.8745** (+0.0026) | **0.9037** (+0.008) | **0.866 / 0.842** | 0.744 | **73 min** |

* **Gate: pass** (OOF ≥ bar − 0.003 and gold up). The OOF gain alone is at the noise floor, but gold, Lateral
  Meniscus (+0.034 gold) and ACL / Medial OA / Fracture all move the right way — consistent with the
  Kaggle evidence that slice density matters most for thin structures.
* Efficiency: batch 8 × accumulate 1 + channels_last filled 21.8 / 24.6 GB at 100 % GPU and cut the epoch time to
  ~5.25 min (from 6.4–6.9) despite 22 % more slices per study.
* Backup: Kaggle Model `hlly34/rsna-knee-convnext-s-288-dense` → `exp-024-fold0`; local `runs/EXP-024/`.
* Next: submit fold 0 (expect ≥ EXP-019's 0.923), train folds 1–4 (~4 × 73 min ≈ 5 h ≈ $1.5–2), submit the 5-fold
  ensemble; then CoAtNet @384 on the dense cache.
