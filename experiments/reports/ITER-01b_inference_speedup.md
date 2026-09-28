# Iteration 01b — inference speed-up (2026-09-28)

## 1. Change
Inference ideas 1–4 from ITER-01, same EXP-009 checkpoints:
1. **Selective decode** — read all slice headers (no pixels, threaded) to sort/orient, decode only the ~72 slices
   sampled into the stack instead of every slice (~165/study).
2. **Faster reads** — thread pool per study; raw `PixelData` buffer decode for uncompressed 8/16-bit files
   (bit-identical to `pixel_array`, unit-tested).
3. **CPU/GPU overlap** — a process pool builds stacks in memory while the main process runs the GPU on finished
   studies (`--mode stream`, now the default); no npz round-trip.
4. **One pass for all checkpoints** — windows built once per input contract and fed to every model.

The training cache is unchanged. The only numeric difference is the intensity window, now taken from the decoded
slices instead of all slices (`pct_stride=1` reproduces the training cache exactly — unit-tested).

## 2. Benchmark (EXP-011, Kaggle T4 + 4 CPUs, 150 fresh studies per variant, 5 checkpoints)
| Variant | s/study | ~1,300 studies | Gold-58 ens. | vs old: max / mean \|Δp\|, r |
|---|---|---|---|---|
| A old cache path | 1.23 | 26.6 min | 0.8930 | — |
| B stream, full decode (2–4) | 1.05 | 22.8 min | 0.8930 | 0.001 / 0.0001, 1.0 |
| **C stream + selective (1–4)** | **0.34** | **7.3 min** | **0.8938** | 0.026 / 0.0015, 0.99997 |
| D C + every 4th slice in window | 0.42 | 9.1 min | 0.8937 | 0.020 / 0.0016, 0.99997 |

Selective decode is the big lever (3.6×); streaming/one-pass adds ~15 % and removes the GPU from the critical
path. D's extra window slices buy nothing → C is the default (`--selective 1 --pct_stride 0`).

## 3. Submission (EXP-012)
Same 5 checkpoints as EXP-009 via the new path: **public LB 0.914** (= EXP-009), submit → scored **~11 min**
(19:21:31 → 19:32:54 UTC, including Kaggle queue/scoring) vs ≲ 30 min for EXP-009.

## 4. Conclusion and next
* Keep. Inference is no longer the bottleneck: ~7 min preprocessing + ~0.2 s/study for 5 EfficientNet-B0 models —
  room for ~20× more model compute inside a 3 h budget. Efficiency-prize candidate: a single fold now needs well
  under 10 min.
* Remaining inference ideas: 2nd T4 (shard studies), more eval windows (k_eval 48 → 64, accuracy), threads tuning.
* Cost: 2 bench runs (~25 min T4), 1 submission.
