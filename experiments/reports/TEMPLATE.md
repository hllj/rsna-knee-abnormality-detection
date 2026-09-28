# Iteration NN — <one-line hypothesis> (YYYY-MM-DD)

## 1. Hypothesis and change
* What changed vs the current best (config / code / labels), and why we expect it to help.
* Config: `bash scripts/vast/train.sh <config> ... <overrides>`

## 2. One-fold check (fold 0, one architecture)
| | OOF vs report labels | Gold-58 | Weakest gold labels | min/epoch |
|---|---|---|---|---|
| current best, fold 0 | | | | |
| this change, fold 0 | | | | |

Gate: continue to 5 folds only if fold-0 OOF ≥ best fold-0 OOF − 0.003 (noise floor) and gold did not drop
by more than ~0.02. Otherwise stop here, record as negative, go to §6.

## 3. Submissions
| Exp | Folds | OOF | Gold-58 | Public LB |
|---|---|---|---|---|
| EXP-… | 0 | | | |
| EXP-… | 0-4 | | | |

## 4. CV–LB
* ΔOOF / Δgold / ΔLB vs current best, and the updated correlations from `experiments/EXPERIMENTS.md → CV vs LB`.
* Does LB move with OOF? If not, which one do we trust for this change and why?

## 5. Backups
* Kaggle Model variations: `hlly34/<model>/PyTorch/<exp>-fold0..4`
* Local: `runs/EXP-…/`

## 6. Conclusion and next
* Keep / drop. New current best?
* Next iteration (ranked): 1. … 2. … 3. …
* Cost: GPU hours, $ spent, submissions used / left today.
