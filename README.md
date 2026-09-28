# RSNA Knee Abnormality Detection 2026

Training / inference pipeline for [rsna-knee-abnormality-detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection),
built on [hllj/kaggle-boilerplate](https://github.com/hllj/kaggle-boilerplate) (OmegaConf configs, PyTorch Lightning, W&B, pre-commit, CI).

**Read [docs/COMPETITION_NOTES.md](docs/COMPETITION_NOTES.md) first**: rules, host rulings, what works, public assets, plan.

## Approach

```
reports ──LLM──► soft labels (several sources) ──per-finding silence priors + blend──► targets ◄── OOF teacher preds (pseudo)
                                                                                         │
DICOM ──sort by position, 150 mm crop, 1–99% window──► per-study stack: 6 slots (plane × fluid), 72 slices, uint8
                                                                                         │
bag of K 2.5D windows (3 adjacent slices = RGB) ─► timm backbone ─► + slot embedding ─► per-finding attention-MIL ─► 12 logits
```

## Layout

```
configs/default.yaml        all settings; configs/exp/*.yaml override (effnet_224, coatnet_384, convnext_288, dinov2_s_224, teacher_pseudo)
src/dataset/dicom.py        DICOM → cached stack (shared by train cache and test inference)
src/dataset/datasets.py     bag-of-windows Dataset
src/dataset/augmentations.py study-consistent affine + intensity (no flips)
src/dataset/folds.py        multilabel-stratified folds; gold 58 → fold -1 (eval only)
src/labeling/blend.py       label sources → silence handling → blend → optional OOF pseudo-label mix
src/labeling/llm_label.py   LLM report labeller (any OpenAI-compatible endpoint, incl. local vLLM)
src/models/model.py         KneeMILModel (timm backbone + slot embedding + attention-MIL head)
src/models/train_pl.py      Lightning training, top-k epoch weight averaging, OOF + gold predictions
src/models/inference.py     test DICOM → cache → ensemble (mean or rank) → submission.csv
scripts/                    download_data.sh, preprocess.py, eval_labels.py, merge_oof.py, push_to_kaggle.sh
submission/kernel/          offline Kaggle inference notebook + kernel-metadata.json
tests/                      unit tests + end-to-end smoke test on synthetic DICOMs
```

## Quickstart

```bash
conda env create -f environment.yml && conda activate rsna-knee && pip install -e . && pre-commit install
bash scripts/download_data.sh                 # csvs + public label sets into data/raw, data/labels
python scripts/eval_labels.py                 # score label sources / blend vs the 58 gold studies

# DICOMs (~570 GB) must be where you preprocess:  data/raw/train_series/<study>/<series>/*.dcm
python scripts/preprocess.py                  # -> data/processed/cache_256/<uid>.npz  (~20 GB)
python scripts/preprocess.py exp=coatnet_384  # separate 384px cache for the CoAtNet config

python -m src.models.train_pl exp=effnet_224 fold=0                       # quick baseline
for f in 0 1 2 3 4; do python -m src.models.train_pl exp=coatnet_384 fold=$f; done
python scripts/merge_oof.py --exp coatnet2_384                            # teacher OOF
python -m src.models.train_pl exp=teacher_pseudo fold=-1                  # student on blended pseudo-labels
```

Any key can be overridden on the command line (`training.lr=2e-4 logging.wandb.enabled=true`).
Outputs: `models/<exp>/fold<k>.pt` (weight-averaged export), `models/oof/<exp>_{oof,gold}_fold<k>.csv`, `logs/`.

### Own LLM labels

```bash
export OPENAI_API_KEY=...        # or OPENAI_BASE_URL=http://localhost:8000/v1 for a local vLLM server
python -m src.labeling.llm_label --model <model-id> --out data/labels/my_llm_labels.csv
python scripts/eval_labels.py data/labels/my_llm_labels.csv
# then add my_llm_labels.csv to labels.sources in configs/default.yaml
```

### Where to preprocess without 570 GB locally

* A Kaggle notebook with the competition attached (`DATA_DIR=/kaggle/working/data`, symlink
  `/kaggle/input/rsna-knee-abnormality-detection` to `data/raw`). Output is capped at ~20 GB, so use
  `preprocess.res=224` or fewer slices, then save the output as a dataset.
* A cloud VM: `kaggle competitions download -c rsna-knee-abnormality-detection` then `preprocess.py`.

## Submit (code competition)

```bash
bash scripts/push_to_kaggle.sh models/effnet_b0_224/fold*.pt
```

This uploads `src/` + `configs/` as the private dataset `<user>/rsna-knee-code` and the weights as
`<user>/rsna-knee-weights`, then pushes `submission/kernel` (GPU, internet off). Open the kernel on
Kaggle, check the run, and press **Submit**. If DICOM decoding fails on JPEG-2000 series, attach a
dataset with a `wheels/` folder containing `pylibjpeg*` wheels (the notebook installs from it offline).

## Tests

```bash
pytest -m "not slow"   # unit tests
pytest -m slow         # end-to-end: synthetic DICOMs → preprocess → train 2 epochs → inference → submission.csv
```
