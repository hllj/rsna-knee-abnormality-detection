# ITER-14 plan (2026-10-18): S6 recipe (discussion 749359) → DINOv2 + ConvNeXt retrain

Source: nartaa "0.950 with a second architecture: Heliosli's S6 ConvNeXt retrained on our labels"
(`kaggle forums topics show 749359`). Code: `nartaa/rsna-knee-s6fall-weights/training_code/` (Apache-2.0); weights
`E_prior_s42_fall.pt` (CC BY-NC 4.0, public 0.950 alone). Inference: `nartaa/rsna-knee-s6fall-solo`.

## Recipe (as published)
- Heliosli S6: `convnext_small.fb_in22k_ft_in1k`, per-finding spatial attention + window-attention MIL over the study.
- Views `v4`: full FOV + 100 mm knee-centred crops placed by the SKM-TEA anatomy V-Net (heliosli/rsna-orthofoundation-fold0-epoch7-20261004).
- Cache 320 px → crop 288, 3 slices/window, 3.5 mm spacing; DICOMs read on the fly (`fly_data.py`), anchors precomputed.
- Labels: JEV-blend soft labels (`labels_v1_blendjev.parquet`, CC0 sources), prior fill, weight-normalised BCE (arm E_prior).
- 64 windows/study, 16 epochs, AdamW lr 1e-4 / head 1e-3, wd 0.05, batch 2, cosine 5 % warm-up, bf16, grad ckpt, seed 42,
  all 4,263 studies (gold-58 excluded). 15.0 h on an RTX 5060 laptop (34,096 steps).

## What has to change
1. **Paths** (hard-coded `/home/danial/...`): `s6common.py:12` (anatomy.pt), `s6common.py:14` (raw DICOM root),
   `make_data.py:16`, `train_s6f0_all.py:590`, `run_all_alldata.sh:37-41,67` (python, INIT safetensors, folds_v0.json
   from `nartaa/rsna-knee-hpo-assets`). Make them env vars in a wrapper `scripts/vast/train_s6.sh`.
2. **Data on the box**: the raw competition DICOMs (our caches do not fit this loader), heliosli's `cloud_code` +
   `anatomy.pt`, the MONAI 0.6.0 / `compact_*` files packed in `s6_extras.bin` (unpack as the solo notebook does), timm
   init weights pre-downloaded (HF_HUB_OFFLINE=1). Then `precompute_anchors.py` for all 4,321 studies (one-off) and check
   `pool_ok.json` membership (reuse theirs; 86 studies they deleted locally could be added back later).
3. **DINOv2 (argv only — the ViT path is in heliosli's original `compact_train.py`, so inference loads it too)**:
   `--backbone vit_base_patch14_dinov2.lvd142m --crop 280` (14-px patches → square 20×20 grid; `spatial_pool` asserts a
   square grid), `--llrd 0.75 --lr 5e-5 --head-lr 1e-3` (ViT fine-tune; our guess, not tuned), rest unchanged. First a
   ~200-step memory/speed probe at batch 2 × 64 windows on 24 GB; fall back to `--train-windows 48` or ViT-S
   (`vit_small_patch14_dinov2.lvd142m`) if it does not fit or is too slow.
4. **ConvNeXt retrain**: same argv, **different seed (43)**. A seed-42 repro would be a near-copy of nartaa's public
   weights, which EXP-127 uses directly; a second seed is a usable extra member.
5. **Gate**: DINOv2 fold 0 on folds_v0 vs nartaa's ConvNeXt fold 0 (OOF 0.907, gold-58 0.931). Run the all-data model in
   parallel (time) and drop it if the gate fails.
6. **Inference in our notebook**: a stage `submission/external/s6_stage.py` from `rsna-knee-s6fall-solo` with per-checkpoint
   `B_GEOMETRY` (backbone, crop) and a weights dataset of ours; a fusion slot in `build_kernels.py`. EXP-127 (EXP-124 +
   nartaa's S6fall weights) builds this plumbing first; the new checkpoints are then extra members of the same stage.
7. **Kaggle runtime**: EXP-124 already uses ~5 h of 9 h. Measure the S6 stage in EXP-127 first; DINOv2-B costs ~2× the
   ConvNeXt-S per window, so ViT-S may be needed to fit.
8. **Rule status**: SKM-TEA anatomy net = external public data (no OAI / fastMRI) → pick-1 material; not pick 2 under the
   strict "competition data only" rule unless the owner decides otherwise.

## Training plan (owner 10-18: DINOv2 ViT-B, all data, from a cache — not the raw DICOMs on vast)
Raw DICOMs are read once, on Kaggle (mounted there), to build the cache; the trainer then runs in `--px` mode
(`train_s6f0_all.py:596`, heliosli's original path). `fly_data.render(uid)` returns exactly the px-cache rows
(px uint8 [N,S,S] + meta [N,12]), so the cache = render every study once (+ the anchor V-Net in the same pass).

| Step | Where | What | Time | Cost |
|---|---|---|---|---|
| 0 | local | vendor `training_code/` → `external/s6_train/`; env-var paths (s6common, make_data); `build_px_shard.py` (anchor + render → JPEG-q95 shard + meta + index), `unpack_px.py` (shard → `px.u8.npy` / `meta.f32.npy` / `index.json`) | ~2 h | $0 |
| 1 | Kaggle GPU | **pilot, 50 studies**: s/study, rows/study, JPEG GB, JPEG error vs lossless rows, parity with `render()`. Gate: full cache ≤ ~40 GB and build ≤ ~6 h | ~45 min | $0 (~1 h quota) |
| 2 | Kaggle GPU ×2 | full cache, 4,321 studies (4,263 train + 58 gold) in 2 notebooks run in parallel, ≤ 20 GB output each → dataset `hlly34/rsna-knee-s6-px320` | ~3-5 h | $0 (quota measured in step 1) |
| 3 | vast 1× 4090 | download shards (~20-35 GB), unpack, `--check-targets`, **200-step probe → report s/step**, then the full run: `--backbone vit_base_patch14_dinov2.lvd142m --crop 280 --size 320 --lr 5e-5 --llrd 0.75 --head-lr 1e-3`, rest = nartaa's argv (16 ep, 64 windows, batch 2, bf16, grad ckpt, all data, their targets/pool, seed 42), `--resume` | ~1 h + ~6.3 h | ~$5 (cap $10) |
| 4 | vast | gold-58: DINOv2 vs nartaa's public ConvNeXt on the same cache rows, plus their rank blend | ~15 min | incl. |
| 5 | vast → Kaggle | upload weights (Kaggle Model `rsna-knee-s6-dinov2-b`), destroy the box | ~20 min | incl. |
| 6 | Kaggle | EXP-128 = EXP-127 (S6 stage, nartaa's ConvNeXt) + our DINOv2 as a second member; test run, check the 9 h budget (ViT-B at 280 ≈ 2.6× ConvNeXt-S per window), submit with owner OK | ~1.5 h | $0 |

Timeline: steps 0-2 on 10-18, training overnight, steps 4-6 on 10-19, one day of slack before the 10-20 freeze.
Insurance (optional, +$2.5-3): ConvNeXt-S seed 43 in parallel on a 2× 4090 box, in case the untuned ViT LR fails.
Approvals: steps 0-2 use only Kaggle quota; step 3 rents a GPU.

### Step 1 result: pilot (hlly34/rsna-knee-s6-cache-pilot v2, 10-18 12:24 local)
50/50 studies, 0 errors, T4 + 4 CPUs, 0.148 studies/s (CPU-bound). Per study 600 rows. Projected for 4,321 studies:
raw px 247 GB, lossless zlib 171 GB, **JPEG q95 71.7 GB** (estimate was 15-25 GB: MRI rows compress ~3.4×, not ~10×).
JPEG q95 error: MAE 0.68 grey levels, p99.9 4.4, max 12 → kept. Kaggle output cap ~20 GB → **5 shards** (~14 GB, ~1.6 h each,
2 GPU sessions at a time → ~5 h wall, ~8 h quota). vast: ~72 GB download (~$0.2 at $0.0026/GB), ~320 GB disk.

### Step 2 result: full cache (10-18 12:29-15:14 local)
5 shards × ~864 studies, ~50 min each on Kaggle T4 (0.3 studies/s), **4,321/4,321 studies, 0 errors**, 70.9 GB JPEG q95,
4.9 h Kaggle GPU quota.

### Step 3-5 result: EXP-128 = ConvNeXt-S, nartaa's exact argv, seed 43, all data (owner switched from DINOv2 on 10-18)
- Box: vast 56544406, 1× RTX 4090 (AU), 18:30-23:23 local, destroyed after the upload. First offer (52207942) never got its
  GPU (destroyed). `kaggle kernels output` pulled shards at 2.8 MB/s from AU → `scripts/vast/fetch_kernel_output.py` (48
  threads, ~73 MB/s). Cost ≈ $3.4 (credit 4.59 → ~1.2).
- Training: 16 epochs, 34,096 steps, 3.6 h (330-372 windows/s, GPU 100 %), 0 failed studies. Loss per epoch tracks
  nartaa's run (ep1 0.549 vs 0.552, ep8 0.4334 vs 0.4334, ep16 **0.3923 vs 0.3911**).
- Weights: private Kaggle dataset `hlly34/rsna-knee-s6-cnx-s43` (`E_prior_s43_fall.pt`, sha256 88b81233…, + DONE, run_config,
  gold58).
- **Gold-58 macro AUC (our JPEG cache): ours s43 0.929, nartaa public s42 0.932, rank mean of both 0.934.** Pipeline check:
  nartaa's s42 on our cache 0.932 vs their own gold 0.931 (fold-0 model).

### Inference
- `make_s6_stage.py` → `s6_stage_<name>.py`: heliosli's compact stage with an inline nartaa-geometry manifest (320/288, eval
  stride 1, v4, no state head / side model), bundle assembled from the checkpoints' own datasets (sha-checked, nothing re-hosted);
  compact_infer averages every `E_prior_s*_f*.pt`.
- EXP-126 (EXP-124 + heliosli compact-v1 S6 0.3) **timed out (> 9 h)**: compact-v1 = 4 folds @352 + side model + state head,
  as heavy as R6. Owner chose option 1: S6fall instead (1 model @288, ~1/6 of the model compute; decode/anchors ~1.2 h).
- EXP-127 = EXP-124 + S6fall s42 at 0.3 in the heliosli slot (`s6_stage_s42.py`); EXP-128 = same with s42 + s43
  (`s6_stage_s42s43.py`). Gold-58 cannot separate the share (0.939 for every m in 0-0.7). No time guard (owner: option 1 only).

### Submissions
| EXP | Notebook | What | Status |
|---|---|---|---|
| EXP-126 | v82 | EXP-124 + heliosli compact-v1 S6 0.3 | **timeout (> 9 h)** |
| EXP-127 | v84 | EXP-124 + nartaa S6fall s42 0.3 (v83 failed: JSON `null` in the generated stage, fixed bed2ae1) | **LB 0.958** (within 9 h) |
| EXP-128 | v85 | EXP-124 + S6 bundle s42 + our s43 0.3 (S6 output differs from EXP-127's by up to 0.11 → both seeds averaged) | **LB 0.958** (within 9 h) |

Reading: S6fall (1 model @288) fits the 9 h budget where compact-v1 did not; adding it, with or without our seed, keeps the
public score at 0.958 (= EXP-119/123/124/125). Public SD ~0.003, so 3-decimal ties cannot rank these recipes.

## Cost / time (superseded by the training plan above)
3 runs (DINOv2 fold 0, DINOv2 all-data, ConvNeXt seed 43 all-data) on 3× RTX 3090: ~8-10 h each in parallel + ~2 h data
and anchors → ~$12-15. **Not rented — awaiting owner OK.**
