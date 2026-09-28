"""
Build the training cache: data/processed/cache_<res>/<uid>.npz for every training study.

    python scripts/preprocess.py                       # uses configs/default.yaml
    python scripts/preprocess.py exp=coatnet_384       # 384px cache for the CoAtNet config

The raw DICOMs are ~570 GB; the cache at 256px with 72 slices/study is ~20 GB.
Run it where the data lives (a Kaggle notebook with the competition attached, or a cloud box).
"""
import os
import sys

import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_config  # noqa: E402
from src.constants import ID_COL  # noqa: E402
from src.models.inference import build_cache  # noqa: E402


def main():
    cfg = get_config()
    train = pd.read_csv(cfg.data.train_file, dtype={ID_COL: str})
    series = pd.read_csv(cfg.data.train_series_file, dtype={ID_COL: str, "SeriesInstanceUID": str})
    ids = [u for u in train[ID_COL] if os.path.isdir(os.path.join(cfg.data.train_dicom_dir, u))]
    done = set()
    if os.path.isdir(cfg.paths.cache_dir):
        done = {f[:-4] for f in os.listdir(cfg.paths.cache_dir) if f.endswith(".npz")}
    todo = [u for u in ids if u not in done]
    print(f"{len(ids)} studies on disk, {len(todo)} to cache -> {cfg.paths.cache_dir}")
    pp = OmegaConf.to_container(cfg.preprocess, resolve=True)
    chunk = 200  # chunked so progress shows and a crash loses little work
    for i in tqdm(range(0, len(todo), chunk)):
        build_cache(todo[i : i + chunk], cfg.data.train_dicom_dir, series, pp, cfg.paths.cache_dir,
                    cfg.preprocess.num_workers)


if __name__ == "__main__":
    main()
