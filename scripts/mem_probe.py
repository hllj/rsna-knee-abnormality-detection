"""
Peak GPU memory and speed of a few training steps for a config at several batch sizes (random inputs, same
model / precision / gradient checkpointing / window count as the real run). Pick the largest batch that leaves
~1 GB headroom before starting a multi-hour run.

    python scripts/mem_probe.py exp=raptor_repro --batches 4 6 8
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_config  # noqa: E402
from src.models.model import build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", nargs="+", type=int, default=[4, 6, 8])
    ap.add_argument("--steps", type=int, default=6)
    a, rest = ap.parse_known_args()
    cfg = get_config(rest)
    dev = torch.device("cuda")
    model = build_model(cfg).to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    k = int(cfg.windows.k_train) * (6 if cfg.preprocess.get("layout") == "series" else 1)
    s = int(cfg.model.img_size)
    dtype = torch.bfloat16 if "bf16" in str(cfg.training.precision) else torch.float16
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    for b in a.batches:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            t0 = None
            for i in range(a.steps):
                if i == 2:
                    torch.cuda.synchronize()
                    t0 = time.time()
                x = torch.randn(b, k, 3, s, s, device=dev)
                slot = torch.randint(0, len(cfg.preprocess.slots), (b, k), device=dev)
                mask = torch.ones(b, k, dtype=torch.bool, device=dev)
                pos = torch.rand(b, k, device=dev)
                with torch.autocast("cuda", dtype=dtype):
                    loss = model(x, slot, mask, pos).float().mean()
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            dt = (time.time() - t0) / (a.steps - 2)
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"batch {b:2d} x {k} windows @{s}: peak {peak:.1f} / {total:.1f} GiB | {dt:.2f} s/step | "
                  f"{b / dt:.1f} studies/s", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"batch {b:2d}: OOM", flush=True)
            opt.zero_grad(set_to_none=True)


if __name__ == "__main__":
    main()
