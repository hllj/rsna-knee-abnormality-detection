"""
PyTorch Lightning training script.

    python -m src.models.train_pl exp=effnet_224 fold=0
    python -m src.models.train_pl exp=coatnet_384 fold=-1          # full fit (no report val fold)
    for f in 0 1 2 3 4; do python -m src.models.train_pl exp=coatnet_384 fold=$f; done

Outputs (models/<exp_name>/):
    fold{f}.pt           top-k weight-averaged export {state_dict, cfg} used by inference
    ../oof/<exp>_fold{f}.csv   out-of-fold predictions (for pseudo-labels / blending)
"""
import copy
import json
import math
import os
import time
from typing import Dict, List

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.config import get_config
from src.constants import ID_COL, LABELS
from src.dataset.datasets import KneeStudyDataset
from src.dataset.folds import gold_ids, make_folds
from src.evaluation.metrics import macro_auc
from src.labeling.blend import build_targets
from src.models.model import build_model
from src.utils import set_seed, setup_logging


class KneeModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = build_model(cfg)
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        pw = cfg.training.pos_weight
        self.register_buffer("pos_weight", torch.tensor(list(pw), dtype=torch.float32) if pw else None)
        self._val: Dict[int, List] = {}
        self.val_names: List[str] = []  # one tag per val dataloader: "report" / "gold"

    def forward(self, batch):
        return self.model(batch["x"], batch.get("slot"), batch.get("mask"))

    def loss(self, logits, y):
        if self.cfg.training.loss == "focal":
            p = torch.sigmoid(logits)
            ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            pt = p * y + (1 - p) * (1 - y)
            return (ce * (1 - pt) ** 2).mean()
        return F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pos_weight)

    def training_step(self, batch, batch_idx):
        loss = self.loss(self(batch).float(), batch["y"])
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch["y"]))
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(batch).float()
        self._val.setdefault(dataloader_idx, []).append(
            (batch["uid"], torch.sigmoid(logits).cpu().numpy(), batch["y"].cpu().numpy())
        )

    def on_validation_epoch_end(self):
        for idx, rows in self._val.items():
            p = np.concatenate([r[1] for r in rows])
            y = np.concatenate([r[2] for r in rows])
            tag = self.val_names[idx]
            m, per = macro_auc(y, p)
            self.log(f"val_auc_{tag}", m, prog_bar=True, add_dataloader_idx=False)
            for k, v in per.items():
                self.log(f"val_{tag}/{k}", v, add_dataloader_idx=False)
        self._val = {}

    def configure_optimizers(self):
        t = self.cfg.training
        bb = [p for n, p in self.model.named_parameters() if n.startswith("backbone.")]
        rest = [p for n, p in self.model.named_parameters() if not n.startswith("backbone.")]
        opt = torch.optim.AdamW(
            [{"params": bb, "lr": t.lr * t.backbone_lr_mult}, {"params": rest, "lr": t.lr}],
            weight_decay=t.weight_decay,
        )
        total = self.trainer.estimated_stepping_batches
        warm = int(total * t.warmup_epochs / max(1, t.num_epochs))

        def lr_lambda(step):
            if step < warm:
                return (step + 1) / max(1, warm)
            prog = (step - warm) / max(1, total - warm)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


class TopKAverage(Callback):
    """Keep the best-k epoch weights (by `monitor`) on CPU and load their average at the end."""

    def __init__(self, monitor: str, k: int = 3):
        self.monitor, self.k = monitor, k
        self.kept: List = []  # (score, state_dict)

    def on_validation_end(self, trainer, pl_module):
        # not on_validation_epoch_end: callbacks run before the module logs that epoch's metric
        if trainer.sanity_checking or self.monitor not in trainer.callback_metrics:
            return
        score = float(trainer.callback_metrics[self.monitor])
        sd = {k: v.detach().cpu().clone() for k, v in pl_module.model.state_dict().items()}
        self.kept.append((score, sd))
        self.kept = sorted(self.kept, key=lambda t: -t[0])[: self.k]

    def on_fit_end(self, trainer, pl_module):
        if not self.kept:
            return
        avg = copy.deepcopy(self.kept[0][1])
        for key, v in avg.items():
            if v.is_floating_point():
                avg[key] = sum(sd[key] for _, sd in self.kept) / len(self.kept)
        pl_module.model.load_state_dict(avg)
        print(f"[topk] averaged epochs with {self.monitor} = {[round(s, 4) for s, _ in self.kept]}")


def make_loader(ds, cfg, shuffle):
    return DataLoader(
        ds, batch_size=cfg.training.batch_size if shuffle else max(1, cfg.training.batch_size // 2),
        shuffle=shuffle, num_workers=cfg.resources.num_workers, pin_memory=True,
        drop_last=shuffle, persistent_workers=cfg.resources.num_workers > 0,
    )


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device) -> pd.DataFrame:
    model.eval().to(device)
    uids, preds = [], []
    for b in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(b["x"].to(device), b["slot"].to(device), b["mask"].to(device))
        preds.append(torch.sigmoid(logits.float()).cpu().numpy())
        uids += list(b["uid"])
    return pd.DataFrame(np.concatenate(preds), columns=LABELS).assign(**{ID_COL: uids})[[ID_COL] + LABELS]


def main():
    cfg = get_config()
    set_seed(cfg.seed)
    log = setup_logging(cfg.paths.log_dir, f"{cfg.exp_name}_fold{cfg.fold}")
    out_dir = os.path.join(cfg.paths.model_dir, cfg.exp_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cfg.paths.oof_dir, exist_ok=True)

    t_start = time.time()
    train_df = pd.read_csv(cfg.data.train_file)
    targets = build_targets(cfg)
    # report-only targets (no pseudo-labels): the common ruler for OOF AUC across rounds
    report_targets = build_targets(OmegaConf.merge(cfg, {"labels": {"pseudo": {"enabled": False}}}))
    report_targets.rename_axis(ID_COL).reset_index().to_csv(
        os.path.join(cfg.paths.oof_dir, "report_targets.csv"), index=False)
    if not os.path.exists(cfg.data.folds_file):
        # folds must not depend on pseudo-labels, or OOF predictions leak between rounds
        os.makedirs(os.path.dirname(cfg.data.folds_file) or ".", exist_ok=True)
        make_folds(train_df, report_targets, cfg.data.num_folds, cfg.seed).to_csv(cfg.data.folds_file, index=False)
    folds = pd.read_csv(cfg.data.folds_file, dtype={ID_COL: str})
    cached = {f[:-4] for f in os.listdir(cfg.paths.cache_dir) if f.endswith(".npz")}
    folds = folds[folds[ID_COL].isin(cached)]

    gold = train_df[train_df[ID_COL].isin(gold_ids(train_df))].set_index(ID_COL)[LABELS]
    y = {u: targets.loc[u, LABELS].values.astype(np.float32) for u in targets.index}
    y.update({u: gold.loc[u].values.astype(np.float32) for u in gold.index})

    tr_ids = folds.loc[(folds.fold != cfg.fold) & (folds.fold >= 0), ID_COL].tolist()
    va_ids = folds.loc[folds.fold == cfg.fold, ID_COL].tolist() if cfg.fold >= 0 else []
    gd_ids = folds.loc[folds.fold == -1, ID_COL].tolist()
    tr_ids = [u for u in tr_ids if u in y]
    log.info(f"train {len(tr_ids)} | val {len(va_ids)} | gold {len(gd_ids)}")

    module = KneeModule(cfg)
    dc = module.model.data_config
    common = dict(cache_dir=cfg.paths.cache_dir, targets=y, img_size=cfg.model.img_size, mean=dc["mean"], std=dc["std"])
    train_ds = KneeStudyDataset(
        tr_ids, k=cfg.windows.k_train, train=True,
        aug_cfg=OmegaConf.to_container(cfg.augmentation) if cfg.augmentation.enabled else None,
        window_dropout=cfg.augmentation.window_dropout, **common,
    )
    val_loaders, module.val_names = [], []
    if va_ids:
        val_loaders.append(make_loader(KneeStudyDataset(va_ids, k=cfg.windows.k_eval, **common), cfg, False))
        module.val_names.append("report")
    if cfg.evaluation.gold_eval and gd_ids:
        val_loaders.append(make_loader(KneeStudyDataset(gd_ids, k=cfg.windows.k_eval, **common), cfg, False))
        module.val_names.append("gold")
    monitor = cfg.training.monitor if va_ids else "val_auc_gold"

    loggers = [CSVLogger(cfg.paths.log_dir, name=cfg.exp_name, version=f"fold{cfg.fold}")]
    if cfg.logging.wandb.enabled:
        loggers.append(WandbLogger(
            project=cfg.logging.wandb.project, entity=cfg.logging.wandb.entity,
            name=f"{cfg.exp_name}_fold{cfg.fold}", tags=list(cfg.logging.wandb.tags),
            config=OmegaConf.to_container(cfg, resolve=True),
        ))
    topk = TopKAverage(monitor, cfg.training.top_k_average)
    trainer = L.Trainer(
        max_epochs=cfg.training.num_epochs, precision=cfg.training.precision,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        gradient_clip_val=cfg.training.gradient_clip_val, devices=cfg.resources.devices,
        logger=loggers, callbacks=[topk, LearningRateMonitor("step")],
        enable_checkpointing=False, log_every_n_steps=20,
    )
    trainer.fit(module, make_loader(train_ds, cfg, True), val_loaders or None)

    export = os.path.join(out_dir, f"fold{cfg.fold}.pt")
    torch.save({"state_dict": module.model.state_dict(), "cfg": OmegaConf.to_container(cfg, resolve=True)}, export)
    log.info(f"saved {export}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = {}
    for name, ids in (("oof", va_ids), ("gold", gd_ids)):
        if not ids:
            continue
        pred = predict(module.model, make_loader(KneeStudyDataset(ids, k=cfg.windows.k_eval, **common), cfg, False), device)
        pred.to_csv(os.path.join(cfg.paths.oof_dir, f"{cfg.exp_name}_{name}_fold{cfg.fold}.csv"), index=False)
        truth = np.stack([y[u] for u in pred[ID_COL]])
        m, per = macro_auc(truth, pred[LABELS].values)
        log.info(f"[{name}] macro AUC {m:.4f} | " + " ".join(f"{k}={v:.3f}" for k, v in per.items()))
        metrics[f"{name}_auc"], metrics[f"{name}_per_label"] = round(m, 5), {k: round(v, 4) for k, v in per.items()}

    # one JSON per run: read by scripts/track.py into experiments/registry.csv
    metrics.update(exp_name=cfg.exp_name, fold=cfg.fold, n_train=len(tr_ids), n_val=len(va_ids),
                   minutes=round((time.time() - t_start) / 60, 1), backbone=cfg.model.backbone,
                   img_size=cfg.model.img_size, epochs=cfg.training.num_epochs,
                   labels=list(cfg.labels.sources), pseudo=bool(cfg.labels.pseudo.enabled),
                   topk_scores=[round(s_, 4) for s_, _ in topk.kept])
    with open(os.path.join(out_dir, f"fold{cfg.fold}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)


if __name__ == "__main__":
    main()
