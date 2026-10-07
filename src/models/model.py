"""
2.5D multiple-instance model.

  windows (B, K, 3, H, W) --timm backbone--> features (B, K, F)
    + learned slot embedding (which plane / sequence the window came from)
    --> per-finding attention pooling over the K windows --> 12 logits

Per-finding attention lets an ACL tear draw on two sagittal slices while OA spreads across
many coronal ones; a single shared pooling makes these compete (the Raptor ablation found
this head mattered more than the backbone).
"""
from typing import Optional

import timm
import torch
import torch.nn as nn

from src.constants import NUM_LABELS


class AttentionMILHead(nn.Module):
    """Gated attention pooling with separate attention weights and classifier per finding."""

    def __init__(self, dim: int, n_out: int = NUM_LABELS, hidden: int = 256, drop: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.att_v = nn.Linear(dim, hidden)
        self.att_u = nn.Linear(dim, hidden)
        self.att_w = nn.Linear(hidden, n_out)
        self.drop = nn.Dropout(drop)
        self.cls_w = nn.Parameter(torch.empty(n_out, dim))
        self.cls_b = nn.Parameter(torch.zeros(n_out))
        nn.init.trunc_normal_(self.cls_w, std=0.02)

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm(h)
        a = self.att_w(self.drop(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))))  # B,K,N
        if mask is not None:
            a = a.masked_fill(~mask[..., None], float("-inf"))
        a = torch.softmax(a.float(), dim=1).to(h.dtype)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled * self.cls_w).sum(-1) + self.cls_b


class PoolHead(nn.Module):
    """Mean / max pooling baseline head."""

    def __init__(self, dim: int, n_out: int = NUM_LABELS, mode: str = "mean", drop: float = 0.2):
        super().__init__()
        self.mode = mode
        self.fc = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(drop), nn.Linear(dim, n_out))

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        m = mask[..., None].to(h.dtype)
        if self.mode == "max":
            pooled = h.masked_fill(~mask[..., None], float("-inf")).amax(1)
        else:
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.fc(pooled)


class KneeMILModel(nn.Module):
    def __init__(
        self,
        backbone: str = "tf_efficientnet_b0.ns_jft_in1k",
        pretrained: bool = True,
        num_slots: int = 6,
        head: str = "attn_mil",
        slot_embedding: bool = True,
        drop_rate: float = 0.2,
        drop_path_rate: float = 0.1,
        grad_checkpointing: bool = False,
        img_size: Optional[int] = None,
        channels_last: bool = False,
    ):
        super().__init__()
        kw = dict(pretrained=pretrained, num_classes=0, drop_path_rate=drop_path_rate)
        if img_size is not None and any(t in backbone for t in ("vit", "dinov2", "eva", "deit")):
            kw["img_size"] = img_size  # ViTs need their pos-embed resized at build time
        self.backbone = timm.create_model(backbone, **kw)
        if grad_checkpointing:
            self.backbone.set_grad_checkpointing(True)
        # NHWC layout: faster convolutions with AMP on tensor-core GPUs (Ampere+); numerically equivalent
        self.channels_last = channels_last
        if channels_last:
            self.backbone.to(memory_format=torch.channels_last)
        dim = self.backbone.num_features
        self.slot_emb = nn.Embedding(num_slots, dim) if slot_embedding else None
        if self.slot_emb is not None:
            nn.init.zeros_(self.slot_emb.weight)
        self.head = (
            AttentionMILHead(dim, drop=drop_rate) if head == "attn_mil" else PoolHead(dim, mode=head, drop=drop_rate)
        )

    @property
    def data_config(self) -> dict:
        """Normalisation constants this backbone was pretrained with."""
        return timm.data.resolve_data_config({}, model=self.backbone)

    def forward(self, x: torch.Tensor, slot: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, k = x.shape[:2]
        x = x.flatten(0, 1)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        h = self.backbone(x).view(b, k, -1)
        if self.slot_emb is not None and slot is not None:
            h = h + self.slot_emb(slot)
        return self.head(h, mask)


def build_model(cfg, pretrained: Optional[bool] = None) -> KneeMILModel:
    m = cfg.model
    return KneeMILModel(
        backbone=m.backbone,
        pretrained=m.pretrained if pretrained is None else pretrained,
        num_slots=len(cfg.preprocess.slots),
        head=m.head,
        slot_embedding=m.slot_embedding,
        drop_rate=m.drop_rate,
        drop_path_rate=m.drop_path_rate,
        grad_checkpointing=m.grad_checkpointing,
        img_size=m.img_size,
        channels_last=bool(m.get("channels_last", False)),
    )
