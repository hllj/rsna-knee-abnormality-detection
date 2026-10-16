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

# RadImageNet ResNet-50 (Lab-Rasool/RadImageNet ResNet50.pt) stores nn.Sequential(*resnet50.children()[:-2]):
# backbone.0 conv1, .1 bn1, (.2 relu, .3 maxpool), .4-.7 layer1-4 -> timm resnet50 names
_RADIMAGENET_PREFIX = {"backbone.0.": "conv1.", "backbone.1.": "bn1.", "backbone.4.": "layer1.",
                       "backbone.5.": "layer2.", "backbone.6.": "layer3.", "backbone.7.": "layer4."}


def load_backbone_weights(backbone: nn.Module, path: str) -> None:
    """Load an external encoder checkpoint (RadImageNet layout or plain timm keys) into a timm backbone."""
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.state_dict() if hasattr(sd, "state_dict") else sd
    out = {}
    for k, v in sd.items():
        for src, dst in _RADIMAGENET_PREFIX.items():
            if k.startswith(src):
                k = dst + k[len(src):]
                break
        out[k] = v
    missing, unexpected = backbone.load_state_dict(out, strict=False)
    missing = [k for k in missing if not k.startswith(("fc.", "head."))]
    if missing or unexpected:
        raise RuntimeError(f"pretrained_path {path}: missing {missing[:5]}, unexpected {unexpected[:5]}")


def freeze_vit_except_last(backbone: nn.Module, n: int) -> None:
    """Train only the last n transformer blocks and the final norm of a timm ViT (pilkwang's DINOv2 recipe)."""
    for p in backbone.parameters():
        p.requires_grad = False
    for mod in list(backbone.blocks[-n:]) + [getattr(backbone, a) for a in ("norm", "fc_norm") if hasattr(backbone, a)]:
        for p in mod.parameters():
            p.requires_grad = True


class AttentionMILHead(nn.Module):
    """Gated attention pooling with separate attention weights and classifier per finding."""

    def __init__(self, dim: int, n_out: int = NUM_LABELS, hidden: int = 256, drop: float = 0.2, gated: bool = True):
        super().__init__()
        self.gated = gated  # False = Raptor's plain tanh attention
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
        g = torch.tanh(self.att_v(h))
        if self.gated:
            g = g * torch.sigmoid(self.att_u(h))
        a = self.att_w(self.drop(g))  # B,K,N
        if mask is not None:
            mask = mask | ~mask.any(1, keepdim=True)  # an all-masked study attends to everything (no NaN)
            a = a.masked_fill(~mask[..., None], float("-inf"))
        a = torch.softmax(a.float(), dim=1).to(h.dtype)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled * self.cls_w).sum(-1) + self.cls_b


class TransformerMILHead(nn.Module):
    """goodpjw's reader head: project to `dim`, add slot + slice-position embeddings, a small transformer over all
    windows of the study, then one attention pooling per finding."""

    def __init__(self, dim_in: int, num_slots: int, n_out: int = NUM_LABELS, dim: int = 384, depth: int = 2,
                 drop: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.Dropout(drop), nn.Linear(dim_in, dim), nn.LayerNorm(dim))
        self.slot_emb = nn.Embedding(num_slots, dim)
        self.pos_emb = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(dim, 8, dim * 2, dropout=0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False) if depth else None
        self.att = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n_out))
        self.cls_w = nn.Parameter(torch.randn(n_out, dim) * 0.02)
        self.cls_b = nn.Parameter(torch.zeros(n_out))

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None, slot: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, k, _ = h.shape
        if slot is None:
            slot = torch.zeros(b, k, dtype=torch.long, device=h.device)
        if pos is None:
            pos = torch.zeros(b, k, device=h.device)
        h = self.proj(h) + self.slot_emb(slot) + self.pos_emb(pos.unsqueeze(-1).to(h.dtype))
        pad = ~mask if mask is not None else torch.zeros(b, k, dtype=torch.bool, device=h.device)
        pad = pad & ~pad.all(1, keepdim=True)  # a study with no series at all attends to everything
        if self.tf is not None:
            h = self.tf(h, src_key_padding_mask=pad)
        a = self.att(h).float().masked_fill(pad.unsqueeze(-1), -1e4).softmax(1)  # B,K,N
        pooled = torch.einsum("bkn,bkd->bnd", a.to(h.dtype), h)
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
        pretrained_path: Optional[str] = None,
        unfreeze_last: Optional[int] = None,
    ):
        super().__init__()
        kw = dict(pretrained=pretrained and not pretrained_path, num_classes=0, drop_path_rate=drop_path_rate)
        if img_size is not None and any(t in backbone for t in ("vit", "dinov2", "eva", "deit")):
            kw["img_size"] = img_size  # ViTs need their pos-embed resized at build time
        self.backbone = timm.create_model(backbone, **kw)
        if pretrained and pretrained_path:  # training only; inference loads our own checkpoint over it
            load_backbone_weights(self.backbone, pretrained_path)
        if unfreeze_last:
            freeze_vit_except_last(self.backbone, int(unfreeze_last))
        if grad_checkpointing:
            self.backbone.set_grad_checkpointing(True)
        # NHWC layout: faster convolutions with AMP on tensor-core GPUs (Ampere+); numerically equivalent
        self.channels_last = channels_last
        if channels_last:
            self.backbone.to(memory_format=torch.channels_last)
        dim = self.backbone.num_features
        self.head_type = head
        # transformer_mil carries its own slot / position embeddings at the head width
        self.slot_emb = nn.Embedding(num_slots, dim) if slot_embedding and head != "transformer_mil" else None
        if self.slot_emb is not None:
            nn.init.zeros_(self.slot_emb.weight)
        if head in ("attn_mil", "tanh_mil"):
            self.head = AttentionMILHead(dim, drop=drop_rate, gated=head == "attn_mil")
        elif head == "transformer_mil":
            self.head = TransformerMILHead(dim, num_slots, drop=drop_rate)
        else:
            self.head = PoolHead(dim, mode=head, drop=drop_rate)

    @property
    def data_config(self) -> dict:
        """Normalisation constants this backbone was pretrained with."""
        return timm.data.resolve_data_config({}, model=self.backbone)

    def forward(self, x: torch.Tensor, slot: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, k = x.shape[:2]
        x = x.flatten(0, 1)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        h = self.backbone(x).view(b, k, -1)
        if self.head_type == "transformer_mil":
            return self.head(h, mask, slot, pos)
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
        pretrained_path=m.get("pretrained_path"),
        unfreeze_last=m.get("unfreeze_last"),
    )
