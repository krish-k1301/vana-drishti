"""TSViT (Tarasiou et al., CVPR 2023) for BraDD: vendored upstream `TSViT` module, padding-aware forward.

Upstream `models/TSViT/TSViTdense.py::TSViT.forward` takes [B,T,H,W,C+1] whose last channel is the day of
year / 365 and runs its temporal transformer over every date, padding included (upstream `Attention` has no
mask). This wrapper builds the upstream module unchanged (same parameters, same state-dict keys) and
re-implements only the forward:
  * input is images [B,T,C,H,W] and days [B,T] (days since the earliest image/label date, from 1; 0 = pad);
  * the temporal position is the one-hot of `day mod 366` fed to upstream's 366-way linear lookup
    (upstream indexes day-of-year; BraDD batches carry offsets, not calendar dates, so the table is read
    as a 366-day cyclic offset code);
  * padded dates are removed from the keys of every temporal attention layer (their queries are dropped
    anyway, only the class tokens are kept), so padding never changes the output.
With no padded dates the output equals upstream `TSViT.forward` (tested).
"""
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from src.models.tier_bc.common import padded_dates
from src.models.tier_bc.upstream import THIRD_PARTY_DIR, import_isolated

UPSTREAM_DIR = THIRD_PARTY_DIR / "deepsatmodels_tsvit"
_TSVIT_MODULE = "models.TSViT.TSViTdense"
UpstreamTSViT = import_isolated(UPSTREAM_DIR, ("models", "utils"), (_TSVIT_MODULE,))[_TSVIT_MODULE].TSViT


def masked_attention(prenorm_attn: nn.Module, x: torch.Tensor, key_bias: torch.Tensor) -> torch.Tensor:
    """Upstream `PreNorm(Attention)` with an additive key bias [N,1,1,L] (-inf on padded keys)."""
    attn = prenorm_attn.fn
    qkv = attn.to_qkv(prenorm_attn.norm(x)).chunk(3, dim=-1)
    q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=attn.heads) for t in qkv)
    dots = torch.einsum("b h i d, b h j d -> b h i j", q, k) * attn.scale + key_bias
    out = torch.einsum("b h i j, b h j d -> b h i d", dots.softmax(dim=-1), v)
    return attn.to_out(rearrange(out, "b h n d -> b n (h d)"))


def masked_transformer(transformer: nn.Module, x: torch.Tensor, key_valid: torch.Tensor) -> torch.Tensor:
    """Upstream `Transformer.forward` where keys with `key_valid` [N,L] False get zero attention."""
    key_bias = torch.zeros(key_valid.shape, dtype=x.dtype, device=x.device)
    key_bias = key_bias.masked_fill(~key_valid, float("-inf"))[:, None, None, :]
    for attn, ff in transformer.layers:
        x = masked_attention(attn, x, key_bias) + x
        x = ff(x) + x
    return transformer.norm(x)


class TSViTSeg(nn.Module):
    """Upstream TSViT (dense) mapping images [B,T,C,H,W], days [B,T] to logits [B,num_classes,H,W]."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        img_res: int,
        patch_size: int,
        max_seq_len: int,
        dim: int,
        temporal_depth: int,
        spatial_depth: int,
        heads: int,
        dim_head: int,
        dropout: float,
        emb_dropout: float,
        pool: str,
        scale_dim: int,
    ) -> None:
        """Build upstream TSViT; `num_channels` is input_dim + 1 as upstream counts its time channel."""
        super().__init__()
        if img_res % patch_size:
            raise ValueError(f"img_res {img_res} must be divisible by patch_size {patch_size}")
        self.tsvit = UpstreamTSViT({
            "img_res": img_res, "patch_size": patch_size, "num_classes": num_classes,
            "max_seq_len": max_seq_len, "dim": dim, "temporal_depth": temporal_depth,
            "spatial_depth": spatial_depth, "heads": heads, "dim_head": dim_head, "dropout": dropout,
            "emb_dropout": emb_dropout, "pool": pool, "scale_dim": scale_dim, "num_channels": input_dim + 1,
        })
        self.img_res = img_res
        self.day_table_size = self.tsvit.to_temporal_embedding_input.in_features

    def temporal_tokens(self, images: torch.Tensor, days: torch.Tensor) -> torch.Tensor:
        """Temporal encoder: per-patch class tokens [B, num_patches, num_classes, dim]."""
        model = self.tsvit
        batch, steps = days.shape
        day_code = F.one_hot(days.long() % self.day_table_size, self.day_table_size).to(images.dtype)
        temporal_pos = model.to_temporal_embedding_input(day_code)
        x = model.to_patch_embedding(images).reshape(batch, -1, steps, model.dim)
        n_patches = x.shape[1]
        x = (x + temporal_pos.unsqueeze(1)).reshape(-1, steps, model.dim)
        cls_tokens = repeat(model.temporal_token, "() n d -> b n d", b=batch * n_patches)
        valid = (~padded_dates(days)).repeat_interleave(n_patches, dim=0)
        key_valid = torch.cat([torch.ones_like(valid[:, : model.num_classes]), valid], dim=1)
        x = masked_transformer(model.temporal_transformer, torch.cat([cls_tokens, x], dim=1), key_valid)
        return x[:, : model.num_classes].reshape(batch, n_patches, model.num_classes, model.dim)

    def spatial_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        """Spatial encoder and head, unchanged from upstream: tokens [B,N,K,D] to logits [B,K,H,W]."""
        model = self.tsvit
        batch, n_patches, n_cls, dim = tokens.shape
        x = tokens.permute(0, 2, 1, 3).reshape(batch * n_cls, n_patches, dim)
        x = model.dropout(x + model.space_pos_embedding)
        x = model.mlp_head(model.space_transformer(x).reshape(-1, dim))
        x = x.reshape(batch, n_cls, n_patches, model.patch_size ** 2)
        side = model.num_patches_1d
        return rearrange(x, "b c (h1 w1) (p1 p2) -> b c (h1 p1) (w1 p2)", h1=side, w1=side,
                         p1=model.patch_size, p2=model.patch_size)

    def forward(self, images: torch.Tensor, days: torch.Tensor) -> torch.Tensor:
        """Map images [B,T,C,H,W] and days [B,T] (0 = padding) to logits [B,num_classes,H,W]."""
        if images.shape[-2:] != (self.img_res, self.img_res):
            raise ValueError(f"TSViT built for {self.img_res}x{self.img_res}, got {tuple(images.shape[-2:])}")
        return self.spatial_logits(self.temporal_tokens(images, days))
