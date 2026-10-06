"""Exchanger + U-Net (Cai et al., BMVC 2023) for BraDD, built from vendored Exchanger4SITS modules.

Reproduces the `space_encoder_type == 'unet'` branch of upstream `lib/models/sem_seg.py::Segmentor`:
per-pixel Exchanger temporal encoder (collect-update-distribute over learned group tokens), masked mean over
dates, upstream `UNet`, 1x1 conv head. Upstream `Segmentor` itself is not used because it also builds the
loss and parses PASTIS dict batches (and imports the dataset readers); its forward is mirrored here with
images [B,T,C,H,W] and days [B,T] (0 = padding) as input. Differences, all about padding:
  * the padding mask comes from days == 0 (upstream reads a per-pixel mask channel from the batch);
  * upstream `Exchanger.forward` calls its MLP layers without the mask, so with `mlp_norm: batchnorm`
    padded dates enter the BatchNorm statistics in training. The wrapper passes the mask to those layers,
    which upstream `NormLayer` honours with `mlp_norm: maskedbatchnorm` (upstream's own masked variant).
Day offsets index upstream's sinusoid table (`TemporalPositionalEncoding`, type 'default'), whose row 0 is
reserved for padding; real days must be <= `max_temp_len`.
"""
from types import SimpleNamespace

import torch
from torch import nn

from src.models.tier_bc.common import masked_time_mean, padded_dates, require_day_range, require_keys
from src.models.tier_bc.upstream import THIRD_PARTY_DIR, import_isolated

UPSTREAM_DIR = THIRD_PARTY_DIR / "exchanger4sits" / "lib"
_MODULES = import_isolated(UPSTREAM_DIR, ("layers", "modules", "ops", "utils", "losses"), ("modules", "layers"))
Exchanger, UNet = _MODULES["modules"].Exchanger, _MODULES["modules"].UNet
TemporalPositionalEncoding = _MODULES["layers"].TemporalPositionalEncoding

TEMPORAL_ENCODING_KEYS = ("pos_encode_type", "pe_dim", "pe_t", "max_temp_len", "with_gdd_pos")
EXCHANGER_KEYS = ("embed_dims", "num_token_list", "num_heads_list", "drop_path_rate", "mlp_norm", "mlp_act")
GP_BLOCK_KEYS = (
    "add_pos_token", "act_type", "norm_type", "ffn_ratio", "qkv_bias", "drop", "attn_drop",
    "mixer_depth", "mixer_token_expansion", "mixer_channel_expansion", "untied_pos_encode",
)
UNET_KEYS = (
    "base_channels", "num_stages", "strides", "enc_num_convs", "dec_num_convs", "downsamples",
    "enc_dilations", "dec_dilations", "norm_type", "act_type", "upsample_type",
)


def _upper(section: dict) -> dict:
    """Upper-case the keys of one config section, as upstream yacs configs spell them."""
    return {key.upper(): value for key, value in section.items()}


def upstream_config(exchanger: dict, gp_block: dict, unet: dict) -> SimpleNamespace:
    """Upstream yacs-style config object from our lower-case YAML sections.

    GPBLOCK's embed_dims, num_group_tokens, num_heads and drop_path are not listed: upstream `Exchanger`
    overrides them per stage from the EXCHANGER section.
    """
    return SimpleNamespace(EXCHANGER=_upper(exchanger), GPBLOCK=_upper(gp_block), UNET=_upper(unet))


class ExchangerUNet(nn.Module):
    """Upstream Exchanger temporal encoder + UNet + 1x1 head: images [B,T,C,H,W] to logits [B,K,H,W]."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        temporal_encoding: dict,
        exchanger: dict,
        gp_block: dict,
        unet: dict,
    ) -> None:
        """Build upstream modules from the four config sections (mirrors upstream `Segmentor.__init__`)."""
        super().__init__()
        require_keys("exchanger_unet.temporal_encoding", temporal_encoding, TEMPORAL_ENCODING_KEYS)
        require_keys("exchanger_unet.exchanger", exchanger, EXCHANGER_KEYS)
        require_keys("exchanger_unet.gp_block", gp_block, GP_BLOCK_KEYS)
        require_keys("exchanger_unet.unet", unet, UNET_KEYS)
        if temporal_encoding["with_gdd_pos"]:
            raise ValueError("with_gdd_pos needs growing-degree-day inputs, which BraDD does not have")
        config = upstream_config(exchanger, gp_block, unet)
        self.temp_pos_encode = TemporalPositionalEncoding(
            temporal_encoding["pos_encode_type"], temporal_encoding["pe_dim"], T=temporal_encoding["pe_t"],
            with_gdd_pos=False, max_len=temporal_encoding["max_temp_len"],
        )
        self.temp_encoder = Exchanger(config, in_channels=input_dim, pe_dim=temporal_encoding["pe_dim"])
        self.space_encoder = UNet(config, in_channels=self.temp_encoder.out_dim)
        self.cls_head = nn.Conv2d(self.space_encoder.out_dims[0], num_classes, 1)
        self.max_temp_len = temporal_encoding["max_temp_len"]

    def encode_time(self, images: torch.Tensor, days: torch.Tensor) -> torch.Tensor:
        """Per-pixel Exchanger over real dates, masked mean over time: returns features [B,D,H,W]."""
        batch, steps, channels, height, width = images.shape
        pixels = height * width
        padded = padded_dates(days)
        x = images.permute(0, 3, 4, 1, 2).reshape(batch * pixels, steps, channels)
        mask = padded.repeat_interleave(pixels, dim=0)
        temp_pos = self.temp_pos_encode(days.long(), key_padding_mask=padded)
        temp_pos = temp_pos.repeat_interleave(pixels, dim=0).transpose(0, 1)
        encoder = self.temp_encoder
        for mlp, block in zip(encoder.mlp_layers, encoder.temp_encoder_blocks):
            x = mlp(x, mask)
            x = block(x.transpose(0, 1), temp_pos, mask)[0].transpose(0, 1)
        x = masked_time_mean(x, mask, time_dim=1)
        return x.reshape(batch, pixels, -1).transpose(1, 2).reshape(batch, -1, height, width)

    def forward(self, images: torch.Tensor, days: torch.Tensor) -> torch.Tensor:
        """Map images [B,T,C,H,W] and days [B,T] (0 = padding) to logits [B,num_classes,H,W]."""
        require_day_range(days, self.max_temp_len, "exchanger_unet")
        features = self.encode_time(images, days)
        return self.cls_head(self.space_encoder(features)[-1])
