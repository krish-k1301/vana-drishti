"""Galileo (nasaharvest) nano encoder + linear patch head for BraDD change segmentation.

Uses the vendored `single_file_galileo.py` and the nano weights shipped with it (strict load through
upstream `Encoder.load_from_folder`). Mapping of our batch onto Galileo's inputs:
  * images [B,T,2,H,W] (VV, VH z-scored with BraDD train stats) fill the S1 bands of `space_time_x`
    [B,H,W,T,13]; upstream pretraining normalises S1 dB as (x - (mean - m*std)) / (2*m*std) with m = 2
    (`src/data/dataset.py::Normalizer`), i.e. (z + m) / (2m) for a z-score, which is applied here
    (assumes BraDD dB statistics are close to Galileo's pretraining statistics).
  * every other band group (S2, NDVI, ERA5, TC, VIIRS, SRTM, DW, WC, LandScan, location) is masked
    (mask 1), exactly as upstream `GalileoWrapper.preproccess` does for S1-only input;
  * padded dates (day 0) get S1 mask 1, so upstream drops their tokens before attention;
  * months: BraDD batches carry day offsets, not calendar dates, so month = (start_month +
    floor((day - 1) / days_per_month)) mod 12 is a relative month code anchored at `start_month`;
  * upstream's time position table has `max_sequence_length` (24) entries, so a window with more real
    dates is reduced to that many, evenly spaced (first and last kept), deterministically.
Head: as upstream `src/eval/finetune.py::EncoderWithHead` for segmentation, a Linear from the S1 tokens
(mean over real dates, as `GalileoWrapper.forward` with do_pool=False) to num_classes * patch_size**2 logits
per patch, rearranged to pixels. `freeze_encoder=True` gives the linear probe (encoder frozen and in eval mode).
"""
from pathlib import Path

import torch
from einops import rearrange
from torch import nn

from src.models.segment_network import collate_windows
from src.models.tier_bc.common import masked_time_mean, padded_dates
from src.models.tier_bc.upstream import THIRD_PARTY_DIR, import_file

GALILEO = import_file(THIRD_PARTY_DIR / "galileo" / "single_file_galileo.py", "vanadrishti_single_file_galileo")
S1_GROUP = list(GALILEO.SPACE_TIME_BANDS_GROUPS_IDX).index("S1")
S1_BAND_IDX = [GALILEO.SPACE_TIME_BANDS.index(band) for band in GALILEO.S1_BANDS]
MONTHS_PER_YEAR = 12


def evenly_spaced_dates(images: torch.Tensor, days: torch.Tensor, limit: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep at most `limit` real dates per sample, evenly spaced over its real dates, re-padded at the end."""
    if images.shape[1] <= limit:
        return images, days
    kept_images, kept_days = [], []
    for sample_images, sample_days in zip(images, days):
        real = torch.nonzero(~padded_dates(sample_days)).flatten()
        if len(real) > limit:
            real = real[torch.linspace(0, len(real) - 1, limit).round().long()]
        kept_images.append(sample_images[real])
        kept_days.append(sample_days[real])
    return collate_windows(kept_images), collate_windows(kept_days)


class GalileoSeg(nn.Module):
    """Pretrained Galileo encoder on S1 VV/VH with a linear per-patch segmentation head."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        checkpoint_path: str,
        patch_size: int,
        max_timesteps: int,
        start_month: int,
        days_per_month: float,
        std_multiplier: float,
        add_layernorm_on_exit: bool,
        freeze_encoder: bool,
    ) -> None:
        """Load the encoder from folder `checkpoint_path` (config.json + encoder.pt, strict) and add the head."""
        super().__init__()
        if input_dim != len(GALILEO.S1_BANDS):
            raise ValueError(f"Galileo S1 input has {len(GALILEO.S1_BANDS)} bands (VV, VH), got {input_dim}")
        if not 0 <= start_month < MONTHS_PER_YEAR:
            raise ValueError(f"start_month must be in [0, {MONTHS_PER_YEAR}), got {start_month}")
        self.encoder = GALILEO.Encoder.load_from_folder(Path(checkpoint_path), device=torch.device("cpu"))
        if max_timesteps > self.encoder.max_sequence_length:
            raise ValueError(f"max_timesteps {max_timesteps} > encoder table {self.encoder.max_sequence_length}")
        if not 1 <= patch_size <= self.encoder.base_patch_size:
            raise ValueError(f"patch_size must be in [1, {self.encoder.base_patch_size}], got {patch_size}")
        self.head = nn.Linear(self.encoder.embedding_size, num_classes * patch_size ** 2)
        self.num_classes, self.patch_size, self.max_timesteps = num_classes, patch_size, max_timesteps
        self.start_month, self.days_per_month = start_month, days_per_month
        self.std_multiplier, self.add_layernorm_on_exit = std_multiplier, add_layernorm_on_exit
        self.freeze_encoder = freeze_encoder
        self.encoder.requires_grad_(not freeze_encoder)

    def train(self, mode: bool = True) -> "GalileoSeg":
        """Switch modes; a frozen (linear-probe) encoder always stays in eval mode (no drop path)."""
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def galileo_inputs(self, images: torch.Tensor, days: torch.Tensor) -> list[torch.Tensor]:
        """Build upstream Encoder inputs (s_t_x, sp_x, t_x, st_x, their masks, months) from our batch."""
        batch, steps, _, height, width = images.shape
        new = images.new_zeros
        s_t_x = new(batch, height, width, steps, len(GALILEO.SPACE_TIME_BANDS))
        scaled = (images + self.std_multiplier) / (2 * self.std_multiplier)
        s_t_x[..., S1_BAND_IDX] = scaled.permute(0, 3, 4, 1, 2)
        s_t_m = images.new_ones(batch, height, width, steps, len(GALILEO.SPACE_TIME_BANDS_GROUPS_IDX))
        s_t_m[..., S1_GROUP] = padded_dates(days).to(images.dtype)[:, None, None, :]
        sp_x = new(batch, height, width, len(GALILEO.SPACE_BANDS))
        sp_m = images.new_ones(batch, height, width, len(GALILEO.SPACE_BAND_GROUPS_IDX))
        t_x = new(batch, steps, len(GALILEO.TIME_BANDS))
        t_m = images.new_ones(batch, steps, len(GALILEO.TIME_BAND_GROUPS_IDX))
        st_x = new(batch, len(GALILEO.STATIC_BANDS))
        st_m = images.new_ones(batch, len(GALILEO.STATIC_BAND_GROUPS_IDX))
        months_elapsed = torch.div((days - 1).clamp(min=0), self.days_per_month, rounding_mode="floor")
        months = (self.start_month + months_elapsed.long()) % MONTHS_PER_YEAR
        return [s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months]

    def forward(self, images: torch.Tensor, days: torch.Tensor) -> torch.Tensor:
        """Map images [B,T,2,H,W] and days [B,T] (0 = padding) to logits [B,num_classes,H,W]."""
        height, width = images.shape[-2:]
        if height != width or height % self.patch_size:
            raise ValueError(f"Galileo needs square inputs divisible by patch_size, got {(height, width)}")
        images, days = evenly_spaced_dates(images, days, self.max_timesteps)
        out = self.encoder(*self.galileo_inputs(images, days), patch_size=self.patch_size,
                           add_layernorm_on_exit=self.add_layernorm_on_exit)
        s1_tokens = out[0][:, :, :, :, S1_GROUP]
        features = masked_time_mean(s1_tokens, padded_dates(days), time_dim=3)
        logits = self.head(features)
        return rearrange(logits, "b h w (c i j) -> b c (h i) (w j)", c=self.num_classes,
                         i=self.patch_size, j=self.patch_size)
