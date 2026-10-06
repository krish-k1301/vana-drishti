"""AnySat (Astruc et al., CVPR 2025) encoder + linear dense head for BraDD change segmentation.

BLOCKED here: the pretrained weights are only on huggingface.co, which this environment cannot reach, so
`checkpoint_path: null` builds a randomly initialised encoder (tests only). With the owner's copy of
`AnySat.pth` (hub URL in vendored `hubconf.py::AnySat.from_pretrained`) the encoder loads strictly, as upstream.

Built through vendored `hubconf.AnySat` (the released feature extractor) with `output='dense'`, which returns
per-pixel features [B,H,W,2*D] (patch token concatenated with its sub-patch token). Head as upstream
`src/models/networks/Fine_tuning.py::Fine` with keep_subpatch: LayerNorm(2*D) + Linear(2*D, num_classes).
Inputs: S1 VV/VH z-scored (AnySat expects (x - mean) / std) under modality `s1-asc`, the only 2-band S1
projector (AnySat's own BraDD runs used 3-band `s1` = VV, VH, VV/VH ratio and another checkpoint).
Dates: day offsets from 1, as AnySat's own BraDD loader (`src/data/BraDD.py`) builds them.
Padding: upstream `PatchLTAEMulti` does not pass a temporal pad mask to its L-TAE, so padded dates would be
attended. The wrapper therefore runs the encoder once per sample on that sample's real dates only.
`freeze_encoder=True` is the linear probe (encoder frozen and kept in eval mode).
"""
import importlib
from types import ModuleType

import torch
from torch import nn

from src.models.tier_bc.common import padded_dates
from src.models.tier_bc.upstream import THIRD_PARTY_DIR, isolated_sys_path

UPSTREAM_DIR = THIRD_PARTY_DIR / "anysat"
_TOP_LEVEL = ("hubconf", "src", "models")
RES_STEP_M = 10  # hubconf.AnySat.forward divides patch_size (metres) by 10 m, its base resolution


def _hubconf() -> ModuleType:
    """Import the vendored hubconf module (its own `src` package must not clash with ours)."""
    with isolated_sys_path(UPSTREAM_DIR, _TOP_LEVEL):
        return importlib.import_module("hubconf")


HUBCONF = _hubconf()


def build_anysat(model_size: str, flash_attn: bool) -> nn.Module:
    """Construct upstream `hubconf.AnySat` (it imports its encoder modules lazily, hence the isolation)."""
    with isolated_sys_path(UPSTREAM_DIR, _TOP_LEVEL):
        return HUBCONF.AnySat(model_size=model_size, flash_attn=flash_attn)


class AnySatSeg(nn.Module):
    """AnySat dense features + LayerNorm/Linear head: images [B,T,C,H,W], days [B,T] to logits [B,K,H,W]."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        model_size: str,
        modality: str,
        patch_size_m: int,
        flash_attn: bool,
        checkpoint_path: str | None,
        freeze_encoder: bool,
    ) -> None:
        """Build the encoder, load `checkpoint_path` strictly if given, add the head."""
        super().__init__()
        self.anysat = build_anysat(model_size, flash_attn)
        projector = getattr(self.anysat.model, f"projector_{modality}", None)
        if projector is None or "T" not in HUBCONF.get_default_config(model_size)["projectors"][modality]:
            raise ValueError(f"AnySat has no time-series projector for modality '{modality}'")
        if projector.patch_embed.in_channels != input_dim:
            raise ValueError(f"modality '{modality}' takes {projector.patch_embed.in_channels} bands, not {input_dim}")
        if patch_size_m % RES_STEP_M:
            raise ValueError(f"patch_size_m must be a multiple of {RES_STEP_M}, got {patch_size_m}")
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["state_dict"]
            self.anysat.model.load_state_dict(state)
        width = 2 * self.anysat.model.embed_dim
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, num_classes))
        self.modality, self.patch_size_m, self.freeze_encoder = modality, patch_size_m, freeze_encoder
        self.anysat.requires_grad_(not freeze_encoder)

    def train(self, mode: bool = True) -> "AnySatSeg":
        """Switch modes; a frozen (linear-probe) encoder always stays in eval mode."""
        super().train(mode)
        if self.freeze_encoder:
            self.anysat.eval()
        return self

    def forward(self, images: torch.Tensor, days: torch.Tensor) -> torch.Tensor:
        """Map images [B,T,C,H,W] and days [B,T] (0 = padding) to logits [B,num_classes,H,W]."""
        real = ~padded_dates(days)
        features = [
            self.anysat(
                {self.modality: images[b : b + 1, real[b]], f"{self.modality}_dates": days[b : b + 1, real[b]]},
                patch_size=self.patch_size_m, output="dense", output_modality=self.modality,
            )
            for b in range(images.shape[0])
        ]
        return self.head(torch.cat(features)).permute(0, 3, 1, 2)
