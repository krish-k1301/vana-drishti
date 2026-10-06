"""Time-padding wrapper around upstream UNet3D (Tier A benchmark model).

Upstream UNet3D max-pools the time axis twice (kernel 2, stride 2) and crops the skips to the decoder's
length. Consequences: fewer than 4 dates crash, and when T is not a multiple of 4 the last T mod 4 dates
are cut from the decoder output and from the temporal mean. This wrapper appends zero frames so T becomes
the next multiple of 4 (at least 4); upstream already excludes all-`pad_value` frames from its final
temporal mean, so the appended frames only act like the batch padding upstream sees anyway.
"""
import torch
import torch.nn.functional as F

from src.models.vendor import import_upstream

_UNet3D = import_upstream().utae.unet3d.UNet3D

TIME_STRIDE = 4  # product of the two MaxPool3d(2) time strides in upstream UNet3D


def pad_time_to_multiple(images: torch.Tensor, multiple: int, value: float) -> torch.Tensor:
    """Append `value` frames to images [B,T,C,H,W] so T becomes the next positive multiple of `multiple`."""
    length = images.shape[1]
    target = max(multiple, -(-length // multiple) * multiple)
    return F.pad(images, (0, 0, 0, 0, 0, 0, 0, target - length), value=value)


class TimePaddedUNet3D(_UNet3D):
    """Upstream UNet3D that accepts any number of dates by zero-padding time to a multiple of 4."""

    def __init__(self, in_channel: int, n_classes: int, feats: int, pad_value: float, zero_pad: bool) -> None:
        """Build the upstream module with every constructor argument explicit."""
        super().__init__(
            in_channel=in_channel, n_classes=n_classes, feats=feats, pad_value=pad_value, zero_pad=zero_pad
        )

    def forward(self, images: torch.Tensor, batch_positions: torch.Tensor | None = None) -> torch.Tensor:
        """Map images [B,T,C,H,W] to logits [B,n_classes,H,W]; `batch_positions` is unused upstream."""
        return super().forward(pad_time_to_multiple(images, TIME_STRIDE, self.pad_value))
