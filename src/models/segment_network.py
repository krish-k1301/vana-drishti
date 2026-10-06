"""Segment-wise forward pass (re-implementation of upstream `Network._segment_wise`) for any backbone."""
import torch
from torch import nn

PAD_VALUE = 0


def collate_windows(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Stack variable-length [T_b, ...] tensors, zero-padding at the END of the time axis.

    Same result as upstream `source.data_module.collate_tensor`.
    """
    length = max(t.shape[0] for t in tensors)
    padded = [
        torch.cat([t, t.new_full((length - t.shape[0], *t.shape[1:]), PAD_VALUE)]) for t in tensors
    ]
    return torch.stack(padded)


class SegmentNetwork(nn.Module):
    """Run a backbone once per consecutive label interval and stack the logits to [B, t-1, L, H, W].

    For interval (start, end) the dates kept are `start - error_days_before < day < end + error_days_after`
    (strict, as upstream). Differences from upstream `Network._segment_wise`, all deliberate:
      * `target_days` is never modified (upstream shifts views of it in place, so for t > 2 every interval
        after the first uses `start` itself as lower bound, and the caller's tensor is changed).
      * margins before/after are separate (Phase 4 needs a trailing margin of 0).
      * with an explicit `pad_mask`, padded dates are never selected and padded frames are set to exactly
        `PAD_VALUE` before the backbone sees them, because upstream backbones infer padding from
        frames equal to `pad_value` everywhere (PRD 4.4 issue 6). A real frame that is entirely `PAD_VALUE`
        would be mistaken for padding, so it raises instead of silently disappearing.
    Without `pad_mask` selection is by day only, exactly like upstream.
    """

    def __init__(self, backbone: nn.Module, error_days_before: int, error_days_after: int) -> None:
        """Wrap `backbone`, which maps (images [B,T,C,H,W], days [B,T]) to logits [B,L,H,W]."""
        super().__init__()
        self.backbone = backbone
        self.error_days_before = error_days_before
        self.error_days_after = error_days_after

    def forward(
        self,
        images: torch.Tensor,
        days: torch.Tensor,
        target_days: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits [B, t-1, L, H, W] for images [B,T,C,H,W], days [B,T], target_days [B,t]."""
        valid = torch.ones_like(days, dtype=torch.bool)
        if pad_mask is not None:
            images = self._apply_pad_mask(images, pad_mask)
            valid = ~pad_mask
        out = []
        for i in range(target_days.shape[1] - 1):
            lower = target_days[:, i] - self.error_days_before
            upper = target_days[:, i + 1] + self.error_days_after
            keep = valid & (lower[:, None] < days) & (days < upper[:, None])
            window_images, window_days = self._select(images, days, keep)
            out.append(self.backbone(window_images, window_days))
        return torch.stack(out, dim=1)

    @staticmethod
    def _apply_pad_mask(images: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """Set padded frames to PAD_VALUE and reject real frames that would look like padding."""
        all_pad = (images == PAD_VALUE).flatten(2).all(dim=-1)
        if (all_pad & ~pad_mask).any():
            raise ValueError(
                f"a non-padded frame is entirely {PAD_VALUE}; the backbone would treat it as padding"
            )
        return images.masked_fill(pad_mask[:, :, None, None, None], PAD_VALUE)

    @staticmethod
    def _select(
        images: torch.Tensor, days: torch.Tensor, keep: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather the kept dates of every sample and re-collate with end padding."""
        counts = keep.sum(dim=1)
        if (counts == 0).any():
            empty = torch.nonzero(counts == 0).flatten().tolist()
            raise ValueError(f"no image dates inside the label window for batch rows {empty}")
        batch = range(images.shape[0])
        window_images = collate_windows([images[b, keep[b]] for b in batch])
        window_days = collate_windows([days[b, keep[b]] for b in batch])
        return window_images, window_days
