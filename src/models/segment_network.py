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
    (strict, as upstream), or `... <= end + error_days_after` with `inclusive_end=True` (Phase 4: with a
    trailing margin of 0 the image acquired exactly at the cutoff t_c is used, nothing after it).
    Differences from upstream `Network._segment_wise`, all deliberate:
      * `target_days` is never modified (upstream shifts views of it in place, so for t > 2 every interval
        after the first uses `start` itself as lower bound, and the caller's tensor is changed).
      * margins before/after are separate (Phase 4 needs a trailing margin of 0).
      * with an explicit `pad_mask`, padded dates are never selected and padded frames are set to exactly
        `PAD_VALUE` before the backbone sees them, because upstream backbones infer padding from
        frames equal to `pad_value` everywhere (PRD 4.4 issue 6). A real frame that is entirely `PAD_VALUE`
        would be mistaken for padding, so it raises instead of silently disappearing.
      * an interval is invalid when one of its label days is padding (`target_days == 0`, collate pads
        variable-length label sequences with 0) or its window holds no image. The backbone only sees valid
        rows; invalid ones get zero logits and must be dropped from loss and metrics with `interval_mask`.
    Without `pad_mask` selection is by day only, exactly like upstream.
    """

    def __init__(self, backbone: nn.Module, error_days_before: int, error_days_after: int,
                 inclusive_end: bool = False) -> None:
        """Wrap `backbone`, which maps (images [B,T,C,H,W], days [B,T]) to logits [B,L,H,W].

        `inclusive_end=False` is upstream's strict bound; `build_model` always passes it from config.
        """
        super().__init__()
        self.backbone = backbone
        self.error_days_before = error_days_before
        self.error_days_after = error_days_after
        self.inclusive_end = inclusive_end

    def window(self, days: torch.Tensor, target_days: torch.Tensor, valid: torch.Tensor, i: int) -> torch.Tensor:
        """[B,T] mask of the dates used for label interval i (valid = non-padded dates)."""
        lower = target_days[:, i] - self.error_days_before
        upper = target_days[:, i + 1] + self.error_days_after
        below_upper = days <= upper[:, None] if self.inclusive_end else days < upper[:, None]
        return valid & (lower[:, None] < days) & below_upper

    def interval_mask(self, days: torch.Tensor, target_days: torch.Tensor,
                      pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """[B, t-1] bool: True where the interval has two real label days and >= 1 image in its window."""
        valid = torch.ones_like(days, dtype=torch.bool) if pad_mask is None else ~pad_mask
        real_labels = target_days > 0
        columns = [real_labels[:, i] & real_labels[:, i + 1] & self.window(days, target_days, valid, i).any(dim=1)
                   for i in range(target_days.shape[1] - 1)]
        return torch.stack(columns, dim=1)

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
        rows_ok = self.interval_mask(days, target_days, pad_mask)
        if not rows_ok.any():
            raise ValueError("no label interval in the batch has a real label pair and an image in its window")
        results = []
        for i in range(target_days.shape[1] - 1):
            rows = rows_ok[:, i]
            if rows.any():
                keep = self.window(days, target_days, valid, i)[rows]
                results.append((i, rows, self.backbone(*self._select(images[rows], days[rows], keep))))
        if rows_ok.all():
            return torch.stack([logits for _, _, logits in results], dim=1)
        template = results[0][2]
        out = template.new_zeros((days.shape[0], target_days.shape[1] - 1, *template.shape[1:]))
        for i, rows, logits in results:
            out[rows, i] = logits
        return out

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
        batch = range(images.shape[0])
        window_images = collate_windows([images[b, keep[b]] for b in batch])
        window_days = collate_windows([days[b, keep[b]] for b in batch])
        return window_images, window_days
