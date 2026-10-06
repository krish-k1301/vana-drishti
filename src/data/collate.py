"""Batch collation with end-padding of time axes and an explicit padding mask (PRD 4.4 issue 6)."""
from __future__ import annotations

import torch
from torch.nn.functional import pad


def pad_stack(tensors: list[torch.Tensor], value: float = 0) -> torch.Tensor:
    """Pad dim 0 of every tensor at the END to the longest length, then stack (upstream collate_tensor)."""
    longest = max(int(t.shape[0]) if t.dim() > 0 else 0 for t in tensors)
    padded = []
    for t in tensors:
        if t.dim() > 0 and t.shape[0] < longest:
            t = pad(t, [0, 0] * (t.dim() - 1) + [0, longest - t.shape[0]], value=value)
        padded.append(t)
    return torch.stack(padded)


def collate_batch(samples: list[dict]) -> dict:
    """Collate item dicts into a batch; adds `PadMask` [B,T] (True = padded date).

    Time-like axes (Images/ImageDays along T, Targets/TargetDays along t) are padded with 0 at
    the end. Padded label dates are therefore recognisable by `TargetDays == 0`.
    """
    batch = {key: pad_stack([s[key] for s in samples]) for key in samples[0]}
    lengths = torch.tensor([int(s["ImageDays"].shape[0]) for s in samples])
    batch["PadMask"] = torch.arange(batch["ImageDays"].shape[1])[None, :] >= lengths[:, None]
    return batch
