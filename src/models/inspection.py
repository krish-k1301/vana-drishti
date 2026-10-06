"""Model inspection helpers: per-stage U-TAE feature-map shapes and parameter counts."""
from collections.abc import Callable

import torch
from torch import nn


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Return the number of (trainable) parameters in `model`."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def _stage_modules(utae: nn.Module) -> dict[str, nn.Module]:
    """Name the encoder, temporal encoder and decoder stages of an upstream UTAE.

    Encoder blocks run through `smart_forward`, which calls `forward` directly and so skips hooks; the
    hooks go on their last inner layer instead, whose output has the block's output shape.
    """
    stages = {"in_conv": utae.in_conv.conv}
    stages.update({f"down_{i + 1}": block.conv2 for i, block in enumerate(utae.down_blocks)})
    stages["temporal_encoder"] = utae.temporal_encoder
    stages.update({f"up_{i + 1}": block for i, block in enumerate(utae.up_blocks)})
    stages["out_conv"] = utae.out_conv
    return stages


def describe_utae_shapes(utae: nn.Module, h: int, w: int, num_dates: int = 3) -> dict[str, tuple[int, ...]]:
    """Run one random sample through an upstream UTAE and return each stage's output shape [C,H,W].

    Shared spatial blocks run on all dates folded into the batch, so only the per-frame [C,H,W] is kept.
    """
    shapes: dict[str, tuple[int, ...]] = {}

    def hook(name: str) -> Callable[..., None]:
        """Return a forward hook that stores the output shape under `name`."""
        def record(_module: nn.Module, _inputs: tuple, output: torch.Tensor | tuple) -> None:
            """Store the last three dims of the (first) output."""
            tensor = output[0] if isinstance(output, tuple) else output
            shapes[name] = tuple(tensor.shape[-3:])
        return record

    handles = [m.register_forward_hook(hook(n)) for n, m in _stage_modules(utae).items()]
    was_training = utae.training
    utae.eval()
    try:
        in_channels = next(m for m in utae.in_conv.modules() if isinstance(m, nn.Conv2d)).in_channels
        images = torch.randn(1, num_dates, in_channels, h, w)
        days = torch.arange(1, num_dates + 1)[None]
        with torch.no_grad():
            utae(images, days)
    finally:
        utae.train(was_training)
        for handle in handles:
            handle.remove()
    return shapes
