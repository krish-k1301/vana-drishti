"""Padding-aware wrappers around upstream ConvLSTM_Seg / ConvGRU_Seg (Tier A benchmark models).

Upstream runs the recurrence over every time step, including the zero frames appended as padding, so a
sample's prediction depends on how long the other samples in its batch are. Here the state is frozen
on padded steps (padding is always at the end, so the result is the state after the last real date).
On inputs without padding the output is identical to upstream; weights and module names are unchanged.
"""
import torch

from src.models.vendor import import_upstream

_SOURCE = import_upstream()
_ConvLSTM_Seg = _SOURCE.utae.convlstm.ConvLSTM_Seg
_ConvGRU_Seg = _SOURCE.utae.convgru.ConvGRU_Seg

_SUPPORTED_KERNEL = (3, 3)


def _check_kernel(kernel_size: tuple[int, int]) -> None:
    """Upstream's classification conv hard-codes padding=1, so only 3x3 kernels keep H and W."""
    if tuple(kernel_size) != _SUPPORTED_KERNEL:
        raise ValueError(f"kernel_size must be {_SUPPORTED_KERNEL} (upstream head uses padding=1)")


def frame_pad_mask(images: torch.Tensor, pad_value: float) -> torch.Tensor:
    """Return the [B,T] mask of frames whose every value equals `pad_value` (upstream convention)."""
    return (images == pad_value).flatten(2).all(dim=-1)


def _keep(step_valid: torch.Tensor) -> torch.Tensor:
    """Broadcast a [B] validity vector to [B,1,1,1]."""
    return step_valid[:, None, None, None]


class MaskedConvLSTMSeg(_ConvLSTM_Seg):
    """Upstream ConvLSTM_Seg whose recurrence skips padded dates; head reads the last cell state."""

    def __init__(
        self,
        num_classes: int,
        input_size: tuple[int, int],
        input_dim: int,
        hidden_dim: int,
        kernel_size: tuple[int, int],
        pad_value: float,
    ) -> None:
        """Build the upstream module with every constructor argument explicit."""
        _check_kernel(kernel_size)
        super().__init__(
            num_classes=num_classes,
            input_size=tuple(input_size),
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=tuple(kernel_size),
            pad_value=pad_value,
        )

    def forward(self, images: torch.Tensor, batch_positions: torch.Tensor | None = None) -> torch.Tensor:
        """Map images [B,T,C,H,W] to logits [B,num_classes,H,W]; `batch_positions` is unused."""
        valid = ~frame_pad_mask(images, self.pad_value)
        cell = self.convlstm_encoder.cell_list[0]
        h, c = cell.init_hidden(images.shape[0], images.device)
        for t in range(images.shape[1]):
            h_new, c_new = cell(input_tensor=images[:, t], cur_state=[h, c])
            keep = _keep(valid[:, t])
            h = torch.where(keep, h_new, h)
            c = torch.where(keep, c_new, c)
        return self.classification_layer(c)


class MaskedConvGRUSeg(_ConvGRU_Seg):
    """Upstream ConvGRU_Seg whose recurrence skips padded dates; head reads the last hidden state."""

    def __init__(
        self,
        num_classes: int,
        input_size: tuple[int, int],
        input_dim: int,
        hidden_dim: int,
        kernel_size: tuple[int, int],
        pad_value: float,
    ) -> None:
        """Build the upstream module with every constructor argument explicit."""
        _check_kernel(kernel_size)
        super().__init__(
            num_classes=num_classes,
            input_size=tuple(input_size),
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=tuple(kernel_size),
            pad_value=pad_value,
        )

    def forward(self, images: torch.Tensor, batch_positions: torch.Tensor | None = None) -> torch.Tensor:
        """Map images [B,T,C,H,W] to logits [B,num_classes,H,W]; `batch_positions` is unused."""
        valid = ~frame_pad_mask(images, self.pad_value)
        cell = self.convgru_encoder.cell_list[0]
        h = cell.init_hidden(images.shape[0], images.device)
        for t in range(images.shape[1]):
            h_new = cell(input_tensor=images[:, t], cur_state=h)
            h = torch.where(_keep(valid[:, t]), h_new, h)
        return self.classification_layer(h)
