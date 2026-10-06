"""Binary focal loss matching upstream `source.loss.focal_loss.FocalLoss`, with no default alpha/gamma."""
import torch
from torch import nn


class FocalLoss(nn.Module):
    """Alpha-balanced focal loss on 2-class logits [N,2,H,W] against integer targets [N,H,W].

    loss = -mean over pixels of  w * (1 - p_t)^gamma * log(p_t + EPSILON),  w = alpha for positives and
    1 - alpha for negatives, p_t = softmax probability of the true class. Same formula, epsilon and
    NaN-skipping sum as upstream; alpha and gamma are required (upstream defaults gamma to 2.0, the
    paper's best is 1.0, PRD 4.4 issue 4).
    """

    EPSILON = 1e-6

    def __init__(self, alpha: float, gamma: float) -> None:
        """Store the positive-class weight `alpha` and focusing exponent `gamma`."""
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return the scalar loss; any non-zero target counts as the positive class (as upstream)."""
        if logits.shape[1] != 2:
            raise ValueError(f"FocalLoss expects 2 classes, got logits of shape {tuple(logits.shape)}")
        positive = target.bool()
        probs = torch.softmax(logits, dim=1)
        p_true = torch.where(positive, probs[:, 1], probs[:, 0])
        weight = torch.where(positive, self.alpha, 1.0 - self.alpha)
        terms = weight * (1.0 - p_true) ** self.gamma * torch.log(p_true + self.EPSILON)
        return -terms.nansum() / p_true.numel()
