# varifocal_loss.py

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .losses_utils import weighted_loss, weight_reduce_loss

@weighted_loss
def varifocal_loss(pred: Tensor,
                   target: Tensor,
                   alpha: float = 0.75,
                   gamma: float = 2.0,
                   iou_weighted: bool = True) -> Tensor:
    """Element-wise Varifocal Loss.

    Args:
        pred (Tensor): Raw logits of shape (...).
        target (Tensor): Continuous targets in [0, 1], same shape as pred.
        alpha (float): Weighting factor for negative examples.
        gamma (float): Focusing parameter.
        iou_weighted (bool): If True, positives are weighted by `target`.
                             If False, positives get weight 1 and negatives get α·p^γ.

    Returns:
        Tensor: Element-wise varifocal loss (no reduction). The wrapper
                handles `weight`, `reduction` and `avg_factor`.
    """
    # Convert logits to probabilities
    pred_sigmoid = pred.sigmoid()

    if iou_weighted:
        # w = t + α * (σ(p))^γ * (1 – t)
        weight = target + alpha * (pred_sigmoid.pow(gamma)) * (1 - target)
    else:
        # w = 1 for positive (t>0), α * (σ(p))^γ for negative
        weight = torch.where(
            target > 0,
            torch.ones_like(target),
            alpha * (pred_sigmoid.pow(gamma))
        )

    # Detach weight so that BCEWithLogits does not see requires_grad=True
    weight = weight.detach()

    # Compute BCE-with-logits using the detached weight
    loss = F.binary_cross_entropy_with_logits(
        pred,               # raw logits
        target,             # continuous target (e.g., Gaussian value)
        weight=weight,      # now detached
        reduction='none'    # no reduction here
    )
    return loss


class VarifocalLoss(nn.Module):
    """Varifocal Loss wrapper (supports optional sigmoid or already‐sigmoided input).

    Args:
        use_sigmoid (bool): If True, inputs to `forward()` are treated as
                            raw logits and varifocal_loss calls sigmoid internally.
                            If False, inputs are probabilities in [0,1].
        alpha (float): α parameter in weight computation.
        gamma (float): γ parameter in weight computation.
        iou_weighted (bool): Whether to weight positive samples by `target`.
        reduction (str): One of {"none", "mean", "sum"}.
        loss_weight (float): A global multiplier on the final loss.
    """
    def __init__(self,
                 use_sigmoid: bool = True,
                 alpha: float = 0.75,
                 gamma: float = 2.0,
                 iou_weighted: bool = True,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.alpha = alpha
        self.gamma = gamma
        self.iou_weighted = iou_weighted
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Tensor = None,
                avg_factor: float = None,
                reduction_override: str = None) -> Tensor:
        """Compute Varifocal Loss.

        Args:
            pred (Tensor): If `use_sigmoid=True`, these are raw logits.
                           If `use_sigmoid=False`, these are probabilities in [0,1].
            target (Tensor): Continuous targets in [0,1], same shape as `pred`.
            weight (Tensor, optional): Sample-wise weights (applied after elementwise loss).
            avg_factor (float, optional): Normalization factor (for 'mean' reduction).
            reduction_override (str, optional): Override the default `self.reduction`.

        Returns:
            Tensor: A single scalar (if reduction is 'mean'/'sum') or element-wise losses (if 'none'),
                    multiplied by `loss_weight`.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction

        if self.use_sigmoid:
            # Call elementwise varifocal_loss (already detaches weights internally)
            loss = varifocal_loss(
                pred,
                target,
                weight=weight,
                alpha=self.alpha,
                gamma=self.gamma,
                iou_weighted=self.iou_weighted,
                reduction=reduction,
                avg_factor=avg_factor
            )
        else:
            # Inputs are already probabilities; compute weight without gradient:
            pred_sigmoid = pred.clamp(min=1e-6, max=1.0 - 1e-6)
            if self.iou_weighted:
                vf_weight = target + self.alpha * (pred_sigmoid.pow(self.gamma)) * (1 - target)
            else:
                vf_weight = torch.where(
                    target > 0,
                    torch.ones_like(target),
                    self.alpha * (pred_sigmoid.pow(self.gamma))
                )
            vf_weight = vf_weight.detach()
            # Binary cross-entropy on probabilities
            loss = F.binary_cross_entropy(
                pred_sigmoid,
                target,
                weight=vf_weight,
                reduction='none'
            )
            # Apply sample-wise weight + reduce
            loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

        return self.loss_weight * loss
