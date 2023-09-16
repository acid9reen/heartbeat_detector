import torch
from torch.nn import functional as F


class DiceLoss(object):
    def __init__(self, smooth: float = 1e-6, gamma: float = 2) -> None:
        self._smooth = smooth
        self._gamma = gamma

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = input.view(-1)
        target = target.view(-1)

        intersection = (input * target).sum()
        dice = (
            (self._gamma * intersection + self._smooth) /
            (input.sum() + target.sum() + self._smooth)
        )

        return 1 - dice


class BceDiceLoss(DiceLoss):
    def __init__(self, smooth: float = 0.1e-6, gamma: float = 2) -> None:
        super().__init__(smooth, gamma)

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice_loss = super().__call__(input, target)
        bce_loss = F.binary_cross_entropy(input, target, reduction='mean')

        return dice_loss + bce_loss
