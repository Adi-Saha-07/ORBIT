"""
Loss Functions for Binary Satellite Change Detection in PyTorch.

Combines Binary Cross Entropy (BCE) with Dice Loss to handle the severe
class imbalance inherent in bi-temporal aerial change segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Differentiable Soft Dice Loss directly optimizing the F1 / overlap score.
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice

class BCEDiceLoss(nn.Module):
    """
    Hybrid Loss: BCE provides smooth pixel-level gradient propagation,
    while Dice Loss forces the model to focus on the sparse change regions.
    """
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return (self.bce_weight * loss_bce) + (self.dice_weight * loss_dice)
