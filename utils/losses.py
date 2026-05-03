import torch
import torch.nn as nn


class LpLoss(nn.Module):
    def __init__(self, p=2, reduction='mean'):
        super().__init__()
        self.p = p
        self.reduction = reduction

    def forward(self, pred, target, mask=None):
        """
        pred, target: (B, N) or (B, s1, s2)
        mask: optional (B, s1, s2) binary, 1=physical domain
        """
        if mask is not None:
            pred   = pred * mask
            target = target * mask

        diff = torch.norm(
            pred.reshape(pred.shape[0], -1) - target.reshape(target.shape[0], -1),
            p=self.p, dim=1
        )
        norm = torch.norm(
            target.reshape(target.shape[0], -1),
            p=self.p, dim=1
        )
        loss = diff / (norm + 1e-8)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss