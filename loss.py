import torch
import torch.nn as nn

class DiceLoss(nn.Module):

    def __init__(self, normalize=True):
        super().__init__()
        self.smooth = 1.0
        self.normalize = normalize

    def forward(self, pred, targ):
        assert pred.size() == targ.size()
        if self.normalize: 
            pred = torch.sigmoid(pred)

        intersection = (pred * targ).sum()
        dice_score = (2. * intersection + self.smooth) / ( pred.sum() + targ.sum() + self.smooth )
        return 1. - dice_score

class DiceBCELoss(nn.Module):
    def __init__(self, weighted_bce=True):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.functional.binary_cross_entropy_with_logits
        self.weighted_bce = weighted_bce

    def forward(self, pred, targ):
        ratio = torch.sum(targ == 0) / torch.sum(targ == 1)
        weight = torch.ones_like(targ)
        weight[targ==1] = ratio
        if self.weighted_bce:
            return self.dice(pred, targ) + self.bce(pred, targ, weight=weight)
        else:
            return self.dice(pred, targ) + self.bce(pred, targ)
