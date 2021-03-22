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
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, targ):
        return self.dice(pred, targ) + self.bce(pred, targ)
