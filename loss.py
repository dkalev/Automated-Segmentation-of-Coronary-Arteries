import torch
import torch.nn as nn
from metrics import dice_score


class DiceLoss(nn.Module):

    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize

    def forward(self, pred, targ):
        assert pred.size() == targ.size()
        if self.normalize: 
            pred = torch.sigmoid(pred)

        return 1. - dice_score(pred, targ)


class DiceBCELoss(nn.Module):
    def __init__(self, weighted_bce=True):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.functional.binary_cross_entropy_with_logits
        self.weighted_bce = weighted_bce

    @staticmethod
    def get_weight(targ):
        ratio = torch.sum(targ == 0) / torch.sum(targ == 1)
        weight = torch.ones_like(targ)
        weight[targ==1] = ratio
        return weight

    def forward(self, pred, targ):
        weight = self.get_weight(targ) if self.weighted_bce else None
        return self.dice(pred, targ) + self.bce(pred, targ, weight=weight)


class DiceBCE_OHNMLoss(nn.Module):
    """ Online Hard Negative Mining """
    def __init__(self, weighted_bce=True, ohnm_ratio=3):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.functional.binary_cross_entropy_with_logits
        self.weighted_bce = weighted_bce
        self.ohnm_ratio = ohnm_ratio
        
    def forward(self, pred, targ):
        pred, targ = pred.flatten(), targ.flatten()
        n_pos = targ.sum().int().item()
        n_neg = targ.numel() - n_pos
        assert n_pos * self.ohnm_ratio <= n_neg
        
        losses = self.bce(pred, targ, reduction='none')
        _, hns_idxs = losses[targ==0].topk(n_pos*self.ohnm_ratio)
        pos_idxs = torch.where(targ==1)[0]
        idxs = torch.cat([hns_idxs, pos_idxs])

        return self.dice(pred[idxs], targ[idxs]) + losses[idxs].mean()
