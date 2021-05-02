import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self, normalize=True, eps=1e-10):
        super().__init__()
        self.normalize = normalize
        self.eps = eps

    def forward(self, pred, targ):
        assert pred.size() == targ.size()
        if self.normalize: 
            pred = torch.sigmoid(pred)
        
        targ = targ.type(pred.dtype)
        intersection = torch.einsum("bc...,bc...->bc", pred, targ)

        union = (torch.einsum("bc...->bc", pred) + torch.einsum("bc...->bc", targ))

        loss = 1 - (2 * intersection + self.eps) / (union + self.eps)

        return loss.mean()


class GeneralizedDice(nn.Module):
    def __init__(self, normalize=True, eps=1e-10):
        super().__init__()
        self.normalize = normalize
        self.eps = eps

    def forward(self, pred, targ):
        assert pred.size() == targ.size()
        if self.normalize: 
            pred = torch.sigmoid(pred)

        targ = targ.type(pred.dtype)
        w = 1 / ((torch.einsum("bc...->bc", targ) + self.eps) ** 2)
        intersection = w * torch.einsum("bc...,bc...->bc", pred, targ)
        union = w * (torch.einsum("bc...->bc", pred) + torch.einsum("bc...->bc", targ))

        loss = 1 - 2 * (torch.einsum("bc->b", intersection) + self.eps) / (torch.einsum("bc->b", union) + self.eps)
        return loss.mean()


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
        self.dice = GeneralizedDice()
        self.bce = nn.functional.binary_cross_entropy_with_logits
        self.weighted_bce = weighted_bce
        self.ohnm_ratio = ohnm_ratio
        
    def forward(self, pred, targ):
        n_pos = targ.sum().int().item()
        n_neg = targ.numel() - n_pos
        assert n_pos * self.ohnm_ratio <= n_neg
        
        losses = self.bce(pred, targ, reduction='none')
        n_hns = n_pos*self.ohnm_ratio if n_pos != 0 else int(0.1 * n_neg)
        _, hns_idxs = losses[targ==0].flatten().topk(n_hns)
        pos_idxs = torch.nonzero(targ.flatten()==1)
        idxs = torch.cat([hns_idxs.flatten(), pos_idxs.flatten()])

        pred = pred.flatten()[idxs].view(*pred.shape[:2], -1)
        targ = targ.flatten()[idxs].view(*targ.shape[:2], -1)
        losses = losses.flatten()[idxs]

        return self.dice(pred, targ) + losses.mean()
