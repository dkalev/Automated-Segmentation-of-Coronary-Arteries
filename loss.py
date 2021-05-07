import torch
import torch.nn as nn


class BaseLoss(nn.Module):
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize
    
    def __call__(self, preds, targs):
        assert len(preds.shape) > 2
        assert preds.size() == targs.size()

        if self.normalize:
            preds = torch.sigmoid(preds)

        targs = targs.type(preds.dtype)
        return self.forward(preds, targs)


class DiceLoss(BaseLoss):
    def __init__(self, *args, generalized=False, eps=1e-10, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps
        self.generalized = generalized
    
    def get_weight(self, targs):
        if not self.generalized: return torch.ones(targs.shape[:2]).to(targs.device)
        return  1 / ((torch.einsum("bc...->bc", targs) + self.eps) ** 2)

    def forward(self, preds, targs):
        w = self.get_weight(targs)
        intersection = w * torch.einsum('bc...,bc...->bc', preds, targs)
        denom = w * (torch.einsum("bc...->bc", preds) + torch.einsum("bc...->bc", targs))
        loss = 1 - (2 * intersection + self.eps) / (denom + self.eps)
        return loss.mean()


class CombinedLoss(BaseLoss):
    def __init__(self, *args, dice_kwargs=None, **kwargs):
        normalize = 'normalize' in kwargs and kwargs['normalize']
        if normalize: del kwargs['normalize'] # don't pass to base class to avoid normalizing twice
        super().__init__(*args, **kwargs)

        if normalize:
            self.bce = nn.functional.binary_cross_entropy_with_logits
        else:
            self.bce = nn.functional.binary_cross_entropy
        dice_kwargs = {'normalize': normalize} if dice_kwargs is None else dice_kwargs
        self.dice = DiceLoss(**dice_kwargs)


class DiceBCELoss(CombinedLoss):
    def __init__(self, *args, weighted_bce=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.weighted_bce = weighted_bce
        
    @staticmethod
    def get_weight(targs):
        ratio = torch.sum(targs == 0) / torch.sum(targs == 1)
        weight = torch.ones_like(targs)
        weight[targs==1] = ratio
        return weight

    def forward(self, preds, targs):
        weight = self.get_weight(targs) if self.weighted_bce else None
        return self.dice(preds, targs) + self.bce(preds, targs, weight=weight)


class DiceBCE_OHNMLoss(CombinedLoss):
    """ Online Hard Negative Mining """
    def __init__(self, *args, ohnm_ratio=3, default_neg_perc=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.ohnm_ratio = ohnm_ratio
        # percentage of negative samples used in patches without positive samples
        self.default_neg_perc = default_neg_perc

    def get_num_hns(self, targs):
        n_pos = targs.sum().int().item()
        n_neg = targs.numel() - n_pos
        if n_pos == 0:
            return int(self.default_neg_perc * n_neg)
        return min(n_pos * self.ohnm_ratio, n_neg)

    def get_idxs(self, losses, targs):
        n_hns = self.get_num_hns(targs)
        _, hns_idxs = losses[targs==0].flatten().topk(n_hns)
        pos_idxs = torch.nonzero(targs.flatten()==1)
        return torch.cat([hns_idxs.flatten(), pos_idxs.flatten()])

    @staticmethod
    def get_samples(data, idxs):
        return data.flatten()[idxs].view(*data.shape[:2], -1)
        
    def forward(self, preds, targs):
        losses = self.bce(preds, targs, reduction='none')
        idxs = self.get_idxs(losses, targs)
        preds = self.get_samples(preds, idxs)
        targs = self.get_samples(targs, idxs)
        losses = losses.flatten()[idxs]
        return self.dice(preds, targs) + losses.mean()
