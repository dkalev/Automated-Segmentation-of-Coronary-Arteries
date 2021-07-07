import torch
import torch.nn as nn


class BaseLoss(nn.Module):
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize

    def __call__(self, preds, targs):
        assert len(preds.shape) > 2, preds.shape
        assert preds.size() == targs.size(), f'preds: {preds.size()}, targs: {targs.size()}'

        if self.normalize:
            preds = torch.sigmoid(preds)

        targs = targs.type(preds.dtype)
        return self.forward(preds, targs)


class BCEWrappedLoss(BaseLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.crit = nn.BCEWithLogitsLoss()

    def forward(self, preds, targs):
        return self.crit(preds, targs)


class DiceLoss(BaseLoss):
    def __init__(self, *args, generalized=False, eps=1e-10, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps
        self.generalized = generalized

    def forward(self, preds, targs):
        sum_dims = list(range(2, len(preds.shape)))
        intersection = torch.sum(preds * targs, dim=sum_dims)
        denom =  preds.sum(dim=sum_dims) + targs.sum(dim=sum_dims)
        if self.generalized:
            w =  1 / ((targs.sum(dim=sum_dims) + self.eps) ** 2)
            intersection = w * intersection
            denom = w * denom
        loss = 1 - (2 * intersection + self.eps) / (denom + self.eps)
        return loss.mean()


class CombinedLoss(BaseLoss):
    def __init__(self, dice_kwargs=None, normalize=True):
        super().__init__(normalize=False) # normalize the individual components separately

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
        ratio = (torch.sum(targs == 0) / torch.sum(targs == 1)).type_as(targs)
        weight = torch.ones_like(targs, dtype=targs.dtype)
        weight[targs==1] = ratio
        return weight

    def forward(self, preds, targs):
        weight = self.get_weight(targs) if self.weighted_bce else None
        return self.dice(preds, targs) + self.bce(preds, targs, weight=weight)


class DiceBCE_OHNMLoss(CombinedLoss):
    """ Online Hard Negative Mining """
    def __init__(self, *args, ohnm_ratio=30, default_neg_perc=0.1, **kwargs):
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
        _, hns_idxs = losses.clone().masked_fill(targs==1,0).flatten().topk(n_hns)
        pos_idxs = torch.nonzero(targs.flatten()==1)
        idxs = torch.cat([hns_idxs.flatten(), pos_idxs.flatten()])
        return idxs

    @staticmethod
    def get_samples(data, idxs):
        return data.flatten()[idxs].view(*data.shape[:2], -1)

    def forward(self, preds, targs):
        losses = self.bce(preds.clone(), targs, reduction='none')
        idxs = self.get_idxs(losses, targs)
        preds = self.get_samples(preds, idxs)
        targs = self.get_samples(targs, idxs)
        losses = losses.flatten()[idxs]
        return self.dice(preds, targs) + losses.mean()

