from abc import abstractmethod
import torch
import torch.nn as nn


class BaseLoss(nn.Module):
    """ Base class for all segmentation losses. Ensures that preds and targs have 
        the correct shapes and sizes and normalizes the preds if necessary
    """
    def __init__(self, normalize:bool=True):
        """
        Args:
            normalize (bool, optional): Applies sigmoid to predictions on forward. Defaults to True.
        """
        super().__init__()
        self.normalize = normalize
    
    @abstractmethod
    def forward(preds: torch.Tensor, targs: torch.Tensor) -> torch.Tensor: pass

    def __call__(self, preds:torch.Tensor, targs:torch.Tensor) -> torch.Tensor:
        """ Wraps around the implementations of the concrete losses to ensure preds and targs
            shapes are correct and applies a sigmoid to normalize preds if required

        Args:
            preds (torch.Tensor): binary predictions
            targs (torch.Tensor): binary targets

        Returns:
            torch.Tensor: finall loss
        """
        assert len(preds.shape) > 2, preds.shape
        assert preds.size() == targs.size(), f'preds: {preds.size()}, targs: {targs.size()}'

        if self.normalize: preds = torch.sigmoid(preds)

        targs = targs.type(preds.dtype)
        return self.forward(preds, targs)


class BCEWrappedLoss(BaseLoss):
    """ Wrapper around torch.nn.BCEWithLogitsLoss, ensuring it shares the same
        interface as the rest of the losses
    """
    def __init__(self, *args, **kwargs):
        super().__init__(normalize=False)
        self.crit = nn.BCEWithLogitsLoss(*args, **kwargs)

    def forward(self, preds:torch.Tensor, targs:torch.Tensor) -> torch.Tensor:
        return self.crit(preds, targs)


class DiceLoss(BaseLoss):
    """ Dice Loss implementation defined as 1 - dice score """
    def __init__(self, *args, generalized:bool=False, eps:float=1e-10, **kwargs):
        """
        Args:
            generalized (bool, optional): If True uses generalized dice loss.
            See https://arxiv.org/abs/1707.03237. Defaults to False.
            eps (float, optional): [description]. Defaults to 1e-10.
        """
        super().__init__(*args, **kwargs)
        self.eps = eps
        self.generalized = generalized

    def forward(self, preds:torch.Tensor, targs:torch.Tensor) -> torch.Tensor:
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
    """ Base class for losses combining BCE and Dice loss """
    def __init__(self, normalize:bool=True, **dice_kwargs):
        super().__init__(normalize=False) # normalize the individual components separately

        if normalize:
            self.bce = nn.functional.binary_cross_entropy_with_logits
        else:
            self.bce = nn.functional.binary_cross_entropy
        self.dice = DiceLoss(normalize=normalize, **dice_kwargs)


class DiceBCELoss(CombinedLoss):
    """ Combines Binary Cross Entropy and Dice loss """
    def __init__(self, weighted_bce:bool=True, **kwargs):
        """
        Args:
            weighted_bce (bool, optional): If set to true computes weights for the
            BCE loss based on the ratio foreground / background voxels within a batch
            to mitigate class imbalance issues. Defaults to True.
        """
        super().__init__(**kwargs)
        self.weighted_bce = weighted_bce

    @staticmethod
    def get_weight(targs:torch.Tensor) -> torch.Tensor:
        """ Computes weights for the BCE loss based on the ratio foreground / background
            voxels within a batch to mitigate class imbalance issues.

        Args:
            targs (torch.Tensor): binary targets

        Returns:
            torch.Tensor: weights for each voxels to be used in the BCE loss
        """
        ratio = (torch.sum(targs == 0) / torch.sum(targs == 1)).type_as(targs)
        weight = torch.ones_like(targs, dtype=targs.dtype)
        weight[targs==1] = ratio
        return weight

    def forward(self, preds:torch.Tensor, targs:torch.Tensor) -> torch.Tensor:
        weight = self.get_weight(targs) if self.weighted_bce else None
        return self.dice(preds, targs) + self.bce(preds, targs, weight=weight)


class DiceBCE_OHNMLoss(CombinedLoss):
    """ Combined Dice and BCE loss with Online Hard Negative Mining """
    def __init__(self, ohnm_ratio:int=30, default_neg_perc:float=0.1, **kwargs):
        """
        Args:
            ohnm_ratio (int, optional): Ratio between positive and negative samples.
            For each positive sample (voxel) ohnm_ratio negative ones will be used.
            Defaults to 30.
            default_neg_perc (float, optional): In batches with no positive samples
            what percentage of the negative samples to be used. Defaults to 0.1.
        """
        super().__init__(**kwargs)
        self.ohnm_ratio = ohnm_ratio
        self.default_neg_perc = default_neg_perc

    def get_num_hns(self, targs:torch.Tensor) -> int:
        """ Computes the number of hard negative samples to use given a batch of 
            targets

        Args:
            targs (torch.Tensor): binary targets

        Returns:
            int: number of hard negative samples to use in final loss
        """
        n_pos = int(targs.sum().item())
        n_neg = targs.numel() - n_pos
        if n_pos == 0:
            return int(self.default_neg_perc * n_neg)
        return min(n_pos * self.ohnm_ratio, n_neg)

    def get_idxs(self, losses:torch.Tensor, targs:torch.Tensor) -> torch.Tensor:
        """ Get indexes of voxels to be used in final loss. This includes all positive (foreground)
            voxels and the top n negative (background) voxels with highest losses

        Args:
            losses (torch.Tensor): losses per individual voxels (from the BCE loss)
            targs (torch.Tensor): binary targets

        Returns:
            torch.Tensor: the indeces of all voxels to be used in the final loss
        """
        n_hns = self.get_num_hns(targs)
        _, hns_idxs = losses.clone().masked_fill(targs==1,0).flatten().topk(n_hns)
        pos_idxs = torch.nonzero(targs.flatten()==1)
        idxs = torch.cat([hns_idxs.flatten(), pos_idxs.flatten()])
        return idxs

    @staticmethod
    def get_samples(data:torch.Tensor, idxs:torch.Tensor) -> torch.Tensor:
        return data.flatten()[idxs].view(*data.shape[:2], -1)

    def forward(self, preds:torch.Tensor, targs:torch.Tensor) -> torch.Tensor:
        losses = self.bce(preds.clone(), targs, reduction='none')
        idxs = self.get_idxs(losses, targs)
        preds = self.get_samples(preds, idxs)
        targs = self.get_samples(targs, idxs)
        losses = losses.flatten()[idxs]
        return self.dice(preds, targs) + losses.mean()

