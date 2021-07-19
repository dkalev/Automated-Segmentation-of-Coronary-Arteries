from typing import Iterable, Union
import numpy as np
import torch
from scipy.spatial import cKDTree


def dice_score(pred:torch.Tensor, targ:torch.Tensor, eps:float=1e-10) -> torch.Tensor:
    """ Computes the average dice score within an batch of binary predictions and targets

    Args:
        pred (torch.Tensor): tensor of shape [bs,1,...spatial_dims] containing the binary predictions
        targ (torch.Tensor): tensor of shape [bs,1,...spatial_dims] containing the binary targets
        eps (float, optional): smoothing factor added both to numerator and denominator in final 
            equation to ensure stable results. Defaults to 1e-10.

    Returns:
        torch.Tensor: The average score across the samples in the batch
    """
    pred, targ = pred.clone(), targ.clone()
    targ = targ.float()
    # if a batch patches [bs,1,w,h,d] sum over spatial dimensions and average over batch
    # otherwise sum over all dimensions e.g. volume of shape [w,h,d]
    sum_dims = list(range(2, len(pred.shape))) if len(pred.shape) == 5 else []
    intersection = torch.sum(pred * targ, dim=sum_dims)
    union = torch.sum(pred, dim=sum_dims) + torch.sum(targ, dim=sum_dims)
    return torch.mean((2. * intersection + eps) / ( union + eps ))

def hausdorff_95(preds: Union[torch.Tensor, np.ndarray],
                 targs: Union[torch.Tensor, np.ndarray],
                 spacing: Union[Iterable, np.ndarray]) -> float:
    """ Code from https://github.com/Ramtingh/ASOCA_MICCAI2020_Evaluation/blob/master/evaluation.py
        Computes the Hausdorff 95 percentile distance between two 3D volumes (in this case binary
        predictions and targets).

    Args:
        preds (Union[torch.Tensor, np.ndarray]): binary predictions
        targs (Union[torch.Tensor, np.ndarray]): binary targets
        spacing (Union[Iterable, np.ndarray]): the spacing between the centers of individual voxels
            along each of the axis

    Returns:
        float: the Hausdorff 95 percentile distance between predictions and targets
    """

    if isinstance(preds, torch.Tensor): preds = preds.numpy()
    if not isinstance(spacing, np.ndarray): spacing = np.array(spacing)

    if np.issubdtype(preds.dtype, np.integer): preds = (preds > 0.5)
    preds = preds.astype(np.uint8)
    targs = targs.astype(np.uint8)

    preds_coords = spacing * np.array(np.where(preds), dtype=np.uint8).T
    targs_coords = spacing * np.array(np.where(targs), dtype=np.uint8).T

    preds_kdtree = cKDTree(preds_coords)
    targs_kdtree = cKDTree(targs_coords)

    pt_dist,_ = preds_kdtree.query(targs_coords)
    tp_dist,_ = targs_kdtree.query(preds_coords)

    return max(np.quantile(pt_dist,0.95), np.quantile(tp_dist,0.95))
