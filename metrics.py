import numpy as np
import torch
from scipy.spatial import cKDTree
from torchmetrics.functional import auroc

def dice_score(pred, targ, eps=1e-10):
    pred, targ = pred.clone(), targ.clone()
    targ = targ.float()
    sum_dims = list(range(2, len(pred.shape)))
    intersection = torch.sum(pred * targ, dim=sum_dims)
    union = torch.sum(pred, dim=sum_dims) + torch.sum(targ, dim=sum_dims)
    return torch.mean((2. * intersection + eps) / ( union + eps ))

def hausdorff_95(preds, targs, spacing):
    # Code from https://github.com/Ramtingh/ASOCA_MICCAI2020_Evaluation/blob/master/evaluation.py
    if isinstance(preds, torch.Tensor): preds = preds.numpy()
    if np.issubdtype(preds.dtype, np.integer): preds = (preds > 0.5)
    if not isinstance(spacing, np.ndarray): spacing = np.array(spacing)
    preds = preds.astype(np.uint8)
    targs = targs.astype(np.uint8)

    preds_coords = spacing * np.array(np.where(preds), dtype=np.uint8).T
    targs_coords = spacing * np.array(np.where(targs), dtype=np.uint8).T

    preds_kdtree = cKDTree(preds_coords)
    targs_kdtree = cKDTree(targs_coords)

    pt_dist,_ = preds_kdtree.query(targs_coords)
    tp_dist,_ = targs_kdtree.query(preds_coords)

    return max(np.quantile(pt_dist,0.95), np.quantile(tp_dist,0.95))

def roc_auc(preds, targs):
    targs_numpy = targs.cpu().flatten().numpy().astype(int)
    if targs_numpy.sum() > 0:
        preds_numpy = preds.cpu().flatten()
        return auroc(targs_numpy, preds_numpy)
    return 0.5
