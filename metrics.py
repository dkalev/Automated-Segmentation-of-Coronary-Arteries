import numpy as np
import torch
from scipy.spatial import cKDTree
from torchmetrics.functional import auroc

def dice_score(pred, targ, eps=1e-10):
    targ = targ.float()
    sum_dims = list(range(2, len(pred.shape)))
    intersection = torch.sum(pred * targ, dim=sum_dims)
    union = torch.sum(pred, dim=sum_dims) + torch.sum(targ, dim=sum_dims)
    return torch.mean((2. * intersection + eps) / ( union + eps ))

def hausdorff_95(pred, targ, spacing):
    # Code from https://github.com/Ramtingh/ASOCA_MICCAI2020_Evaluation/blob/master/evaluation.py
    pred_points = spacing * np.array(np.where(pred), dtype=np.uint16).T
    pred_kdtree = cKDTree(pred_points)
    
    targ_points = spacing * np.array(np.where(targ), dtype=np.uint16).T
    targ_kdtree = cKDTree(targ_points)
    
    pred_targ_dist,_ = pred_kdtree.query(targ_points)
    dist_pred_dist,_ = targ_kdtree.query(pred_points)
    return max(np.quantile(pred_targ_dist,0.95), np.quantile(dist_pred_dist,0.95))

def roc_auc(preds, targs):
    targs_numpy = targs.cpu().flatten().numpy().astype(int)
    if targs_numpy.sum() > 0:
        preds_numpy = preds.cpu().flatten()
        return auroc(targs_numpy, preds_numpy)
    return 0.5