import numpy as np
from scipy.spatial import cKDTree

def dice_score(pred, targ, smooth=1):
    intersection = (pred * targ).sum()
    return (2. * intersection + smooth) / ( pred.sum() + targ.sum() + smooth )

def hausdorff_95(pred, targ, spacing):
    # Code from https://github.com/Ramtingh/ASOCA_MICCAI2020_Evaluation/blob/master/evaluation.py
    pred_points = spacing * np.array(np.where(pred), dtype=np.uint16).T
    pred_kdtree = cKDTree(pred_points)
    
    targ_points = spacing * np.array(np.where(targ), dtype=np.uint16).T
    targ_kdtree = cKDTree(targ_points)
    
    pred_targ_dist,_ = pred_kdtree.query(targ_points)
    dist_pred_dist,_ = targ_kdtree.query(pred_points)
    return max(np.quantile(pred_targ_dist,0.95), np.quantile(dist_pred_dist,0.95))