import os
import nrrd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Union, Set, Dict
from tqdm import tqdm


def get_pos_idxs(x:np.ndarray) -> np.ndarray:
    return np.stack(np.nonzero(x)).T

def get_foreground_idxs(targs:np.ndarray) -> np.ndarray:
    return get_pos_idxs(targs)

def get_background_idxs(targs:np.ndarray, heart_mask:np.ndarray) -> np.ndarray:
    background = (1-targs) * (1-heart_mask)
    return get_pos_idxs(1-heart_mask)

def filter_invalid_idxs(idxs:np.ndarray, patch_size:int, vol_shape: Tuple[int,int,int]) -> np.ndarray:
    idxs = idxs.copy()
    fits_left  = np.all((idxs - patch_size / 2) >= 0, axis=1)
    fits_right = np.all((idxs + patch_size / 2 - vol_shape) <= 0, axis=1)
    return idxs[fits_left&fits_right]

def filter_containing_foregroung(targs:np.ndarray, idxs:np.ndarray, patch_size:int, n_samples:int=None, thresh:float=0.0) -> np.ndarray:
    n_samples = n_samples or len(idxs)
    
    res = set()
    while len(res) < n_samples:
        coords = idxs[np.random.choice(len(idxs), n_samples - len(res), replace=False)]
        for coord in coords:
            bbox = get_patch_bbox(coord, patch_size)
            patch = targs[bbox]
            if patch.sum() <= thresh:
                res.add(tuple(coord))
    return np.array(list(res))

def get_patch_bbox(center_coords:np.ndarray, patch_size:int) -> Tuple[slice, slice, slice]:
    bbox = np.array([
        center_coords - patch_size / 2,
        center_coords + patch_size / 2]
    )
    x, y, z = bbox.T.astype(int)
    return slice(x[0],x[1]), slice(y[0],y[1]), slice(z[0],z[1]) 

def get_vol_id(vol_path):
    return int(vol_path.name.split('.')[0])

def get_paths(folder:Union[str, Path], sort_fn=lambda x: x) -> List:
    paths = [ Path(folder, path) for path in os.listdir(Path(folder)) if 'npy' in path or 'nrrd' in path]
    paths = sorted(paths, key=sort_fn)
    return paths

def get_vol_paths(vol_dir:Union[str,Path],
                  vol_subdir:str='Train',
                  targ_subdir:str='Train_Masks',
                  heart_mask_subdir='Train_heart_mask') -> List:
    
    vol_paths  = get_paths(Path(vol_dir, vol_subdir), sort_fn=get_vol_id)
    targ_paths = get_paths(Path(vol_dir, targ_subdir), sort_fn=get_vol_id)
    heart_mask_paths = get_paths(Path(vol_dir, heart_mask_subdir), sort_fn=get_vol_id)
    
    vol_ids = [ get_vol_id(path) for path in vol_paths ]
    
    return list(zip(vol_ids, vol_paths, targ_paths, heart_mask_paths))

def sample_hard_coords(targs: np.ndarray, 
                       idxs: np.ndarray, 
                       patch_size:int, 
                       blacklist: Set[List], 
                       n_samples:int,
                       thresh:float=0.001) -> np.ndarray:
    
    res = set()
    while len(res) < n_samples:
        coords = idxs[np.random.choice(len(idxs), n_samples - len(res), replace=False)]
        for coord in coords:
            if tuple(coord) not in blacklist and tuple(coord) not in res:
                bbox = get_patch_bbox(coord, patch_size)
                patch = targs[bbox]
                if patch.sum() <= thresh:
                    res.add(tuple(coord))
    return np.array(list(res))

def get_pos_coords(targs: np.ndarray, patch_size:int, n_samples:int) -> np.ndarray:
    coords = get_foreground_idxs(targs)
    coords = filter_invalid_idxs(coords, patch_size, targs.shape)
    sampled = np.random.choice(len(coords), n_samples, replace=False)
    coords = coords[sampled]
    labels = np.ones((len(coords), 1))
    return np.hstack([coords, labels])

def get_neg_coords(targs:np.ndarray, heart_mask:np.ndarray, patch_size:int, n_samples:int) -> np.ndarray:
    coords = get_background_idxs(targs, heart_mask)
    coords = filter_invalid_idxs(coords, patch_size, targs.shape)
    coords = filter_containing_foregroung(targs, coords, patch_size, n_samples=n_samples)
    labels = np.zeros((len(coords), 1))
    return np.hstack([coords, labels])

def get_vol_hard_mask(vol:np.ndarray, targs:np.ndarray, heart_mask:np.ndarray) -> np.ndarray:
    mask = vol * (1-targs) * (1-heart_mask)
    mask[(
        (mask != 0) & 
        (mask > np.percentile(mask, 0.05)) &
        (mask < np.percentile(mask, 0.95))
    )] = 1
    return mask

def get_hard_neg_coords(vol_hard_mask:np.ndarray,
                        targs:np.ndarray,
                        already_sampled: Set[Tuple[int,int,int]],
                        patch_size:int,
                        n_samples:int):
    
    coords = get_foreground_idxs(vol_hard_mask)
    coords = filter_invalid_idxs(coords, patch_size, vol_hard_mask.shape)
    coords = sample_hard_coords(targs, coords, patch_size, already_sampled, n_samples=n_samples)
    labels = np.zeros((len(coords), 1))
    return np.hstack([coords, labels])

def normalize_vols(vol_paths:List[Union[Path,str]], output_dir:Union[Path, str], stats:Dict):
    os.makedirs(output_dir, exist_ok=True)
    for vol_id, path, _, _ in tqdm(vol_paths):
        vol, _  = nrrd.read(path, index_order='C')
        vol = np.clip(vol, stats['percentile_00_5'], stats['percentile_99_5'])
        vol = (vol - stats['mean']) / stats['std']
        np.save(Path(output_dir, f'{vol_id}.npy'), vol)

def get_patch_coords(vol_paths:List, patch_size:int, n_patches:int=100000):
    assert n_patches % 4 == 0
    res = {}
    n_per_vol = n_patches // len(vol_paths)
    
    for vol_id, vol_path, targ_path, heart_mask_path in tqdm(vol_paths):
        vol           = np.load(vol_path)
        targs, _      = nrrd.read(targ_path, index_order='C')
        heart_mask, _ = nrrd.read(heart_mask_path, index_order='C')          
        
        targs = targs.astype(np.uint8)
        heart_mask = heart_mask.astype(np.uint8)
        
        pos_coords = get_pos_coords(targs, patch_size, n_samples=n_per_vol//2)
        neg_coords = get_neg_coords(targs, heart_mask, patch_size, n_per_vol//4)
        
        already_sampled = set([tuple(x) for x in neg_coords[:,:-1].tolist()])
        vol_hard_mask = get_vol_hard_mask(vol, targs, heart_mask)
        hard_neg_coords = get_hard_neg_coords(vol_hard_mask, targs, already_sampled, patch_size, n_per_vol//4)
        
        coords = np.vstack((pos_coords, neg_coords, hard_neg_coords)).astype(int)
        coords = np.random.permutation(coords)
        res[vol_id] = coords
    return res
