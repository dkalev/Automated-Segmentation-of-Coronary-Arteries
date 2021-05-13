import h5py
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
import torch
import sys
import numpy as np
import h5py
import json
sys.path.append('..')
from data_utils.helpers import vol2patches, get_patch_padding

def reshape(data, patch_shape=(16,16,16)):
    return data.reshape(-1, *patch_shape)

def build_3d_mnist(input_path, output_path, padding=None):
    with h5py.File(input_path, 'r') as dataset:
        x_train = dataset["X_train"][:]
        x_test = dataset["X_test"][:]
        
        x_train = reshape(x_train)
        x_test = reshape(x_test)

        if padding:
            x_train = np.pad(x_train[:1000], padding)
            x_test = np.pad(x_test[:200], padding)
        
        y_train = (x_train > 0).astype(int)
        y_test  = (x_test  > 0).astype(int)

    with h5py.File(output_path, 'w') as f:
        train_gr = f.create_group('train')
        train_vol = train_gr.create_dataset('vols', x_train.shape)
        train_masks = train_gr.create_dataset('masks', x_train.shape, dtype=int)
        train_vol[:] = x_train
        train_masks[:] = y_train
        
        test_gr = f.create_group('test')
        test_vol = test_gr.create_dataset('vols', x_test.shape)
        test_masks = test_gr.create_dataset('masks', x_test.shape, dtype=int)
        test_vol[:] = x_test
        test_masks[:] = y_test

def get_bbox(shape, dsize=16):
    x1 = np.random.randint(shape[0]-dsize)
    y1 = np.random.randint(shape[1]-dsize)
    z1 = np.random.randint(shape[2]-dsize)
    return slice(x1,x1+dsize), slice(y1,y1+dsize), slice(z1,z1+dsize)

def build_3d_sparse(output_dir):
    os.makedirs(Path(output_dir, 'train/vols'), exist_ok=True)
    os.makedirs(Path(output_dir, 'train/masks'), exist_ok=True)
    os.makedirs(Path(output_dir, 'valid/vols'), exist_ok=True)
    os.makedirs(Path(output_dir, 'valid/masks'), exist_ok=True)
    meta = {
        'patch_size': [ 128, 128, 128 ],
        'stride': [ 108, 108, 108 ],
        'normalize': False,
        'data_clip_range': None,
        'resample_vols': False,
        'crop_empty': False,
        'vol_meta': {}
    }

    with h5py.File('../dataset/3d_mnist/full_dataset_vectors.h5', 'r') as dataset:
        mnist = dataset["X_train"][:]
        mnist = mnist.reshape(-1,16,16,16)

    valid_ids = np.random.choice(range(40), size=8, replace=False)
    train_ids = [ i for i in range(40) if i not in valid_ids]
    assert set(train_ids) | set(valid_ids) == set(range(40))

    def get_split(fid, train_ids): return 'train' if fid in train_ids else 'valid'

    shape_orig = (214, 512, 512)
    for fid in tqdm(range(40)):
        
        vol = np.empty(shape_orig)
        
        idxs = np.random.choice(range(len(mnist)), size=15)
        for idx in idxs:
            vol[get_bbox(shape_orig)] = mnist[idx]
            
        mask = (vol > 0).astype(int)
            
        vol = torch.tensor(vol).float()
        mask = torch.tensor(mask).float()

        
        padding = get_patch_padding(vol.shape, meta['patch_size'], meta['stride'])
        
        vol_patches, patched_shape = vol2patches(vol, meta['patch_size'], meta['stride'], padding)
        mask_patches, _ = vol2patches(mask, meta['patch_size'], meta['stride'], padding)
        
        foreground_ratio = mask_patches.mean(dim=(1,2,3))
        
        split = get_split(fid, train_ids)
        np.save(Path(output_dir, split, 'vols', f'{fid}.npy'), vol_patches.numpy())
        np.save(Path(output_dir, split, 'masks', f'{fid}.npy'),mask_patches.numpy())
        
        meta['vol_meta'][fid] = {
            'shape_orig': shape_orig,
            'shape_cropped': shape_orig,
            'shape_resized': shape_orig,
            'split': split,
            'orig_spacing': [],
            'shape_patched': patched_shape,
            'n_patches': len(vol_patches),
            'foreground_ratio': foreground_ratio.tolist(),
            'padding': padding}
    with open(Path(output_dir, 'dataset.json'), 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    build_3d_mnist( input_path='../dataset/3d_mnist/full_dataset_vectors.h5',
                    output_path='../dataset/3d_mnist.h5')

    build_3d_mnist( input_path='../dataset/3d_mnist/full_dataset_vectors.h5',
                    output_path='../dataset/3d_mnist_64.h5',
                    padding=((0,0),(24,24),(24,24),(24,24)))