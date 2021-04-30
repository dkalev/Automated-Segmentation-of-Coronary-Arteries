import os
import json
import nrrd
import shutil
import zipfile
import numpy as np
from tqdm import tqdm
from pathlib import Path
from skimage.transform import resize
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from .helpers import get_patch_padding, vol2patches


class DatasetBuilder():
    def __init__(self, logger, *args, num_workers=6, **kwargs):
        self.logger = logger
        self.num_workers = num_workers
        super().__init__(*args, **kwargs)

    def is_valid(self):
        if not Path(self.data_dir).is_dir(): return False
        if not 'dataset.json' in os.listdir(self.data_dir): return False
        if not all([ dir in os.listdir(self.data_dir) for dir in ['train', 'valid']]): return False
        if not all([ dir in os.listdir(Path(self.data_dir, 'train')) for dir in ['vols', 'masks']]): return False
        if not all([ dir in os.listdir(Path(self.data_dir, 'valid')) for dir in ['vols', 'masks']]): return False

        try:
            with open(Path(self.data_dir, 'dataset.json'), 'r') as f:
                meta = json.load(f)
        except:
            return False

        if not set(meta.keys()) == set(['data_clip_range', 'normalize', 'patch_size', 'stride', 'vol_meta']): return False
        if not set(meta['vol_meta'].keys()) == set([str(x) for x in range(40)]): return False
        if not all([set(vol_meta.keys()) == set(['shape_orig', 'shape_cropped', 'shape_resized', 'split', 'orig_spacing', 'shape_patched', 'n_patches', 'contains_arteries', 'padding']) 
            for vol_meta in meta['vol_meta'].values()
        ]): return False

        return True

    def extract_files(self, subdirs):
        # clean up previous data
        for subdir in subdirs:
            if Path(self.data_dir, subdir).is_dir():
                shutil.rmtree(Path(self.data_dir, subdir))

        with zipfile.ZipFile(self.sourcepath, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        data_dir = Path(self.data_dir, 'ASOCA2020Data') # FIXME
        for folder in os.listdir(data_dir):
            shutil.move(str(Path(data_dir, folder)), self.data_dir)
        os.rmdir(data_dir)

    @staticmethod
    def get_resampled_shape(volume, header):
        spacing = np.diagonal(header['space directions'])[::-1]
        target_spacing = np.array([0.625, 0.3964845058, 0.3964845058 ])
        return ((spacing / target_spacing) * volume.shape).round().astype(np.int64), spacing
    
    def get_crop_mask(self, data):
        nonzero = np.argwhere(data)
        top_left, bottom_right = np.min(nonzero, axis=0), np.max(nonzero, axis=0)
        padding = self.patch_size - (bottom_right - top_left)
        padding = np.maximum(padding, 0) # discard negative values which will crop instead
        offset_left = padding // 2
        offset_right = padding - offset_left

        top_left -= offset_left
        bottom_right += offset_right
        
        return (
            slice(top_left[0], bottom_right[0]+1),
            slice(top_left[1], bottom_right[1]+1),
            slice(top_left[2], bottom_right[2]+1),
        )
        
    def normalize_data(self, data):
        if self.data_clip_range is not None:
            lb, ub = self.data_clip_range
            mask = (data > lb) & (data < ub) 
            data = np.clip(data, lb, ub)
            data = data[mask]

        return (data - data.mean()) / data.std()

    def preprocess(self, params):
        vol_id, volume_path, mask_path, data_dir, split = params

        mask, header = nrrd.read(mask_path, index_order='C')

        padding = get_patch_padding(mask.shape, self.patch_size, self.stride)

        if self.crop_empty:
            crop_mask = self.get_crop_mask(mask)
            mask = mask[crop_mask]

        if self.resample_vols:
            new_shape, spacing = self.get_resampled_shape(mask, header)
            dtype = mask.dtype
            mask = resize(mask.astype(float), new_shape, order=0, mode='constant', cval=0, clip=True, anti_aliasing=False).astype(dtype)

        mask_patches, _ = vol2patches(mask, self.patch_size, self.stride, padding)

        contains_arteries = mask_patches.sum(dim=(1,2,3)) > 1

        np.save(Path(data_dir, 'masks', f'{vol_id}.npy'), mask_patches)
        del mask, mask_patches

        volume, _ = nrrd.read(volume_path, index_order='C')

        shape_orig = volume.shape
        shape_cropped = shape_orig

        if self.crop_empty:
            volume = volume[crop_mask]
            shape_cropped = volume.shape

        if self.resample_vols:
            volume = resize(volume, new_shape, order=1, preserve_range=True)

        volume_patches, patched_shape = vol2patches(volume, self.patch_size, self.stride, padding, pad_value=-1000) # -1000 corresponds to air in HU units

        if self.normalize:
            volume_patches = self.normalize_data(volume_patches)

        n_patches = volume_patches.shape[0]

        np.save(Path(data_dir, 'vols', f'{vol_id}.npy'), volume_patches)
        del volume, volume_patches

        return ( vol_id, {
                'shape_orig': shape_orig,
                'shape_cropped': shape_cropped,
                'shape_resized': new_shape.tolist(),
                'split': split,
                'orig_spacing': spacing.tolist(),
                'shape_patched': patched_shape,
                'n_patches': n_patches,
                'contains_arteries': contains_arteries.tolist(),
                'padding': padding
                })

    def build_dataset(self, volume_path, mask_path):
        meta = OrderedDict({
            'patch_size': [ int(x) for x in self.patch_size ],
            'stride': [ int(x) for x in self.stride],
            'normalize': self.normalize,
            'data_clip_range': self.data_clip_range,
        })

        for split in ['train', 'valid']:
            os.makedirs(Path(self.data_dir, split), exist_ok=True)
            for part in ['vols', 'masks']:
                os.makedirs(Path(self.data_dir, split, part), exist_ok=True)
    
        def get_folderpath(file_id, data_dir):
            split = 'train' if file_id < 32 else 'valid'
            return Path(data_dir, split), split

        paths = [ ( file_id,
                    Path(volume_path, f'{file_id}.nrrd'),
                    Path(mask_path, f'{file_id}.nrrd'),
                    *get_folderpath(file_id, self.data_dir)) for file_id in range(40) ]

        with ProcessPoolExecutor(max_workers=self.num_workers) as exec:
            vol_meta = list(tqdm(
                exec.map(self.preprocess, paths),
                total=len(paths)))
        
        meta['vol_meta'] = { m[0]: m[1] for m in vol_meta }

        with open(Path(self.data_dir, 'dataset.json'), 'w') as f:
            json.dump(meta, f, indent=4)
