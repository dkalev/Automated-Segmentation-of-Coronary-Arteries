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
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor
from .helpers import get_patch_padding, vol2patches


class DatasetBuilder():
    def __init__(self, logger, *args,
                num_workers=6,
                valid_idxs=None,
                normalize=False,
                data_clip_range=(0, 400),
                rebuild=False,
                resample_vols=True,
                crop_empty=False,
                norm_type='global',
                sourcepath='ASOCA2020Data.zip', **kwargs):

        self.logger = logger
        self.num_workers = num_workers
        self.valid_idxs = valid_idxs or [1, 9, 13, 19, 22, 28, 38, 39]
        self.train_idxs = [ idx for idx in range(40) if idx not in self.valid_idxs ]

        if data_clip_range == 'percentile':
            self.data_clip_range = data_clip_range
        elif data_clip_range == 'None':
            self.data_clip_range = None
        else:
            self.data_clip_range = list(data_clip_range)

        self.normalize = normalize
        self.rebuild = rebuild
        self.resample_vols = resample_vols
        self.crop_empty = crop_empty
        if norm_type not in ['vol', 'vol_fg', 'global']:
            raise ValueError(f'Unsupported normalization type: {norm_type}')
        else:
            self.norm_type = norm_type
        self.sourcepath = sourcepath

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

        if not set(meta.keys()) == set([
            'data_clip_range',
            'normalize',
            'patch_size',
            'stride',
            'crop_empty',
            'resample_vols',
            'vol_meta'
            ]): return False
        if not set(meta['vol_meta'].keys()) == set([str(x) for x in range(40)]): return False
        if not all([set(vol_meta.keys()) == set(['shape_orig', 'shape_cropped', 'shape_resampled', 'split', 'orig_spacing', 'shape_patched', 'n_patches', 'foreground_ratio', 'padding'])
            for vol_meta in meta['vol_meta'].values()
        ]): return False

        return True

    def is_config_updated(self):
        with open(Path(self.data_dir, 'dataset.json'), 'r') as f:
            meta = json.load(f)
        return not all([ getattr(self, key) == meta[key] for key in meta if hasattr(self, key)])

    def extract_files(self, subdirs):
        # clean up previous data
        for subdir in subdirs:
            if Path(self.data_dir, subdir).is_dir():
                shutil.rmtree(Path(self.data_dir, subdir))

        with zipfile.ZipFile(self.sourcepath, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        data_dir = Path(self.data_dir, 'temp')
        os.makedirs(data_dir, exist_ok=True)
        for folder in os.listdir(data_dir):
            shutil.move(str(Path(data_dir, folder)), self.data_dir)
        os.rmdir(data_dir)

    @staticmethod
    def get_resampled_shape(volume, spacing):
        target_spacing = np.array([0.5, 0.5, 0.5])
        return ((spacing / target_spacing) * volume.shape).round().astype(np.int64)

    def get_crop_bbox(self, data):
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

    def get_clip_bounds(self, data, mask):
        if self.data_clip_range == 'percentile':
            return self.stats['percentile_00_5'], self.stats['percentile_99_5']
        else:
            return self.data_clip_range

    def normalize_data(self, data, mask):
        if self.data_clip_range is not None:
            lb, ub = self.get_clip_bounds(data, mask)
            data = np.clip(data, lb, ub)

        if self.norm_type == 'vol_fg':
            mean = data[mask==1].mean()
            std  = data[mask==1].std()
        elif self.norm_type == 'vol':
            mean = data.mean()
            std  = data.std()
        elif self.norm_type == 'global':
            mean = self.stats['mean']
            std  = self.stats['std']

        return (data - mean) / std

    def preprocess(self, params):
        vol_id, volume_path, mask_path, split = params
        data_dir = Path(self.data_dir, split)

        mask, header = nrrd.read(mask_path, index_order='C')

        padding = get_patch_padding(mask.shape, self.patch_size, self.stride)

        if split == 'train' and self.crop_empty:
            crop_mask = self.get_crop_bbox(mask)
            mask = mask[crop_mask]

        spacing = np.diagonal(header['space directions'])[::-1]
        if self.resample_vols:
            shape_resampled = self.get_resampled_shape(mask, spacing)
            dtype = mask.dtype
            mask = resize(mask.astype(float), shape_resampled, order=0, mode='constant', cval=0, clip=True, anti_aliasing=False).astype(dtype)

        mask_patches, _ = vol2patches(mask, self.patch_size, self.stride, padding)

        foreground_ratio = mask_patches.mean(dim=(1,2,3))

        np.save(Path(data_dir, 'masks', f'{vol_id}.npy'), mask_patches)
        del mask

        volume, _ = nrrd.read(volume_path, index_order='C')

        shape_orig = volume.shape
        shape_cropped = shape_orig

        if split == 'train' and self.crop_empty:
            volume = volume[crop_mask]
            shape_cropped = volume.shape

        if self.resample_vols:
            volume = resize(volume, shape_resampled, order=1, preserve_range=True)

        volume_patches, patched_shape = vol2patches(volume, self.patch_size, self.stride, padding, pad_value=-1000) # -1000 corresponds to air in HU units

        if self.normalize:
            volume_patches = self.normalize_data(volume_patches, mask_patches)

        n_patches = volume_patches.shape[0]

        np.save(Path(data_dir, 'vols', f'{vol_id}.npy'), volume_patches)
        del volume, volume_patches

        if not self.resample_vols:
            shape_resampled = shape_cropped
        else:
            shape_resampled = shape_resampled.tolist()

        return ( vol_id, {
                'shape_orig': shape_orig,
                'shape_cropped': shape_cropped,
                'shape_resampled': shape_resampled,
                'split': split,
                'orig_spacing': spacing.tolist(),
                'shape_patched': patched_shape,
                'n_patches': n_patches,
                'foreground_ratio': foreground_ratio.tolist(),
                'padding': padding
                })

    def _get_foreground(self, paths):
        vol_path, mask_path = paths
        mask, _ = nrrd.read(mask_path, index_order='C')
        vol, _ = nrrd.read(vol_path, index_order='C')
        return vol[mask==1].flatten()[::10]

    def build_dataset(self, volume_path, mask_path):
        meta = OrderedDict({
            'patch_size': [ int(x) for x in self.patch_size ],
            'stride': [ int(x) for x in self.stride],
            'normalize': self.normalize,
            'data_clip_range': self.data_clip_range,
            'resample_vols': self.resample_vols,
            'crop_empty': self.crop_empty,
        })

        for split in ['train', 'valid']:
            os.makedirs(Path(self.data_dir, split), exist_ok=True)
            for part in ['vols', 'masks']:
                os.makedirs(Path(self.data_dir, split, part), exist_ok=True)

        paths = [ ( file_id,
                    Path(volume_path, f'{file_id}.nrrd'),
                    Path(mask_path, f'{file_id}.nrrd'),
                    'train' if file_id not in self.valid_idxs else 'valid'
                    ) for file_id in range(40) ]


        with ProcessPoolExecutor(max_workers=self.num_workers, mp_context=get_context('spawn')) as exec:
           train_paths = [ (p[1], p[2]) for p in paths if p[-1] == 'train' ]
           fg_voxels = list(tqdm(exec.map(self._get_foreground, train_paths), total=len(train_paths)))
           fg_voxels = np.concatenate(fg_voxels)
           self.stats = {
                    'mean': np.mean(fg_voxels),
                    'std': np.std(fg_voxels),
                    'percentile_00_5': np.percentile(fg_voxels, 0.5),
                    'percentile_99_5': np.percentile(fg_voxels, 99.5),
                   }
           vol_meta = list(tqdm(
                exec.map(self.preprocess, paths),
                total=len(paths)))

        meta['vol_meta'] = { m[0]: m[1] for m in vol_meta }

        with open(Path(self.data_dir, 'dataset.json'), 'w') as f:
            json.dump(meta, f, indent=4)

