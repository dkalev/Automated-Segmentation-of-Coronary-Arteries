import os
import re
import ast
import h5py
import json
import nrrd
import torch
import shutil
import zipfile
import numpy as np
from tqdm import tqdm
from pathlib import Path
from skimage.transform import resize
from collections import defaultdict, OrderedDict
from concurrent.futures import ProcessPoolExecutor
from .helpers import get_n_patches, get_patch_padding, vol2patches, patches2vol


class DatasetBuilder():
    def __init__(self, logger, *args, **kwargs):
        self.logger = logger
        super().__init__(*args, **kwargs)

    @staticmethod
    def is_valid(f):
        if not all([attr in f.attrs for attr in ['patch_size', 'stride', 'vol_shapes', 'normalize']]): return False
        if not all( key in f.keys() for key in ['train', 'valid']): return False
        if not all( key in f['train'].keys() for key in ['volumes', 'masks']): return False
        if not all( key in f['valid'].keys() for key in ['volumes', 'masks']): return False
        return True

    def extract_files(self, subdirs):
        # clean up previous data
        for subdir in subdirs:
            if Path(self.output_dir, subdir).is_dir():
                shutil.rmtree(Path(self.output_dir, subdir))

        with zipfile.ZipFile(self.sourcepath, 'r') as zip_ref:
            zip_ref.extractall(self.output_dir)
        data_dir = Path(self.output_dir, 'ASOCA2020Data') # FIXME
        for folder in os.listdir(data_dir):
            shutil.move(str(Path(data_dir, folder)), self.output_dir)
        os.rmdir(data_dir)

    def verify_dataset(self):
        """ Checks if the HDF5 dataset exists, can be used and has the
            required patch size and stride.

            Returns:
                One of ['build', 'resample', 'use'] indicating whether the
            dataset has to be rebuild from scratch, resampled or is ready to use.
        """
        if not self.datapath.is_file():
            self.logger.info(f'HDF5 dataset not found at {self.datapath}')
            return 'build'

        try:
            with h5py.File(self.datapath, 'r') as f:
                if not self.is_valid(f):
                    raise Exception
                elif all([ np.all(self.patch_size == f.attrs['patch_size']),
                           np.all(self.stride == f.attrs['stride']),
                           self.normalize == f.attrs['normalize'] ]):
                    self.logger.info(f'Using HDF5 dataset found at: {self.datapath}')
                    return 'use'
                else:
                    self.logger.info(f'Resampling existing HDF5 dataset with patch size: {self.patch_size}, stride: {self.stride}')
                    return 'resample'
        except Exception:
            self.logger.info(f'Found corrupted HDF5 dataset. Building from scratch.')
            os.remove(self.datapath)
            return 'build'

    def get_total_patches(self, filepaths):
        total = 0
        for filepath in filepaths:
            volume_shape = nrrd.read_header(str(filepath))['sizes'][::-1] # reverse order
            total += get_n_patches(volume_shape, self.patch_size, self.stride)
        return total

    def normalize_ds(self, ds, batch_size=1000):
        axis = None if self.normalize == 'global' else 0
        mean = 0
        std = 0
        self.logger.info('Computing dataset mean and std.')
        for cur in tqdm(range(0, len(ds), batch_size)):
            m = cur * batch_size
            n = batch_size
            batch_mean = ds[cur:cur+batch_size].mean(axis=axis)
            batch_std  = ds[cur:cur+batch_size].std(axis=axis)

            # the std requires the previous mean, so compute before updating mean variable
            std  = m / (m+n) * std**2 + n / (m+n) * batch_std**2 +\
                        m*n / (m+n)**2 * (mean - batch_mean)**2
            std = np.sqrt(std)
            mean = m / (m+n) * mean + n / (m+n) * batch_mean

        self.logger.info('Normalizing dataset')
        for cur in tqdm(range(0, len(ds), batch_size)):
            ds[cur: cur+batch_size] = (ds[cur: cur+batch_size] - mean) / std

    def clip_data(self, data):
        low, high = self.data_clip_range
        data[data < low] = low
        data[data > high] = high
        return data

    def populate_dataset(self, f, data_paths, hdf_group, num_patches):
        ds_size = (num_patches, *self.patch_size)
        volume_ds = hdf_group.create_dataset('volumes', ds_size)
        mask_ds = hdf_group.create_dataset('masks', ds_size)

        cur_start = 0
        shapes = {}
        for volume_path, mask_path in tqdm(list(data_paths)):
            volume, _ = nrrd.read(volume_path, index_order='C')
            padding = get_patch_padding(volume.shape, self.patch_size, self.stride)
            volume_patches, patched_shape = vol2patches(volume, self.patch_size, self.stride, padding, pad_value=-1000) # -1000 corresponds to air in HU units

            cur_end = cur_start + len(volume_patches)

            if self.data_clip_range is not None:
                volume_patches = self.clip_data(volume_patches)
            volume_ds[cur_start:cur_end] = volume_patches

            mask, _ = nrrd.read(mask_path, index_order='C')
            mask_patches, _ = vol2patches(mask, self.patch_size, self.stride, padding)
            mask_ds[cur_start:cur_end] = mask_patches

            vol_id = int(re.findall('(\d+).nrrd', volume_path.name)[0])
            shapes[vol_id] = {
                'shape_orig': volume.shape,
                'shape_patched': patched_shape,
                'start_idx': cur_start,
                'n_patches': volume_patches.shape[0],
                'padding': padding
            }

            cur_start = cur_end

        if self.normalize != 'none': self.normalize_ds(volume_ds)

        return shapes

    def resample_patches(self, f):
        patch_size_prev = f.attrs['patch_size']
        stride_prev = f.attrs['stride']
        shapes_old = ast.literal_eval(f.attrs['vol_shapes'])

        patch_size_new = self.patch_size
        stride_new = self.stride

        vol_meta = defaultdict(dict)
        for split in f:
            padding_new = { vol_id: get_patch_padding(vol_data['shape_orig'], patch_size_new, stride_new)
                                for vol_id, vol_data in shapes_old[split].items()
            }
            padded_vol_shape = (
                np.sum([vol_data['shape_orig'][0] + np.sum(padding_new[vol_id][-2:]) for vol_id, vol_data in shapes_old[split].items()]),
                512, 512)
            # estimate size of new dataset (n_patches x psize x psize x psize)
            n_patches = get_n_patches(padded_vol_shape, patch_size_new, stride_new)
            ds_size = (n_patches, patch_size_new, patch_size_new, patch_size_new)
            for partition in f[split]:
                f.create_dataset('temp', ds_size)
                cur_old = 0
                cur_new = 0
                for vol_id, vol_prev_meta in tqdm(shapes_old[split].items()):
                    batch = f[split][partition][cur_old: cur_old+vol_prev_meta['n_patches']]
                    batch = torch.from_numpy(batch)
                    batch = batch.view(vol_prev_meta['shape_patched'])

                    vol = patches2vol(batch, patch_size_prev, stride_prev, vol_prev_meta['padding'])
                    pad_value = -1000 if partition == 'volume' else 0 # FIXME pad value cannot be -1000 after normalization

                    patches, patched_shape_new = vol2patches(vol, patch_size_new, stride_new, padding_new[vol_id], pad_value=pad_value)
                    n_patches_new = patches.shape[0]

                    f['temp'][cur_new: cur_new+n_patches_new] = patches

                    vol_meta[split][vol_id] = {
                        'shape_orig': vol_prev_meta['shape_orig'],
                        'shape_patched': patched_shape_new,
                        'start_idx': cur_new,
                        'n_patches': n_patches_new,
                        'padding': padding_new[vol_id]
                    }

                    cur_old += vol_prev_meta['n_patches']
                    cur_new += n_patches_new

                del f[split][partition]
                f[split][partition] = f['temp']
                del f['temp']

                f.attrs['vol_shapes'] = str(dict(vol_meta))

                # update ds attributes
                f.attrs['patch_size'] = patch_size_new
                f.attrs['stride'] = stride_new

    def worker(self, params):
        vol_id, volume_path, mask_path, output_dir, split = params
        volume, header = nrrd.read(volume_path, index_order='C')
        spacing = np.diagonal(header['space directions'])[::-1]
        new_shape = ((spacing / 0.625) * volume.shape).round().astype(np.int64)
        volume = resize(volume, new_shape, order=1, preserve_range=True)
        padding = get_patch_padding(volume.shape, self.patch_size, self.stride)
        volume_patches, patched_shape = vol2patches(volume, self.patch_size, self.stride, padding, pad_value=-1000) # -1000 corresponds to air in HU units

        # if self.data_clip_range is not None:
        #     lb, ub = self.data_clip_range
        #     mask = (volume_patches > lb) & (volume_patches < ub) 
        #     volume_patches = np.clip(volume_patches, lb, ub)
        # else:
        #     mask = np.ones_like(volume_patches)
        # mean = volume_patches[mask].mean()
        # std  = volume_patches[mask].std()
        # volume_patches = (volume_patches - mean) / std

        shape_orig = volume.shape
        n_patches = volume_patches.shape[0]

        np.save(Path(output_dir, 'vols', f'{vol_id}.npy'), volume_patches)
        del volume, volume_patches

        mask, _ = nrrd.read(mask_path, index_order='C')
        mask = resize(mask, new_shape, order=0, mode='constant', cval=0, clip=True, anti_aliasing=False)
        mask_patches, _ = vol2patches(mask, self.patch_size, self.stride, padding)

        np.save(Path(output_dir, 'masks', f'{vol_id}.npy'), mask_patches)
        del mask, mask_patches

        return ( vol_id, {
                'shape_orig': shape_orig,
                'shape_resized': new_shape.tolist(),
                'split': split,
                'orig_spacing': spacing.tolist(),
                'shape_patched': patched_shape,
                'n_patches': n_patches,
                'padding': padding
                })

    def build_dataset(self, volume_path, mask_path, output_dir='processed'):
        meta = OrderedDict({
            'patch_size': [ int(x) for x in self.patch_size ],
            'stride': [ int(x) for x in self.stride],
            'normalize': self.normalize,
            'data_clip_range': [ int(x) for x in self.data_clip_range],
        })

        os.makedirs(output_dir, exist_ok=True)
        for split in ['train', 'valid']:
            os.makedirs(Path(output_dir, split), exist_ok=True)
            for part in ['vols', 'masks']:
                os.makedirs(Path(output_dir, split, part), exist_ok=True)
    
        def get_folderpath(file_id, output_dir):
            split = 'train' if file_id < 32 else 'valid'
            return Path(output_dir, split), split

        paths = [ ( file_id,
                    Path(volume_path, f'{file_id}.nrrd'),
                    Path(mask_path, f'{file_id}.nrrd'),
                    *get_folderpath(file_id, output_dir)) for file_id in range(40) ]
        
        with ProcessPoolExecutor(max_workers=6) as exec:
            vol_meta = list(tqdm(
                exec.map(self.worker, paths),
                total=len(paths)))
        
        meta['vol_meta'] = { m[0]: m[1] for m in vol_meta }

        with open(Path(output_dir, 'dataset.json'), 'w') as f:
            json.dump(meta, f, indent=4)

    def build_hdf5_dataset(self, volume_path, mask_path, output_path='asoca.hdf5'):
        with h5py.File(output_path, 'w') as f:
            f.attrs['patch_size'] = self.patch_size
            f.attrs['stride'] = self.stride
            f.attrs['normalize'] = self.normalize or 'None'

            volume_paths = [ Path(volume_path, filename) for filename in os.listdir(volume_path) ]
            mask_paths = [ Path(mask_path, filename) for filename in os.listdir(mask_path) ]

            train_paths = zip(volume_paths[:32], mask_paths[:32])
            valid_paths = zip(volume_paths[32:], mask_paths[32:])

            N_train = self.get_total_patches(volume_paths[:32])
            N_valid = self.get_total_patches(volume_paths[32:])

            self.logger.info('Building train dataset')
            train_group = f.create_group('train')
            train_shapes = self.populate_dataset(f, train_paths, train_group, N_train)

            self.logger.info('Building valid dataset')
            valid_group = f.create_group('valid')
            valid_shapes = self.populate_dataset(f, valid_paths, valid_group, N_valid)

            f.attrs['vol_shapes'] = str({ 'train': train_shapes, 'valid': valid_shapes })

