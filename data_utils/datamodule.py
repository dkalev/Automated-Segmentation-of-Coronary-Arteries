import os
import re
import ast
import h5py
import nrrd
import torch
import shutil
import zipfile
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from .dataset import AsocaDataset
from .helpers import get_n_patches, get_patch_padding, vol2patches, patches2vol

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger()


class DatasetBuilder():
    @staticmethod
    def is_valid(f):
        if 'patch_size' not in f.attrs: return False
        if 'stride' not in f.attrs: return False
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
            logger.info(f'HDF5 dataset not found at {self.datapath}')
            return 'build'

        try:
            with h5py.File(self.datapath, 'r') as f:
                if not self.is_valid(f):
                    raise Exception
                elif self.patch_size == f.attrs['patch_size'] and self.stride == f.attrs['stride']:
                    logger.info(f'Using HDF5 dataset found at: {self.datapath}')
                    return 'use'
                else:
                    logger.info(f'Resampling existing HDF5 dataset with patch size: {self.patch_size}, stride: {self.stride}')
                    return 'resample' 
        except Exception:
            logger.info(f'Found corrupted HDF5 dataset. Building from scratch.')
            os.remove(self.datapath)
            return 'build'

    def get_total_patches(self, filepaths):
        total = 0
        for filepath in filepaths:
            volume_shape = nrrd.read_header(str(filepath))['sizes']
            total += get_n_patches(volume_shape, self.patch_size, self.stride)
        return total

    @staticmethod
    def normalize_ds(ds, batch_size=1000, global_stats=True):
        # TODO check if computing stats globally works better
        # as in https://www.kaggle.com/akh64bit/full-preprocessing-tutorial
        axis = None if global_stats else 0
        mean = 0
        std = 0
        logger.info('Computing dataset mean and std.')
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

        logger.info('Normalizing dataset')
        for cur in tqdm(range(0, len(ds), batch_size)):
            ds[cur: cur+batch_size] = (ds[cur: cur+batch_size] - mean) / std
    
    def populate_dataset(self, f, data_paths, hdf_group, num_patches):
        ds_size = (num_patches, self.patch_size, self.patch_size, self.patch_size)
        volume_ds = hdf_group.create_dataset('volumes', ds_size)
        mask_ds = hdf_group.create_dataset('masks', ds_size)

        cur_start = 0 
        shapes = {}
        for volume_path, mask_path in tqdm(list(data_paths)):
            volume, _ = nrrd.read(volume_path, index_order='C')
            padding = get_patch_padding(volume.shape, self.patch_size, self.stride)
            volume_patches, patched_shape = vol2patches(volume, self.patch_size, self.stride, padding, pad_value=-1000) # -1000 corresponds to air in HU units

            cur_end = cur_start + len(volume_patches)

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

        if self.normalize:
            self.normalize_ds(volume_ds)
        
        return shapes
    
    def resample_patches(self):
        with h5py.File(self.datapath, 'r+') as f:
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
                    for vol_hash, vol_prev_meta in tqdm(shapes_old[split].items()):
                        batch = f[split][partition][cur_old: cur_old+vol_prev_meta['n_patches']]
                        batch = torch.from_numpy(batch)
                        batch = batch.view(vol_prev_meta['shape_patched'])

                        vol = patches2vol(batch, patch_size_prev, stride_prev, vol_prev_meta['padding'])
                        padding_new = get_patch_padding(vol.shape, patch_size_new, stride_new)
                        pad_value = -1000 if partition == 'volume' else 0

                        patches, patched_shape_new = vol2patches(vol, patch_size_new, stride_new, padding_new, pad_value=pad_value)
                        n_patches_new = patches.shape[0]

                        f['temp'][cur_new: cur_new+n_patches_new] = patches

                        vol_meta[split][vol_hash] = {
                            'shape_orig': vol_prev_meta['shape_orig'],
                            'shape_patched': patched_shape_new,
                            'start_idx': cur_new,
                            'n_patches': n_patches_new,
                            'padding': padding_new
                        }

                        cur_old += vol_prev_meta['n_patches']
                        cur_new += n_patches_new
                    
                    del f[split][partition]
                    f[split][partition] = f['temp']
                    del f['temp']

                    f.attrs['vol_shapes'] = str(vol_meta)

            # update ds attributes
            f.attrs['patch_size'] = patch_size_new
            f.attrs['stride'] = stride_new

    def build_hdf5_dataset(self, volume_path, mask_path, output_path='asoca.hdf5'):
        with h5py.File(output_path, 'w') as f:
            f.attrs['patch_size'] = self.patch_size
            f.attrs['stride'] = self.stride
                
            volume_paths = [ Path(volume_path, filename) for filename in os.listdir(volume_path) ]
            mask_paths = [ Path(mask_path, filename) for filename in os.listdir(mask_path) ]
            
            train_paths = zip(volume_paths[:32], mask_paths[:32])
            valid_paths = zip(volume_paths[32:], mask_paths[32:])
            
            N_train = self.get_total_patches(volume_paths[:32])
            N_valid = self.get_total_patches(volume_paths[32:])
            
            logger.info('Building train dataset')
            train_group = f.create_group('train')
            train_shapes = self.populate_dataset(f, train_paths, train_group, N_train)
            
            logger.info('Building valid dataset')
            valid_group = f.create_group('valid')
            valid_shapes = self.populate_dataset(f, valid_paths, valid_group, N_valid)
            
            f.attrs['vol_shapes'] = str({ 'train': train_shapes, 'valid': valid_shapes })


class AsocaDataModule(LightningDataModule, DatasetBuilder):
    def __init__(self, *args,
                batch_size=1,
                patch_size=32,
                stride=None,
                normalize=True, # has no effect on resample; only when building from scratch
                sourcepath='dataset/ASOCA2020Data.zip',
                output_dir='dataset', **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.stride = patch_size if stride is None else stride
        self.normalize = normalize
        self.sourcepath = sourcepath
        self.output_dir = output_dir
        self.datapath = Path(output_dir, f'asoca.hdf5')

    def prepare_data(self):
        action = self.verify_dataset()

        if action == 'resample':
            self.resample_patches()
        elif action == 'build':
            subdirs = ['Train', 'Train_Masks', 'Test']
            folders_exist = [ Path(self.output_dir, subdir).is_dir() for subdir in subdirs ]
            if not all(folders_exist):
                logger.info(f'Extracting data from {self.sourcepath}')
                self.extract_files(subdirs)
            
            logger.info('Building HDF5 dataset')
            volume_path = Path(self.output_dir, 'Train')
            mask_path = Path(self.output_dir, 'Train_Masks')
            self.build_hdf5_dataset(volume_path, mask_path, output_path=self.datapath)

        logger.info('Done')

    def train_dataloader(self, batch_size=None):
        if batch_size is None: batch_size = self.batch_size
        train_split = AsocaDataset(self.datapath, split='train')
        return DataLoader(train_split, shuffle=True, batch_size=batch_size, num_workers=12, pin_memory=True)

    def val_dataloader(self, batch_size=None):
        if batch_size is None: batch_size = self.batch_size
        valid_split = AsocaDataset(self.datapath, split='valid')
        return DataLoader(valid_split, batch_size=batch_size, num_workers=12, pin_memory=True)

    def volume_dataloader(self, file_id, batch_size=None):
        if batch_size is None: batch_size = self.batch_size
        ds = AsocaDataset(self.datapath, file_id=file_id)
        meta = {
            'shape_orig': ds.shape_orig,
            'shape_patched': ds.shape_patched,
            'n_patches': len(ds),
        }
        return DataLoader(ds, batch_size=batch_size, num_workers=12, pin_memory=True), meta
