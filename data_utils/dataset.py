import os
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class AsocaDataset(Dataset):
    def __init__(self, ds_path='dataset/processed', split='train'):
        self.ds_path = Path(ds_path, split)
        with open(Path(ds_path, 'dataset.json'), 'r') as f:
            meta = json.load(f)
        self.vol_meta = { int(k): v for k, v in meta['vol_meta'].items() if v['split'] == split }

    @property
    def max_patches(self):
        if not hasattr(self, '_max_patches'):
            self._max_patches = np.max([ m['n_patches'] for m in self.vol_meta.values() ])
        return self._max_patches

    @property
    def total_patches(self):
        if not hasattr(self, '_total_patches'):
            self._total_patches = np.sum([ m['n_patches'] for m in self.vol_meta.values() ])
        return self._total_patches

    def __len__(self):
        return self.total_patches

    def split_index(self, index):
        if isinstance(index, int) or isinstance(index, np.int64):
            if index < 0: index = self.len + index
            file_id, idx = index // self.max_patches, index % self.max_patches
        elif isinstance(index, slice):
            file_id = index.start // self.max_patches
            idx = slice(index.start % self.max_patches, index.stop % self.max_patches)
        return file_id, idx

    def __getitem__(self, index):
        file_id, patch_idx = self.split_index(index)
        x = np.load(Path(self.ds_path, 'vols', f'{file_id}.npy'), mmap_mode='r+')
        y = np.load(Path(self.ds_path, 'masks', f'{file_id}.npy'), mmap_mode='r+')
        x, y = x[patch_idx], y[patch_idx]
        x, y = torch.tensor(x), torch.LongTensor(y)

        if len(x.shape) == 3: x, y = x.unsqueeze(0), y.unsqueeze(0)

        return x, y, (file_id, patch_idx)


class AsocaVolumeDataset(AsocaDataset):
    def __init__(self, *args, vol_id, **kwargs):
        split = self.infer_split(kwargs['ds_path'], vol_id)
        super().__init__(*args, split=split, **kwargs)
        self.vol_meta = { vol_id: self.vol_meta[vol_id] }
        self.vol_id = vol_id

    def infer_split(self, ds_path, vol_id):
        vol_ids_train = [ int(fn[:-4]) for fn in os.listdir(Path(ds_path, 'train', 'vols')) ]
        vol_ids_valid = [ int(fn[:-4]) for fn in os.listdir(Path(ds_path, 'valid', 'vols')) ]
        if vol_id in vol_ids_train:
            return 'train'
        elif vol_id in vol_ids_valid:
            return 'valid'
        else:
            raise ValueError('Cannot find volume {vol_id} in train and valid splits for dataset path {ds_path}')

    def get_vol_meta(self):
        return self.vol_meta[self.vol_id]

    # overrides the parent method to ensure only the current volume is accessed
    def split_index(self, index):
        return self.vol_id, index

    def __getitem__(self, index):
        x, _, _ =  super().__getitem__(index)
        return x

