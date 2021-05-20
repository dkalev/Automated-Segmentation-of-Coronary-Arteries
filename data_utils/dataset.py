import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class AsocaDataset(Dataset):
    def __init__(self, ds_path='dataset/processed', heart_mask=True, split='train'):
        self.ds_path = Path(ds_path, split)
        self.heart_mask = heart_mask
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
        file_id, idx = self.split_index(index)
        x = np.load(Path(self.ds_path, 'vols', f'{file_id}.npy'), mmap_mode='r+')
        y = np.load(Path(self.ds_path, 'masks', f'{file_id}.npy'), mmap_mode='r+')
        x, y = x[idx], y[idx]
        x, y = torch.tensor(x), torch.LongTensor(y)

        hm = None
        if self.heart_mask:
            hm = np.load(Path(self.ds_path, 'heart_masks', f'{file_id}.npy'), mmap_mode='r+')
            hm = torch.tensor(hm[idx])

        if len(x.shape) == 3:
            x, y = x.unsqueeze(0), y.unsqueeze(0)
            if self.heart_mask: hm = hm.unsqueeze(0)
        
        return x, y, hm

class AsocaVolumeDataset(AsocaDataset):
    def __init__(self, *args, vol_id, **kwargs):
        super().__init__(*args, **kwargs)
        self.vol_meta = { vol_id: self.vol_meta[vol_id] }

    def get_vol_meta(self, vol_id):
        return self.vol_meta[int(vol_id)]

    def __getitem__(self, index):
        x, _, _ =  super().__getitem__(index)
        return x