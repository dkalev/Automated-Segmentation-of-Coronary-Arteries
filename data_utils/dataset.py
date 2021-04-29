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
        self.vol_meta = { int(key): val for key, val in meta['vol_meta'].items() if val['split'] == split }
        patch_meta = [ m['n_patches'] for m in self.vol_meta.values() ]
        self.max_patches = np.max(patch_meta)
        self.len = np.sum(patch_meta)

    def __len__(self):
        return self.len

    def split_index(self, index):
        if isinstance(index, int) or isinstance(index, np.int64):
            file_id, idx = index // self.max_patches, index % self.max_patches
        elif isinstance(index, slice):
            file_id = index.start // self.max_patches
            idx = slice(index.start % self.max_patches, index.stop % self.max_patches)
        return file_id, idx
    
    def cur_vol(self, file_id):
        if not hasattr(self, '_cur_vol') or self._cur_vol_id != file_id:
            self._cur_vol = np.load(Path(self.ds_path, 'vols', f'{file_id}.npy'))
            self._cur_vol_id = file_id
        return self._cur_vol

    def cur_mask(self, file_id):
        if not hasattr(self, '_cur_mask') or self._cur_mask_id != file_id:
            self._cur_mask = np.load(Path(self.ds_path, 'masks', f'{file_id}.npy'))
            self._cur_mask_id = file_id
        return self._cur_mask

    def __getitem__(self, index):
        file_id, idx = self.split_index(index)
        x = np.load(Path(self.ds_path, 'vols', f'{file_id}.npy'))
        y = np.load(Path(self.ds_path, 'masks', f'{file_id}.npy'))
        x, y = x[idx], y[idx]
        x, y = torch.tensor(x), torch.tensor(y)
        if len(x.shape) == 3: x, y = x.unsqueeze(0), y.unsqueeze(0)
        return x, y
