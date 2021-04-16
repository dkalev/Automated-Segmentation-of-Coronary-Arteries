import ast
import h5py
import torch
from torch.utils.data import Dataset

class AsocaDataset(Dataset):
    def __init__(self, ds_path, file_id=None, split='train'):
        self.ds_path = ds_path
        ds = h5py.File(ds_path, 'r')
        if file_id is not None:
            vol_meta = ast.literal_eval(ds.attrs['vol_shapes'])
            for split in ['train', 'valid']: # FIXME same variable as kwarg
                if file_id in vol_meta[split]:
                    self.split = split
                    break
            if not hasattr(self, 'split'): #FIXME there's gotta be a better way
                raise ValueError(f'File id: {file_id} could not be found in the volumes meta data')

            self.shape_orig = vol_meta[self.split][file_id]['shape_orig']
            self.shape_patched = vol_meta[self.split][file_id]['shape_patched']
            self.start_idx = vol_meta[self.split][file_id]['start_idx']
            self.len = vol_meta[self.split][file_id]['n_patches']
        else:
            self.start_idx = 0
            self.split = split
            self.len = len(ds[split]['volumes'])
        del ds

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        self.ds = h5py.File(self.ds_path, 'r')[self.split]
        x, y = self.ds['volumes'][self.start_idx+index], self.ds['masks'][self.start_idx+index]
        x, y = torch.tensor(x), torch.tensor(y)
        if len(x.shape) == 3:
            x, y = x.unsqueeze(0), y.unsqueeze(0)
        return x, y

