from torch.utils.data import Dataset
import h5py
import torch

class MnistDataset(Dataset):
    def __init__(self, path, split='train'):
        self.path = path
        self.split = split
        self.len = len(h5py.File(self.path, 'r')[self.split]['vols'])
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        self.ds = h5py.File(self.path, 'r')[self.split]
        x, y = self.ds['vols'][index], self.ds['masks'][index]
        x, y = torch.tensor(x), torch.tensor(y)
        if len(x.shape) == 3:
            x, y = x.unsqueeze(0), y.unsqueeze(0)
        return x, y