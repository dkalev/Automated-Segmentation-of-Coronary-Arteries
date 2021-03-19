from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import torch
import torch.nn.functional as F
import os
from pathlib import Path
from tqdm import tqdm
import shutil
import zipfile
import h5py
import nrrd

class AsocaDataset(Dataset):
    def __init__(self, ds_path, split='train',):
        self.ds = h5py.File(ds_path, 'r')[split] # FIXME: might not work with multiprocessing

    def __len__(self):
        return len(self.ds['volumes'])

    def __getitem__(self, index):
        return self.ds['volumes'][index], self.ds['masks'][index]


class AsocaDataModule(LightningDataModule):
    def __init__(self, *args, batch_size=1, patch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.ds_filename = 'asoca.hdf5'

    def prepare_data(self, datapath='dataset/ASOCA2020Data.zip', output_dir='dataset'):
        ds_path = Path(output_dir, self.ds_filename)
        if ds_path.is_file():
            with h5py.File(ds_path, 'r') as f:
                if self.patch_size == f['patch_size']:
                    return
                else:
                    print(f'Building a HDF5 dataset with patch size: {self.patch_size}')
        else:
            print(f'HDF5 dataset not found at {ds_path}')

        subdirs = ['Train', 'Train_Masks', 'Test']

        folders_exist = [ Path(output_dir, subdir).is_dir() for subdir in subdirs ]
        if not all(folders_exist):
            print(f'Unzipping data from {datapath}')
            for subdir in subdirs:
                if Path(output_dir, subdir).is_dir():
                    shutil.rmtree(Path(output_dir, subdir))

            with zipfile.ZipFile(datapath, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
        
        print('Building HDF5 dataset')
        volume_path = Path(output_dir, 'Train')
        mask_path = Path(output_dir, 'Train_Masks')
        self.build_hdf5_dataset(volume_path, mask_path, output_path=ds_path)

        print('Done')

    def setup(self, stage=None):
        self.datapath = 'dataset/asoca.hdf5'

    def train_dataloader(self, batch_size=None):
        if batch_size is None: batch_size = self.batch_size
        train_split = AsocaDataset(self.datapath, split='train')
        return DataLoader(train_split, shuffle=True, batch_size=batch_size, num_workers=12)

    def val_dataloader(self, batch_size=None):
        if batch_size is None: batch_size = self.batch_size
        valid_split = AsocaDataset(self.datapath, split='valid')
        return DataLoader(valid_split, batch_size=batch_size, num_workers=12)

    def get_num_patches(self, filepaths):
        total = 0
        for filepath in filepaths:
            volume_shape = nrrd.read_header(str(filepath))['sizes']
            n_patches = (volume_shape[0] // self.patch_size) * \
                        (volume_shape[1] // self.patch_size) * \
                        (volume_shape[2] // self.patch_size)
            total += n_patches
        return total
    
    def get_patches(self, volume):
        volume = torch.Tensor(volume)
        return volume.unfold(0, self.patch_size, self.patch_size) \
                    .unfold(1, self.patch_size, self.patch_size) \
                    .unfold(2, self.patch_size, self.patch_size) \
                    .reshape(-1, self.patch_size, self.patch_size, self.patch_size) \
                    .numpy()

    def populate_dataset(self, data_paths, hdf_group, ds_name, n_patches):
        ds_size = (n_patches, self.patch_size, self.patch_size, self.patch_size)
        ds = hdf_group.create_dataset(ds_name, ds_size)
        last = 0
        for filepath in tqdm(data_paths):
            volume, _ = nrrd.read(filepath, index_order='C')
            patches = self.get_patches(volume)
            ds[last:last+len(patches),...] = patches
            last += len(patches)
    
    def build_hdf5_dataset(self, volume_path, mask_path, output_path='asoca.hdf5'):
        with h5py.File(output_path, 'w') as f:
            f['patch_size'] = self.patch_size
                
            volume_paths = [ Path(volume_path, filename) for filename in os.listdir(volume_path) ]
            mask_paths = [ Path(mask_path, filename) for filename in os.listdir(mask_path) ]
            
            volume_paths_train = volume_paths[:30]
            mask_paths_train = mask_paths[:30]
            
            volume_paths_valid = volume_paths[30:]
            mask_paths_valid = mask_paths[30:]
            
            N_train = self.get_num_patches(volume_paths_train)
            N_valid = self.get_num_patches(volume_paths_valid)
            
            train_group = f.create_group('train')
            print('Building train dataset')
            self.populate_dataset(volume_paths_train, train_group, 'volumes', N_train)                    
            self.populate_dataset(mask_paths_train, train_group, 'masks', N_train)
            
            print('Building valid dataset')
            valid_group = f.create_group('valid')
            self.populate_dataset(volume_paths_valid, valid_group, 'volumes', N_valid)                    
            self.populate_dataset(mask_paths_valid, valid_group, 'masks', N_valid)
        

if __name__ == '__main__':

    asoca_dm = AsocaDataModule(batch_size=8, patch_size=128)
    asoca_dm.prepare_data()
    asoca_dm.setup()
    train_dl = asoca_dm.train_dataloader()
    valid_dl = asoca_dm.val_dataloader()
    print(next(iter(train_dl))[0].shape)
    print(next(iter(valid_dl))[0].shape)