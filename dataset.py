from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import torch
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import zipfile
import h5py
import nrrd
import logging
from torchvision.transforms import ToTensor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger()

class AsocaDataset(Dataset):
    def __init__(self, ds_path, split='train', transform=None):
        self.ds_path = ds_path
        self.split = split
        ds = h5py.File(ds_path, 'r')[split]
        self.len = len(ds['volumes'])
        del ds
        self.transform = ToTensor() if transform is None else transform

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        self.ds = h5py.File(self.ds_path, 'r')[self.split]
        x, y = self.ds['volumes'][index], self.ds['masks'][index]
        x, y = self.transform(x), self.transform(y)
        return x.unsqueeze(0), y.unsqueeze(0)


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
                if self.patch_size == f.attrs['patch_size']:
                    if not all( key in f.keys() for key in ['train', 'valid']):
                        logger.info(f'Fould corrupted HDF5 dataset. Building from scratch.')
                        os.remove(ds_path)
                    else:
                        logger.info(f'Using HDF5 dataset found at: {ds_path}')
                        return
                else:
                    logger.info(f'Building a HDF5 dataset with patch size: {self.patch_size}')
        else:
            logger.info(f'HDF5 dataset not found at {ds_path}')

        subdirs = ['Train', 'Train_Masks', 'Test']

        folders_exist = [ Path(output_dir, subdir).is_dir() for subdir in subdirs ]
        if not all(folders_exist):
            logger.info(f'Unzipping data from {datapath}')
            for subdir in subdirs:
                if Path(output_dir, subdir).is_dir():
                    shutil.rmtree(Path(output_dir, subdir))

            with zipfile.ZipFile(datapath, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
        
        logger.info('Building HDF5 dataset')
        volume_path = Path(output_dir, 'Train')
        mask_path = Path(output_dir, 'Train_Masks')
        self.build_hdf5_dataset(volume_path, mask_path, output_path=ds_path)

        logger.info('Done')

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
            n_patches = np.prod(volume_shape // self.patch_size)
            total += n_patches
        return total

    def get_patches(self, volume, ds_name='volumes'):
        volume = torch.from_numpy(volume)
        volume = volume.unfold(0, self.patch_size, self.patch_size) \
                    .unfold(1, self.patch_size, self.patch_size) \
                    .unfold(2, self.patch_size, self.patch_size) \
                    .reshape(-1, self.patch_size, self.patch_size, self.patch_size)
        is_used = (volume.sum(axis=[1,2,3]) > 0).numpy() if ds_name == 'masks' else None
        return volume.numpy(), is_used 

    def populate_dataset(self, data_paths, hdf_group, num_patches):
        max_ds_size = (num_patches, self.patch_size, self.patch_size, self.patch_size)
        t_volume_ds = hdf_group.create_dataset('volumes_temp', max_ds_size)
        t_mask_ds = hdf_group.create_dataset('masks_temp', max_ds_size)

        last = 0 
        for volume_path, mask_path in tqdm(list(data_paths)):
            mask, _ = nrrd.read(mask_path, index_order='C')
            mask_patches, is_used = self.get_patches(mask, 'masks')
            mask_patches = mask_patches[is_used]
            t_mask_ds[last:last+len(mask_patches),...] = mask_patches

            volume, _ = nrrd.read(volume_path, index_order='C')
            volume_patches, _ = self.get_patches(volume, 'volumes')
            volume_patches = volume_patches[is_used]
            t_volume_ds[last:last+len(volume_patches),...] = volume_patches

            last += len(volume_patches)

        ds_size = (last, *max_ds_size[1:])
        volume_ds = hdf_group.create_dataset('volumes', ds_size)
        mask_ds = hdf_group.create_dataset('masks', ds_size)
        volume_ds[:] = t_volume_ds[:last]
        mask_ds[:] = t_mask_ds[:last]
        del t_volume_ds
        del t_mask_ds
    
    def build_hdf5_dataset(self, volume_path, mask_path, output_path='asoca.hdf5'):
        with h5py.File(output_path, 'w') as f:
            f.attrs['patch_size'] = self.patch_size
                
            volume_paths = [ Path(volume_path, filename) for filename in os.listdir(volume_path) ]
            mask_paths = [ Path(mask_path, filename) for filename in os.listdir(mask_path) ]
            
            train_paths = zip(volume_paths[:30], mask_paths[:30])
            valid_paths = zip(volume_paths[30:], mask_paths[30:])
            
            N_train_max = self.get_num_patches(volume_paths[:30])
            N_valid_max = self.get_num_patches(volume_paths[30:])
            
            logger.info('Building train dataset')
            train_group = f.create_group('train')
            self.populate_dataset(train_paths, train_group, N_train_max)                    
            
            logger.info('Building valid dataset')
            valid_group = f.create_group('valid')
            self.populate_dataset(valid_paths, valid_group, N_valid_max)                    
        

if __name__ == '__main__':

    asoca_dm = AsocaDataModule(batch_size=16, patch_size=64)
    asoca_dm.prepare_data()
    asoca_dm.setup()
    train_dl = asoca_dm.train_dataloader()
    valid_dl = asoca_dm.val_dataloader()
    logger.info(next(iter(train_dl))[0].shape)
    logger.info(next(iter(valid_dl))[0].shape)