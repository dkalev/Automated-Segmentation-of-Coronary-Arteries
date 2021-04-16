from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import torch.nn.functional as F
import torch
import os
import re
import ast
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import zipfile
import h5py
import nrrd
import logging
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger()

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

def get_padding(input_shape, output_shape):
    input_shape = np.array(input_shape) if not isinstance(input_shape, np.ndarray) else input_shape
    output_shape = np.array(output_shape) if not isinstance(output_shape, np.ndarray) else output_shape
    pad = (output_shape - input_shape).astype(int)
    pad_left = pad // 2
    pad_right = pad - pad_left
    res = list(zip(pad_left[::-1], pad_right[::-1]))
    return tuple(x for y in res for x in y )

def get_patch_padding(vol_shape, patch_size, stride):
    vol_shape = np.array(vol_shape)
    pad_shape = np.ceil((vol_shape - patch_size) / stride) * stride + patch_size
    return get_padding(vol_shape, pad_shape)

def get_n_patches(volume_shape, patch_size, stride):
    # assumes padding is applied to the volume
    volume_shape = np.array(volume_shape) if not isinstance(volume_shape, np.ndarray) else volume_shape
    return int(np.prod(np.ceil((volume_shape - patch_size) / stride) + 1))

def vol2patches(volume, patch_size, stride, padding, pad_value=0):
    if not isinstance(volume, torch.Tensor):
        volume = torch.from_numpy(volume).float()

    padded_vol = F.pad(volume, padding, value=pad_value)
    
    patches = padded_vol.unfold(0, patch_size, stride) \
                        .unfold(1, patch_size, stride) \
                        .unfold(2, patch_size, stride)
    patched_shape = tuple(patches.shape)
    patches = patches.reshape(-1, patch_size, patch_size, patch_size)
    return patches, patched_shape

def patches2vol(patches, patch_size, stride, padding=None):
    offset = patch_size - stride
    p = patches.permute(0,3,1,4,2,5)
    p = torch.cat((p[0], p[1:][:,offset:].flatten(end_dim=1)))
    p = p.permute(1,2,3,4,0)
    p = torch.cat((p[0],p[1:][:,offset:].flatten(end_dim=1)))
    p = p.permute(1,2,3,0)
    p = torch.cat((p[0],p[1:][:,offset:].flatten(end_dim=1)))
    
    p = p.permute(1,2,0)
    
    if padding:
        slicer = [slice(pad_left,-pad_right) for pad_left ,pad_right in list(zip(padding[0::2],padding[1::2]))][::-1]
        # otherwise zero padding results in a slice fetching zero items, instead of all
        slicer = [sl if not (sl.start == 0 and sl.stop == 0) else slice(None,None,None) for sl in slicer ]
        p = p[slicer]
        
    return p

def get_volume_pred(patches, vol_meta, stride):
    out_shape = vol_meta['shape_patched'][:3] + patches.shape[1:]
    res = patches2vol(patches.view(out_shape), stride, stride)
    output_pad = get_padding(res.shape, vol_meta['shape_orig'])
    res = F.pad(res, output_pad)
    return res

class AsocaDataModule(LightningDataModule):
    def __init__(self, *args,
                batch_size=1,
                patch_size=32,
                stride=None,
                normalize=True,
                sourcepath='dataset/ASOCA2020Data.zip',
                output_dir='dataset', **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.stride = patch_size if stride is None else stride
        self.normalize = normalize
        self.sourcepath = sourcepath
        self.output_dir = output_dir
        self.datapath = Path(output_dir, f'asoca-{patch_size}.hdf5')

    def is_valid(self, f):
        if 'patch_size' not in f.attrs: return False
        if 'stride' not in f.attrs: return False
        if not all( key in f.keys() for key in ['train', 'valid']): return False
        if not all( key in f['train'].keys() for key in ['volumes', 'masks']): return False
        if not all( key in f['valid'].keys() for key in ['volumes', 'masks']): return False
        return True

    def prepare_data(self):
        resample = False
        if self.datapath.is_file():
            try:
                with h5py.File(self.datapath, 'r') as f:
                    if not self.is_valid(f):
                        logger.info(f'Found corrupted HDF5 dataset. Building from scratch.')
                        os.remove(self.datapath)
                    elif self.patch_size == f.attrs['patch_size'] and self.stride == f.attrs['stride']:
                        logger.info(f'Using HDF5 dataset found at: {self.datapath}')
                        return
                    else:
                        logger.info(f'Resampling existing HDF5 dataset with patch size: {self.patch_size}, stride: {self.stride}')
                        resample = True
            except Exception:
                logger.info(f'Found corrupted HDF5 dataset. Building from scratch.')
                os.remove(self.datapath)
        else:
            logger.info(f'HDF5 dataset not found at {self.datapath}')

        subdirs = ['Train', 'Train_Masks', 'Test']

        folders_exist = [ Path(self.output_dir, subdir).is_dir() for subdir in subdirs ]
        if not all(folders_exist):
            logger.info(f'Unzipping data from {self.sourcepath}')
            for subdir in subdirs:
                if Path(self.output_dir, subdir).is_dir():
                    shutil.rmtree(Path(self.output_dir, subdir))

            with zipfile.ZipFile(self.sourcepath, 'r') as zip_ref:
                zip_ref.extractall(self.output_dir)
        
        logger.info('Building HDF5 dataset')
        volume_path = Path(self.output_dir, 'Train')
        mask_path = Path(self.output_dir, 'Train_Masks')

        if resample:
            self.resample_patches()
        else:
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

    def get_total_patches(self, filepaths):
        total = 0
        for filepath in filepaths:
            volume_shape = nrrd.read_header(str(filepath))['sizes']
            total += get_n_patches(volume_shape, self.patch_size, self.stride)
        return total

    @staticmethod
    def normalize_ds(ds, global_stats=True):
        # TODO check if computing stats globally works better
        # as in https://www.kaggle.com/akh64bit/full-preprocessing-tutorial
        if global_stats:
            mean = ds[:].mean()
            std  = ds[:].std()
        else:
            mean = ds[:].mean(axis=0)
            std  = ds[:].std(axis=0)
        return (ds[:] - mean) / std
    
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
            volume_ds[:] = self.normalize_ds(volume_ds)
        
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
                    # create a temp dataset for group
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
        

if __name__ == '__main__':

    asoca_dm = AsocaDataModule(batch_size=16, patch_size=64, stride=52, normalize=False)
    asoca_dm.prepare_data()
    asoca_dm.setup()
    train_dl = asoca_dm.train_dataloader()
    valid_dl = asoca_dm.val_dataloader()
    logger.info(next(iter(train_dl))[0].shape)
    logger.info(next(iter(valid_dl))[0].shape)