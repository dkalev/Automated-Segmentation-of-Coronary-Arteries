import h5py
import logging
import traceback
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from .dataset import AsocaNumpyDataset, AsocaDataset
from .dataset_builder import DatasetBuilder
from .sampler import ASOCASampler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger()


class AsocaDataModule(DatasetBuilder, LightningDataModule):
    def __init__(self, *args,
                batch_size=1,
                patch_size=32,
                patch_stride=None,
                normalize='global', # has no effect on resample; only when building from scratch
                data_clip_range=(0, 400),
                sourcepath='dataset/ASOCA2020Data.zip',
                datapath='dataset/asoca.hdf5',
                output_dir='dataset', **kwargs):
        super().__init__(logger, *args, **kwargs)
        self.batch_size = batch_size

        if isinstance(patch_size, int): patch_size = np.array([patch_size, patch_size, patch_size])
        self.patch_size = patch_size

        if patch_stride is None:
            patch_stride = self.patch_size
        elif isinstance(patch_stride, int):
            patch_stride = np.array([patch_stride, patch_stride, patch_stride])
        self.stride = patch_stride
        self.data_clip_range = np.array(data_clip_range) if data_clip_range != 'None' else None

        self.normalize = normalize
        self.sourcepath = sourcepath
        self.output_dir = output_dir
        self.datapath = Path(datapath)

    def prepare_data(self):
        action = self.verify_dataset()

        # if action == 'resample':
        #     with h5py.File(self.datapath, 'r+') as f:
        #         try:
        #             self.resample_patches(f)
        #         except Exception:
        #             if 'temp' in f: del f['temp']
        #             logger.error(traceback.format_exc())
        # elif action == 'build':
        #     subdirs = ['Train', 'Train_Masks', 'Test']
        #     folders_exist = [ Path(self.output_dir, subdir).is_dir() for subdir in subdirs ]
        #     if not all(folders_exist):
        #         logger.info(f'Extracting data from {self.sourcepath}')
        #         self.extract_files(subdirs)

        #     logger.info('Building HDF5 dataset')
        #     volume_path = Path(self.output_dir, 'Train')
        #     mask_path = Path(self.output_dir, 'Train_Masks')
        #     self.build_dataset(volume_path, mask_path, output_dir=Path(self.output_dir, 'processed'))

        logger.info('Done')

    def train_dataloader(self, batch_size=None):
        if batch_size is None: batch_size = self.batch_size
        train_split = AsocaNumpyDataset(split='train')
        sampler = ASOCASampler(train_split.vol_meta, shuffle=True)
        return DataLoader(train_split, sampler=sampler, batch_size=batch_size, num_workers=12, pin_memory=True)

    def val_dataloader(self, batch_size=None):
        if batch_size is None: batch_size = self.batch_size
        valid_split = AsocaNumpyDataset(split='valid')
        sampler = ASOCASampler(valid_split.vol_meta)
        return DataLoader(valid_split, sampler=sampler, batch_size=batch_size, num_workers=12, pin_memory=True)

    def volume_dataloader(self, file_id, batch_size=None):
        if batch_size is None: batch_size = self.batch_size
        ds = AsocaDataset(self.datapath, file_id=file_id)
        meta = {
            'shape_orig': ds.shape_orig,
            'shape_patched': ds.shape_patched,
            'n_patches': len(ds),
        }
        return DataLoader(ds, batch_size=batch_size, num_workers=12, pin_memory=True), meta
