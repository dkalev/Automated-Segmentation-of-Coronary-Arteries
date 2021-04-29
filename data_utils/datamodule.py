import shutil
import logging
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from .dataset import AsocaDataset, AsocaDataset
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
                normalize=False,
                data_clip_range=(0, 400),
                rebuild=False,
                resample_vols=True,
                crop_empty=False,
                sourcepath='dataset/ASOCA2020Data.zip',
                datapath='dataset/asoca.hdf5',
                data_dir='dataset/processed', **kwargs):
        super().__init__(logger, *args, **kwargs)
        self.batch_size = batch_size

        if isinstance(patch_size, int): patch_size = np.array([patch_size, patch_size, patch_size])
        self.patch_size = patch_size

        if patch_stride is None:
            patch_stride = self.patch_size
        elif isinstance(patch_stride, int):
            patch_stride = np.array([patch_stride, patch_stride, patch_stride])
        self.stride = patch_stride
        self.data_clip_range = list(data_clip_range) if data_clip_range != 'None' else None

        self.normalize = normalize
        self.rebuild = rebuild
        self.resample_vols = resample_vols
        self.crop_empty = crop_empty
        self.sourcepath = sourcepath
        self.data_dir = data_dir
        self.datapath = Path(datapath)

    def prepare_data(self):
        if not self.is_valid():
            self.logger.info(f'Corrupted dataset. Building from scratch.')
        elif self.rebuild:
            self.logger.info(f'Rebuild option set to true. Building from scratch.')
        else:
            logger.info(f'Using existing dataset located at {self.data_dir}')
            return

        if Path(self.data_dir).is_dir(): shutil.rmtree(self.data_dir)

        subdirs = ['Train', 'Train_Masks', 'Test']
        folders_exist = [ Path(self.data_dir, subdir).is_dir() for subdir in subdirs ]
        if not all(folders_exist):
            logger.info(f'Extracting data from {self.sourcepath}')
            self.extract_files(subdirs)

        logger.info('Building dataset')
        volume_path = Path(self.data_dir, 'Train')
        mask_path = Path(self.data_dir, 'Train_Masks')
        self.build_dataset(volume_path, mask_path)

        for subdir in subdirs:
            shutil.rmtree(Path(self.data_dir, subdir))

        logger.info('Done')
    
    def train_dataloader(self, batch_size=None):
        if batch_size is None: batch_size = self.batch_size
        train_split = AsocaDataset(split='train')
        sampler = ASOCASampler(train_split.vol_meta, shuffle=True, oversample=True)
        return DataLoader(train_split, sampler=sampler, batch_size=batch_size, num_workers=12, pin_memory=True)

    def val_dataloader(self, batch_size=None):
        if batch_size is None: batch_size = self.batch_size
        valid_split = AsocaDataset(split='valid')
        sampler = ASOCASampler(valid_split.vol_meta)
        return DataLoader(valid_split, sampler=sampler, batch_size=batch_size, num_workers=12, pin_memory=True)

    def volume_dataloader(self, file_id, batch_size=None):
        # FIXME update to work with new dataset
        if batch_size is None: batch_size = self.batch_size
        ds = AsocaDataset(self.datapath, file_id=file_id)
        meta = {
            'shape_orig': ds.shape_orig,
            'shape_patched': ds.shape_patched,
            'n_patches': len(ds),
        }
        return DataLoader(ds, batch_size=batch_size, num_workers=12, pin_memory=True), meta
