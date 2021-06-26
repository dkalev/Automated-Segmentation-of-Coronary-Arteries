from data_utils.distributed_sampler import DistributedSamplerWrapper
import shutil
import logging
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from .dataset import AsocaDataset, AsocaVolumeDataset
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
                oversample=False,
                distributed=False,
                perc_per_epoch_train=1,
                perc_per_epoch_val=1,
                data_dir='dataset/processed',
                sourcepath='dataset/ASOCA2020Data.zip', **kwargs):
        super().__init__(logger, *args, sourcepath=sourcepath, **kwargs)

        self.batch_size = batch_size

        if isinstance(patch_size, int): patch_size = np.array([patch_size, patch_size, patch_size])
        self.patch_size = patch_size

        if patch_stride is None:
            patch_stride = self.patch_size
        elif isinstance(patch_stride, np.ndarray):
            patch_stride = patch_stride.tolist()

        self.stride = patch_stride
        self.oversample = oversample
        self.perc_per_epoch_train = perc_per_epoch_train
        self.perc_per_epoch_val = perc_per_epoch_val
        self.distributed = distributed
        self.data_dir = data_dir

    def prepare_data(self):
        if not self.is_valid():
            self.logger.info(f'Corrupted dataset. Building from scratch.')
        elif self.is_config_updated():
            self.logger.info(f'Changed config. Building from scratch.')
        elif self.rebuild:
            self.logger.info(f'Rebuild option set to true. Building from scratch.')
        else:
            self.logger.info(f'Using existing dataset located at {self.data_dir}')
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

    def train_dataloader(self, batch_size=None, num_workers=None):
        if num_workers is None: num_workers = 3 if self.distributed else 4
        if batch_size is None: batch_size = self.batch_size
        train_split = AsocaDataset(ds_path=self.data_dir, split='train')
        sampler = ASOCASampler(train_split.vol_meta,
                                oversample=self.oversample,
                                perc_per_epoch=self.perc_per_epoch_train)
        if self.distributed:
            sampler = DistributedSamplerWrapper(sampler=sampler)
        return DataLoader(train_split, sampler=sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    def val_dataloader(self, batch_size=None, num_workers=None):
        if num_workers is None: num_workers = 3 if self.distributed else 4
        if batch_size is None: batch_size = self.batch_size
        valid_split = AsocaDataset(ds_path=self.data_dir, split='valid')
        sampler = ASOCASampler(valid_split.vol_meta, perc_per_epoch=self.perc_per_epoch_val)
        if self.distributed: sampler = DistributedSamplerWrapper(sampler=sampler)
        return DataLoader(valid_split, sampler=sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    def volume_dataloader(self, vol_id, batch_size=None):
        if batch_size is None: batch_size = self.batch_size
        ds = AsocaVolumeDataset(ds_path=self.data_dir, vol_id=vol_id)
        meta = ds.get_vol_meta()
        return DataLoader(ds, batch_size=batch_size, num_workers=12, pin_memory=True), meta
