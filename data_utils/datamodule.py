from data_utils.distributed_sampler import DistributedSamplerWrapper
import shutil
import logging
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import torch.distributed as dist

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

    def sync_samplers(self, sampler, split):
        # ensure that each process trains and validates on the same subset of files in ddp on each gpu
        # otherwise because a sampler is initialized in each process 
        # we end up with partial predictions for e.g. 4 volumes instead of
        # full predictions (all patches) for the 2 required volumes
        package = [sampler.sample_ids()]
        dist.barrier()
        # broadcast sends the package object from the specified rank (0) and replaces it on all other ranks
        dist.broadcast_object_list(package, 0)

        sampler.file_ids = package[0]
        dataset = AsocaDataset(ds_path=self.data_dir, split=split)
        dataset.file_ids = package[0]

        return dataset, sampler

    def train_dataloader(self, batch_size=None, num_workers=None):
        if num_workers is None: num_workers = 3 if dist.is_initialized() else 4
        if batch_size is None: batch_size = self.batch_size
        if self.trainer.current_epoch == 0:
            train_ds = AsocaDataset(ds_path=self.data_dir, split='train')
            sampler = ASOCASampler(train_ds.vol_meta,
                                    oversample=self.oversample,
                                    perc_per_epoch=self.perc_per_epoch_train)
        elif self.trainer.current_epoch > 0 and dist.is_initialized():
            sampler = self.trainer.train_dataloader.sampler.sampler

        if dist.is_initialized():
            train_ds, sampler = self.sync_samplers(sampler, 'train')
            sampler = DistributedSamplerWrapper(sampler=sampler, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        return DataLoader(train_ds, sampler=sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    def val_dataloader(self, batch_size=None, num_workers=None):
        if num_workers is None: num_workers = 3 if dist.is_initialized() else 4
        if batch_size is None: batch_size = self.batch_size
        if self.trainer.current_epoch == 0:
            valid_ds = AsocaDataset(ds_path=self.data_dir, split='valid')
            sampler = ASOCASampler(valid_ds.vol_meta, perc_per_epoch=self.perc_per_epoch_val)
        elif self.trainer.current_epoch > 0 and dist.is_initialized():
            sampler = self.trainer.val_dataloaders[0].sampler.sampler

        if dist.is_initialized():
            valid_ds, sampler = self.sync_samplers(sampler, 'valid')
            sampler = DistributedSamplerWrapper(sampler=sampler, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        return DataLoader(valid_ds, sampler=sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    def volume_dataloader(self, vol_id, batch_size=None):
        if batch_size is None: batch_size = self.batch_size
        ds = AsocaVolumeDataset(ds_path=self.data_dir, vol_id=vol_id)
        meta = ds.get_vol_meta()
        return DataLoader(ds, batch_size=batch_size, num_workers=12, pin_memory=True), meta
