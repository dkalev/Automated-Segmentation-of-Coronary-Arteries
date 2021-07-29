from dataclasses import dataclass
from models.segmentation.base import BaseSegmentation
from typing import Tuple
import pytorch_lightning as plt
import torch.distributed as dist
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from pathlib import Path
import json

from omegaconf import DictConfig, OmegaConf, MISSING
from hydra.core.config_store import ConfigStore
import hydra

import torch
import random
import numpy as np


@dataclass
class DatasetConfig:
    class_name: str = MISSING
    perc_per_epoch_train: float = 1.0
    perc_per_epoch_val: float = 1.0
    sample_every_epoch: bool = True
    data_dir: str = MISSING
    sourcepath: str = MISSING


@dataclass
class AsocaClassificationConfig(DatasetConfig):
    patch_size: Tuple[int,int,int] = (68,68,68)
    data_dir: str = 'dataset/classification'
    sourcepath: str = 'dataset/ASOCA2020Data.zip'

@dataclass
class AsocaSegmentationConfig(DatasetConfig):
    patch_size: Tuple[int,int,int] = (128,128,128)
    patch_stride: Tuple[int,int,int] = (108,108,108)
    normalize: bool = True
    data_clip_range: str = 'percentile'
    num_workers: int = 4
    resample_vols: bool = False
    oversample: bool = True
    weight_update_step: float = .0
    crop_empty: bool = False
    data_dir: str = 'dataset/classification'
    sourcepath: str = 'dataset/ASOCA2020Data.zip'

@dataclass
class TrainDatasetConfig:
    batch_size: int = 1
    sample_every_epoch: bool = True

@dataclass
class TrainClassDatasetConfig(TrainDatasetConfig):
    pass

@dataclass
class TrainSegDatasetConfig(TrainDatasetConfig):
    oversample: bool = True
    weight_update_step: float = 0.
    ohnm_ratio: int = 100
    fast_val: bool = False
    skip_empty_patches: bool = False

@dataclass
class TrainerConfig:
    gpus: 4
    n_epochs: 100

@dataclass
class TrainModelConfig:
    lr: 0.001
    optim_type: str = 'adam'

@dataclass
class TrainClassModelConfig(TrainModelConfig):
    pass

@dataclass
class TrainSegModelConfig(TrainModelConfig):
    loss_type: str = 'dice'

@dataclass
class TrainConfig:
    dataset: TrainDatasetConfig
    trainer: TrainerConfig
    model: TrainModelConfig

@dataclass
class Config:
    dataset: DatasetConfig
    model: str
    train: TrainConfig

cs = ConfigStore.instance()
cs.store(name="config_n", node=Config)
cs.store(group="dataset", name="classification_n", node=AsocaClassificationConfig)
cs.store(group="dataset", name="segmentation_n", node=AsocaSegmentationConfig)
cs.store(group="train", name="classification_n", node=TrainConfig)

def get_class(cls):
    parts = cls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)            
    return m

def print_config(cfg: DictConfig):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(OmegaConf.to_yaml(cfg))

def use_multigpu(cfg: DictConfig) -> bool:
    gpus = cfg.train.trainer.gpus
    return (isinstance(gpus, int) and gpus > 1) or (isinstance(gpus, list) and len(gpus) > 1)

@hydra.main(config_path='config/hydra', config_name='config')
def train(cfg: DictConfig):
    if 'equivariant' in cfg.model.class_name and cfg.debug == True:
        cfg.model.params.initialize = False

    print_config(cfg)

    # set seeds for reproducibility
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    dm_params = { **cfg.train.dataset, **cfg.dataset.params }
    dm: plt.LightningDataModule = get_class(cfg.dataset.class_name)(**dm_params)

    model_params = { 'debug': cfg.debug, **cfg.train.model, **cfg.model.params }
    model: plt.LightningModule = get_class(cfg.model.class_name)(**model_params)

    dm.prepare_data()

    if isinstance(model, BaseSegmentation):
        with open(Path(cfg.dataset.params.data_dir, 'dataset.json'), 'r') as f:
            model.ds_meta = json.load(f)

    logger = WandbLogger() if not cfg.debug else None
    trainer_params = {
        'accelerator': 'ddp' if use_multigpu(cfg) else None,
        # disable logging in debug mode
        'checkpoint_callback': True if not cfg.debug else False,
        'logger': logger,
        'gradient_clip_val': 12,
        'callbacks': None if cfg.debug else [ ModelCheckpoint(monitor='valid/loss', mode='min') ],
        'plugins': DDPPlugin(find_unused_parameters=False) if use_multigpu(cfg) else None,
        'replace_sampler_ddp': False,
        'num_sanity_val_steps': 0,
        'reload_dataloaders_every_epoch': True if use_multigpu(cfg) else False,
        **cfg.train.trainer
    }
    trainer = plt.Trainer( **trainer_params)

    if logger:
        params = OmegaConf.to_object(cfg)
        params['dataset'] = { 'datamodule': params['dataset']['class_name'], **params['dataset']['params'] }
        params['train'] = { k:v for subdict in params['train'].values() for k,v in subdict.items() }
        params['model'] = { 'model': params['model']['class_name'], **params['model']['params'] }
        logger.log_hyperparams(params)
        logger.watch(model)

    trainer.fit(model, dm)

if __name__ == "__main__":
    train()

