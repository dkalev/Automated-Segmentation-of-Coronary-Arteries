from dataclasses import dataclass
from typing import Tuple

from omegaconf import DictConfig, OmegaConf, MISSING
from hydra.core.config_store import ConfigStore
import hydra


@dataclass
class DatasetConfig:
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
class TrainConfig:
    n_epochs: int = 100
    batch_size: int = 1
    lr: float = 0.001
    gpus: int = 1
    optim_type: str = 'adam'
    sample_every_epoch: bool = True

@dataclass
class TrainClassificationConfig(TrainConfig):
    pass

@dataclass
class TrainSegmentationConfig(TrainConfig):
    oversample: bool = True
    weight_update_step: float = 0.
    loss_type: str = 'dice'
    ohnm_ratio: int = 100
    fast_val: bool = False
    skip_empty_patches: bool = False

@dataclass
class ModelConfig:
    kernel_size: int = 3

@dataclass
class SteerableConfig:
    repr_type: str = 'spherical'

@dataclass
class GridKwargs:
    type: str = 'cube'

@dataclass
class FTConfig:
    grid_kwargs: GridKwargs

# seg
# cnn
@dataclass
class SegCNNConfig(ModelConfig):
    pass

# unet
@dataclass
class SegUnetConfig(ModelConfig):
    pass
# eunet
@dataclass
class SegUnetConfig(ModelConfig):
    pass
# ftunet
# cubeunet
# icounet

#class
# cnn
# cube
# ico
# gated 
# ft

@dataclass
class Config:
    dataset: DatasetConfig
    model: str
    train: TrainConfig

cs = ConfigStore.instance()
cs.store(name="config_n", node=Config)
cs.store(group="dataset", name="classification_n", node=AsocaClassificationConfig)
cs.store(group="dataset", name="segmentation_n", node=AsocaSegmentationConfig)
cs.store(group="train", name="classification_n", node=TrainClassificationConfig)
cs.store(group="train", name="segmentation_n", node=TrainSegmentationConfig)

def get_class(cls):
    parts = cls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)            
    return m

@hydra.main(config_path='config/hydra', config_name='config')
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(get_class(cfg.model.class_name)(**cfg.model.params))

if __name__ == "__main__":
    my_app()

