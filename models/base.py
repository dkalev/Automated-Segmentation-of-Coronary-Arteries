import torch.distributed as dist
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from typing import Iterable


class Base(pl.LightningModule):
    """BasePL Additional helpers for a LightningModule facilitating logging with wandb"""

    @property
    def train_iter(self):
        if not hasattr(self, '_train_iter'):
            self._train_iter = 0
        return self._train_iter

    @train_iter.setter
    def train_iter(self, val):
        self._train_iter = self._train_iter + 1 if val is None else val

    @property
    def valid_iter(self):
        if not hasattr(self, '_valid_iter'):
            self._valid_iter = 0
        return self._valid_iter

    @valid_iter.setter
    def valid_iter(self, val=None):
        self._valid_iter = self._valid_iter + 1 if val is None else val

    def get_lr(self):
        return self.trainer.lr_schedulers[0]['scheduler'].optimizer.param_groups[0].get('lr')
    
    def parse_padding(self, padding, kernel_size):
        if isinstance(padding, int):
            return tuple(3*[padding])
        elif isinstance(padding, Iterable) and len(padding) == 3 and all(type(p)==int for p in padding):
            return tuple(padding)
        elif padding == 'same':
            return tuple(3*[kernel_size // 2])
        else:
            raise ValueError(f'Parameter padding must be int, tuple, or "same. Given: {padding}')

    def get_sampler(self, split='train'):
        if split == 'train':
            sampler = self.trainer.train_dataloader.sampler
        elif split == 'valid':
            sampler = self.trainer.val_dataloaders[0].sampler
        else:
            raise ValueError(f'split must be one of ["train", "valid"], given: {split}')
        # wrapped in distributed sampler
        if dist.is_initialized(): sampler = sampler.sampler
        return sampler

    def configure_optimizers(self, max_epochs=None):
        if self.optim_type == 'adam':
            optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        elif self.optim_type == 'sgd':
            optimizer = SGD(self.parameters(), lr=self.lr, momentum=0.99, nesterov=True, weight_decay=1e-5)
        else:
            raise ValueError(f'Optimizer must be one of [adam, sgd], given: {self.optim_type}')
        max_epochs = max_epochs or self.trainer.max_epochs
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': LambdaLR(optimizer, lambda epoch: (1 - epoch/max_epochs)**0.9),
            }
        }
