import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
import torchmetrics
from loss import DiceBCELoss, DiceLoss, DiceBCE_OHNMLoss
from metrics import dice_score
import wandb

class Base(pl.LightningModule):
    def __init__(self, *args,
                        lr=1e-3,
                        loss_type='dice',
                        skip_empty_patches=False,
                        mask_heart=False,
                        optim_type='adam',
                        **kwargs):
        super().__init__(*args, **kwargs)
        self.crit = self.get_loss_func(loss_type)
        f1 = torchmetrics.F1()
        self.train_f1 = f1.clone()
        self.valid_f1 = f1.clone()
        self.iou = torchmetrics.IoU(num_classes=2)
        self.lr = lr
        self.skip_empty_patches = skip_empty_patches
        self.mask_heart = mask_heart

        if optim_type in ['adam', 'sgd']:
            self.optim_type = optim_type
        else:
            raise ValueError(f'Unsupported optimizer type: {optim_type}')

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

    def log(self, name: str, value, *args, commit=False, **kwargs):
        split = 'train' if 'train' in name else 'valid'
        wandb.log({
            name: value,
            'iter': getattr(self, f'{split}_iter'),
            'epoch': self.current_epoch
        }, commit=commit)
        return super().log(name, value, *args, **kwargs)

    @staticmethod
    def get_loss_func(name):
        if name == 'bce':
            return nn.BCEWithLogitsLoss()
        elif name == 'dice':
            return DiceLoss()
        elif name == 'gdice':
            return DiceLoss(generalized=True)
        elif name == 'dicebce':
            return DiceBCELoss()
        elif name == 'gdicebce':
            return DiceBCELoss(dice_kwargs={'generalized':True})
        elif name == 'dicebceohnm':
            return DiceBCE_OHNMLoss()
        elif name == 'gdicebceohnm':
            return DiceBCE_OHNMLoss(dice_kwargs={'generalized':True})
        else:
            raise ValueError(f'Unknown loss type: {name}')

    def crop_data(self, data):
        data = data[..., # [batch_size, n_channels]
                     self.crop:-self.crop, # x
                     self.crop:-self.crop, # y
                     self.crop:-self.crop] # z

        return data

    def crop_preds(self, preds, targs):
        if preds.shape == targs.shape:
            return preds
        else:
            return self.crop_data(preds)

    @staticmethod
    def get_empty_patch_mask(targs):
        return targs.sum(dim=[1,2,3,4]) > 0

    def prepare_batch(self, batch, split='train'):
        # crops targets to match the padding lost in the convolutions
        x, targs, hmasks = batch
        targs = self.crop_data(targs)
        if hmasks is None:
            self.mask_heart = False
        else:
            hmasks = self.crop_data(hmasks)
        if self.skip_empty_patches and split == 'train':
            non_empty = self.get_empty_patch_mask(targs)
            x, targs, hmasks = x[non_empty], targs[non_empty], hmasks[non_empty]
        return x, targs, hmasks

    def apply_nonlinearity(self, preds):
        if isinstance(preds, torch.Tensor):
            preds = torch.sigmoid(preds)
        else:
            preds = torch.sigmoid(preds[-1])
        preds = preds.round()
        return preds

    def training_step(self, batch, batch_idx):
        self.train_iter += 1
        x, targs, h_masks = self.prepare_batch(batch, 'train')
        if len(x) == 0: return

        preds = self(x)
        if isinstance(preds, torch.Tensor):
            preds = self.crop_preds(preds, targs)
            if self.mask_heart: preds = preds.masked_fill(h_masks==0, -10)
            loss = self.crit(preds, targs)
        else:
            preds = [ self.crop_preds(pred, targs) for pred in preds ]
            if self.mask_heart:
                preds = [ pred.masked_fill(h_masks==0, -10) for pred in preds ]
            losses = torch.stack([ self.crit(pred, targs) for pred in preds ])
            if hasattr(self, 'ds_weight'):
                loss = losses @ self.ds_weight
            else:
                loss = losses.mean()

        return { 'batch_idx': batch_idx, 'loss': loss, 'preds': preds, 'targs': targs }

    def training_step_end(self, outs):
        if outs is None: return
        batch_idx, preds, targs = outs['batch_idx'], outs['preds'], outs['targs']
        preds = self.apply_nonlinearity(preds)

        self.log(f'train/loss', outs['loss'].item())
        self.log(f'train/dice', dice_score(preds, targs).item(), prog_bar=True)
        self.log(f'train/f1', self.train_f1(preds.flatten().int(), targs.flatten()).item(), commit=True)
        return outs['loss']

    def training_epoch_end(self, outs):
        self.log("train/f1_epoch", self.train_f1.compute(), commit=True)

    def validation_step(self, batch, batch_idx):
        self.valid_iter += 1
        x, targs, h_masks = self.prepare_batch(batch, 'valid')
        preds = self(x)
        if isinstance(preds, torch.Tensor):
            preds = self.crop_preds(preds, targs)
        else:
            preds = self.crop_preds(preds[-1], targs)
        if self.mask_heart: preds = preds.masked_fill(h_masks==0, -10)
        loss = self.crit(preds, targs)

        return { 'batch_idx': batch_idx, 'loss': loss, 'preds': preds, 'targs': targs }

    def validation_step_end(self, outs):
        preds, targs = outs['preds'], outs['targs']
        preds = self.apply_nonlinearity(preds)
        return { 'loss': outs['loss'].item(),
                 'intersection': torch.sum(preds * targs).item(),
                 'denom': preds.sum().item() + targs.sum().item() }

    def validation_epoch_end(self, outs):
        loss  = np.mean([out['loss'] for out in outs])
        loss_std = np.std([out['loss'] for out in outs])
        intersection = np.sum([out['intersection'] for out in outs])
        denom = np.sum([out['denom'] for out in outs])
        dice_score = (2 * intersection + 1e-10) / (denom + 1e-10)
        iou_score = intersection / (denom - intersection)
        self.log(f'valid/loss', loss)
        self.log(f'valid/loss_std', loss_std)
        self.log(f'valid/dice', dice_score)
        self.log(f'valid/iou', iou_score, commit=True)

    def configure_optimizers(self):
        if self.optim_type == 'adam':
            optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        elif self.optim_type == 'sgd':
            optimizer = SGD(self.parameters(), lr=self.lr, momentum=0.99, nesterov=True, weight_decay=1e-5)
        return {
            'optimizer': optimizer,
            'scheduler': ReduceLROnPlateau(optimizer, 'min', patience=5),
            'monitor': 'valid_loss'
        }


class Baseline3DCNN(Base):
    def __init__(self, *args, kernel_size=5, **kwargs):
        super().__init__(*args, **kwargs)

        common_params = {
            'kernel_size': kernel_size,
            'bias': False,
        }

        block_params = [
            {'in_channels':1, 'out_channels': 16 },
            {'in_channels':16, 'out_channels': 32 },
            {'in_channels':32, 'out_channels': 64 },
            {'in_channels':64, 'out_channels': 16 },
        ]

        blocks = [
            nn.Sequential(
                nn.Conv3d(**b_params, **common_params),
                nn.BatchNorm3d(b_params['out_channels']),
                nn.ReLU(inplace=True),
            ) for b_params in block_params
        ]

        blocks.append(
            nn.Conv3d(block_params[-1]['out_channels'], 1, kernel_size=common_params['kernel_size'])
        )

        self.crop = (common_params['kernel_size']//2) * len(blocks)

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
