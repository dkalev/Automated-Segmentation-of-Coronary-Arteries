import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
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
    
    def log(self, name: str, value, *args, commit=False, batch_idx=None, **kwargs):
        wandb.log({name: value, 'batch_idx': batch_idx}, commit=commit)
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
        hmasks = self.crop_data(hmasks)
        hmasks[hmasks==0] = -1
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
        x, targs, h_masks = self.prepare_batch(batch, 'train')
        if len(x) == 0: return

        preds = self(x)
        if isinstance(preds, torch.Tensor):
            preds = self.crop_preds(preds, targs)
            if self.mask_heart: preds = preds * h_masks
            loss = self.crit(preds, targs)
        else:
            preds = [ self.crop_preds(pred, targs) * h_masks for pred in preds ]
            losses = torch.stack([ self.crit(pred, targs) for pred in preds ])
            if hasattr(self, 'ds_weight'):
                loss = losses @ self.ds_weight
            else:
                loss = losses.mean()

        return { 'batch_idx': batch_idx, 'loss': loss, 'preds': preds, 'targs': targs }
    
    def training_step_end(self, outs):
        batch_idx, preds, targs = outs['batch_idx'], outs['preds'], outs['targs']
        preds = self.apply_nonlinearity(preds)

        self.log(f'train/loss', outs['loss'].item(), batch_idx=batch_idx)
        self.log(f'train/dice', dice_score(preds, targs).item(), batch_idx=batch_idx, prog_bar=True)
        self.log(f'train/f1', self.train_f1(preds, targs).item(), batch_idx=batch_idx, commit=True)

    def training_epoch_end(self, outs):
        self.log("train/f1_epoch", self.train_f1.compute(), commit=True)

    def validation_step(self, batch, batch_idx):
        x, targs, h_masks = self.prepare_batch(batch, 'valid')
        preds = self(x)
        if isinstance(preds, torch.Tensor):
            preds = self.crop_preds(preds, targs)
        else:
            preds = self.crop_preds(preds[-1], targs)
        if self.mask_heart: preds = preds * h_masks
        loss = self.crit(preds, targs)

        return { 'batch_idx': batch_idx, 'loss': loss, 'preds': preds, 'targs': targs }

    def validation_step_end(self, outs):
        batch_idx, preds, targs = outs['batch_idx'], outs['preds'], outs['targs']
        preds = self.apply_nonlinearity(preds)

        valid_f1 = self.valid_f1(preds, targs).item()
        valid_iou = self.iou(preds, targs).item()
        valid_loss = outs['loss'].item()
        self.log(f'valid/loss', valid_loss, batch_idx=batch_idx)
        self.log(f'valid/dice', dice_score(preds, targs).item(), batch_idx=batch_idx, prog_bar=True)
        self.log(f'valid/f1', valid_f1, batch_idx=batch_idx)
        self.log(f'valid/iou', valid_iou, batch_idx=batch_idx, commit=True)
        return { 'loss': valid_loss, 'f1': valid_f1, 'iou': valid_iou }

    def validation_epoch_end(self, outs):
        epoch_f1 = np.array([ b['f1'] for b in outs ]).mean()
        epoch_iou = np.array([ b['iou'] for b in outs ]).mean()
        self.log("valid/f1_epoch", epoch_f1)
        self.log("valid/iou_epoch", epoch_iou, commit=True)
    

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
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
            {'in_channels':1, 'out_channels': 4 },
            {'in_channels':4, 'out_channels': 16 },
            {'in_channels':16, 'out_channels': 32 },
            {'in_channels':32, 'out_channels': 4 },
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
