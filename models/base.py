import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from loss import DiceBCELoss, DiceLoss, DiceBCE_OHNMLoss
from metrics import dice_score, roc_auc


class Base(pl.LightningModule):
    def __init__(self, *args, lr=1e-3, loss_type='dice', skip_empty_patches=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.crit = self.get_loss_func(loss_type)
        self.f1 = pl.metrics.F1()
        self.lr = lr
        self.skip_empty_patches = skip_empty_patches

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
        x, targs = batch
        targs = self.crop_data(targs)
        if self.skip_empty_patches and split == 'train':
            mask = self.get_empty_patch_mask(targs)
            x, targs = x[mask], targs[mask]
        return x, targs

    def training_step(self, batch, batch_idx):
        x, targs = self.prepare_batch(batch, 'train')
        if len(x) == 0: return

        preds = self(x)
        if isinstance(preds, torch.Tensor):
            preds = self.crop_preds(preds, targs)
            loss = self.crit(preds, targs)
        else:
            preds = [ self.crop_preds(pred, targs) for pred in preds ]
            losses = torch.stack([ self.crit(pred, targs) for pred in preds ])
            if hasattr(self, 'ds_weight'):
                loss = losses @ self.ds_weight
            else:
                loss = losses.sum()

        self.log_metrics(preds, targs, loss)
        return loss
   
    def validation_step(self, batch, batch_idx):
        x, targs = self.prepare_batch(batch, 'valid')
        if len(x) == 0: return

        preds = self(x)
        if isinstance(preds, torch.Tensor):
            preds = self.crop_preds(preds, targs)
        else:
            preds = self.crop_preds(preds[-1], targs)
        loss = self.crit(preds, targs)
        self.log_metrics(preds, targs, loss, split='valid')
    
    def log_metrics(self, preds, targs, loss, split='train'):
        if isinstance(preds, torch.Tensor):
            preds = torch.sigmoid(preds)
        else:
            preds = torch.sigmoid(preds[-1])
        preds = preds.round()
        self.log(f'{split}_loss', loss.item())
        self.log(f'{split}_f1', self.f1(preds, targs).item())
        if split == 'valid':
            self.log(f'valid_dice', dice_score(preds, targs), prog_bar=True)
            self.log(f'valid_auc', roc_auc(preds, targs))

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
