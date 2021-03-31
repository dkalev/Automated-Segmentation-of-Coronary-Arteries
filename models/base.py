import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
import pytorch_lightning as pl
from loss import DiceBCELoss, DiceLoss

class Base(pl.LightningModule):
    def __init__(self, *args, lr=1e-3, loss_type='dice', **kwargs):
        super().__init__(*args, **kwargs)
        self.crit = self.get_loss_func(loss_type)
        self.accuracy = pl.metrics.Accuracy()
        self.f1 = pl.metrics.F1()
        self.lr = lr
    
    @staticmethod
    def get_loss_func(name):
        if name == 'bce':
            return nn.BCEWithLogitsLoss()
        elif name == 'dice':
            return DiceLoss()
        elif name == 'dicebce':
            return DiceBCELoss()
        else:
            raise ValueError(f'Unknown loss type: {name}')


    @staticmethod
    def mask_targs(targs):
        return targs, torch.ones(targs.shape[0])
    
    def prepare_batch(self, batch):
        # crops targets to match the padding lost in the convolutions
        # removes datapoints with no positive samples (after cropping) from batch
        x, targs = batch
        targs, mask = self.mask_targs(targs)
        x = x[mask,...]
        return x, targs

    def training_step(self, batch, batch_idx):
        x, targs = self.prepare_batch(batch)
        if len(x) == 0: return

        preds = self(x)
        loss = self.crit(preds, targs)
        self.log_metrics(preds, targs, loss)
        return loss
   
    def validation_step(self, batch, batch_idx):
        x, targs = self.prepare_batch(batch)
        if len(x) == 0: return

        preds = self(x)
        loss = self.crit(preds, targs)
        self.log_metrics(preds, targs, loss, split='valid')
    
    def log_metrics(self, preds, targs, loss, split='train'):
        preds = torch.sigmoid(preds)
        self.log(f'{split}_loss', loss.item())
        self.log(f'{split}_acc', self.accuracy(preds, targs).item())
        self.log(f'{split}_f1', self.f1(preds, targs).item())
        if split == 'valid':
            targs_numpy = targs.cpu().flatten().numpy().astype(int)
            preds_numpy = preds.cpu().flatten()
            self.log(f'valid_auc', roc_auc_score(targs_numpy, preds_numpy))

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            'scheduler': ReduceLROnPlateau(optimizer, 'min', patience=5),
            'monitor': 'valid_loss'
        }
class Baseline3DCNN(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        common_params = {
            'kernel_size': 3,
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
            nn.Conv3d(4, 1, kernel_size=common_params['kernel_size'])
        )

        self.crop = (common_params['kernel_size']//2) * len(blocks)

        self.model = nn.Sequential(*blocks)
    
    def forward(self, x):
        return self.model(x)

    def mask_targs(self, targs):
        targs = targs[..., # [batch_size, n_channels]
                     self.crop:-self.crop, # x
                     self.crop:-self.crop, # y
                     self.crop:-self.crop] # z

        mask = targs.sum(dim=[1,2,3,4]) > 0
        return targs[mask,...], mask