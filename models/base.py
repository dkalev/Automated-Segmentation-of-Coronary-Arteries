import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
import pytorch_lightning as pl
from loss import DiceBCELoss

class Base(pl.LightningModule):
    def __init__(self, *args, lr=1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.crit = DiceBCELoss()
        self.accuracy = pl.metrics.Accuracy()
        self.f1 = pl.metrics.F1()
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, targs = batch
        preds = self(x)
        loss = self.crit(preds, targs)
        self.log_metrics(preds, targs, loss)
        return loss
   
    def validation_step(self, batch, batch_idx):
        x, targs = batch
        preds = self(x)
        loss = self.crit(preds, targs)
        self.log_metrics(preds, targs, loss, split='valid')
    
    def log_metrics(self, preds, targs, loss, split='train'):
        preds = torch.sigmoid(preds)
        self.log(f'{split}_loss', loss)
        self.log(f'{split}_acc', self.accuracy(preds, targs))
        self.log(f'{split}_f1', self.f1(preds, targs))
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

        self.model = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(4),
            nn.ReLU(inplace=True),

            nn.Conv3d(4, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.Conv3d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(4),
            nn.ReLU(inplace=True),

            nn.Conv3d(4, 1, kernel_size=3, padding=1, bias=False),
        )
    
    def forward(self, x):
        return self.model(x)
    