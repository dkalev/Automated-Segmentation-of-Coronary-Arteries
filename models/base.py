import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from loss import DiceBCELoss, DiceLoss, DiceBCE_OHNMLoss
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from functools import partial
from metrics import dice_score, hausdorff_95
from collections import defaultdict
from data_utils.helpers import get_volume_pred
import wandb


def get_volume(vol_id, vol, vol_meta, fn):
    return vol_id, fn(vol, vol_meta)

def compute_vol_metrics(data, loss_fn):
    vol_id, pred, targ, spacing = data
    loss = loss_fn(torch.from_numpy(pred), torch.from_numpy(targ)).item()
    dice = dice_score(torch.from_numpy(pred), torch.from_numpy(targ)).item()
    # don't compute hausdorff_95 if more than 50% of the predictions are positive
    # as it is too slow and the model is not doing much anyway
    # don't compute if all the preds are zero either
    hd95 = hausdorff_95(pred, targ, spacing) if pred.mean() < 0.5 and pred.sum() > 0 else np.inf
    return { 'vol_id': vol_id, 'loss': loss, 'dice': dice, 'hd95': hd95 }

class Base(pl.LightningModule):
    def __init__(self, *args,
                        lr=1e-3,
                        loss_type='dice',
                        skip_empty_patches=False,
                        mask_heart=False,
                        optim_type='adam',
                        ds_meta=None,
                        **kwargs):
        super().__init__(*args, **kwargs)
        self.crit = self.get_loss_func(loss_type)
        self.lr = lr
        self.skip_empty_patches = skip_empty_patches
        self.mask_heart = mask_heart

        assert ds_meta is not None
        self.ds_meta = ds_meta

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
        vol_ids, x, targs, hmasks = batch
        targs = self.crop_data(targs)
        if hmasks is None:
            self.mask_heart = False
        else:
            hmasks = self.crop_data(hmasks)
        if self.skip_empty_patches and split == 'train':
            non_empty = self.get_empty_patch_mask(targs)
            x, targs, hmasks = x[non_empty], targs[non_empty], hmasks[non_empty]

        if split == 'train':
            return x, targs, hmasks
        else:
            return vol_ids, x, targs, hmasks

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

        return { 'loss': loss, 'preds': preds, 'targs': targs }

    def training_step_end(self, outs):
        if outs is None: return
        preds, targs = outs['preds'], outs['targs']
        preds = self.apply_nonlinearity(preds)

        self.log(f'train/loss', outs['loss'].item())
        self.log(f'train/dice', dice_score(preds, targs).item(), prog_bar=True, commit=True)
        return outs['loss']

    def validation_step(self, batch, batch_idx):
        self.valid_iter += 1
        vol_ids, x, targs, h_masks = self.prepare_batch(batch, 'valid')
        preds = self(x)
        if isinstance(preds, torch.Tensor):
            preds = self.crop_preds(preds, targs)
        else:
            preds = self.crop_preds(preds[-1], targs)
        if self.mask_heart: preds = preds.masked_fill(h_masks==0, -10)

        return { 'vol_ids': vol_ids, 'preds': preds, 'targs': targs }

    def validation_step_end(self, outs):
        outs['vol_ids'] = outs['vol_ids'].cpu().numpy().astype(np.uint8)
        outs['preds'] = self.apply_nonlinearity(outs['preds'])
        outs['preds'] = outs['preds'].cpu().numpy().astype(np.uint8)
        outs['targs'] = outs['targs'].cpu().numpy().astype(np.uint8)
        return outs

    def validation_epoch_end(self, outs):
        # skip if validation sanity check
        if len(outs) == 2: return

        # gather all patches per volume
        preds = defaultdict(list)
        targs = defaultdict(list)
        for batch in outs:
            for i, vol_id in enumerate(batch['vol_ids']):
                preds[vol_id].append(batch['preds'][i])
                targs[vol_id].append(batch['targs'][i])
        
        preds = { vol_id: np.concatenate(batches) for vol_id, batches in preds.items() } 
        targs = { vol_id: np.concatenate(targs[vol_id]) for vol_id in preds } 
        vol_metas = { vol_id: self.ds_meta['vol_meta'][str(vol_id)] for vol_id in preds }

        # all the magic here is done to ensure there are no reference to self
        # in the functions used in the process pool and hence the base class
        # otherwise the multiprocessing fails when spawning new process due to some issue with cuda
        # ex: https://discuss.pytorch.org/t/runtimeerror-cannot-re-initialize-cuda-in-forked-subprocess-to-use-cuda-with-multiprocessing-you-must-use-the-spawn-start-method/14083
        get_volume_fn = partial(get_volume_pred,
                        patch_size=self.ds_meta['stride'],
                        stride=self.ds_meta['stride'],
                        normalize=False)

        get_volume_fn = partial(get_volume, fn=get_volume_fn)
        compute_vol_metrics_fn = partial(compute_vol_metrics, loss_fn=self.crit)

        with ProcessPoolExecutor(max_workers=4) as exec:
            # build prediction volumes from patches
            pred_vols = dict(tqdm(
                exec.map(get_volume_fn, preds.keys(), preds.values(), vol_metas.values()),
                total=len(preds), position=2, desc='Building preds')
            )
            # build target volumes from patches
            targ_vols = dict(tqdm(
                exec.map(get_volume_fn, targs.keys(), targs.values(), vol_metas.values()),
                total=len(targs), position=2, desc='Building targs')
            )

            # gather all the data needed to compute the metrics
            # ensure that the order of volumes is correct for preds, targs and spacings
            data = [ (vol_id, pred_vols[vol_id], targ_vols[vol_id], vol_metas[vol_id]['orig_spacing'] )
                        for vol_id in pred_vols ]

            # compute metrics for each volume
            metrics = list(tqdm(
                exec.map(compute_vol_metrics_fn, data),
                total=len(data), position=2, desc='Computing metrics')
            )

        # finally aggregate accross volumes and log
        self.log(f'valid/loss', np.mean([v['loss'] for v in metrics]))
        self.log(f'valid/dice', np.mean([v['dice'] for v in metrics]))
        self.log(f'valid/hd95', np.mean([v['hd95'] for v in metrics]), commit=True)

    def configure_optimizers(self):
        if self.optim_type == 'adam':
            optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        elif self.optim_type == 'sgd':
            optimizer = SGD(self.parameters(), lr=self.lr, momentum=0.99, nesterov=True, weight_decay=1e-5)
        return {
            'optimizer': optimizer,
            'scheduler': ReduceLROnPlateau(optimizer, 'min', patience=5),
            'monitor': 'valid/loss'
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
