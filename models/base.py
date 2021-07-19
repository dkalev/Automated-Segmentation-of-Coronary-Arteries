import torch
import torch.distributed as dist
import numpy as np
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from loss import DiceBCELoss, BCEWrappedLoss, DiceLoss, DiceBCE_OHNMLoss
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from tqdm import tqdm
from metrics import dice_score, hausdorff_95
from collections import defaultdict
from data_utils.helpers import get_volume_pred
from typing import Iterable
from itertools import repeat
import wandb


def get_volume(vol_id, patches, vol_meta, patch_size, stride):
    try:
        res = get_volume_pred(patches, vol_meta, patch_size, stride, normalize=False)
        return vol_id, res
    except Exception as e:
        print(e)

def compute_vol_metrics(data):
    vol_id, pred, targ, spacing = data
    dice = dice_score(torch.from_numpy(pred), torch.from_numpy(targ)).item()
    # don't compute hausdorff_95 if more than 50% of the predictions are positive
    # as it is too slow and the model is not doing much anyway
    # don't compute if all the preds are zero either
    hd95 = hausdorff_95(pred, targ, spacing) if pred.mean() < 0.5 and pred.sum() > 0 else np.inf
    return { 'vol_id': vol_id, 'dice': dice, 'hd95': hd95 }


class BasePL(pl.LightningModule):
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


class Base(BasePL):
    def __init__(self, *args,
                        lr=1e-3,
                        loss_type='dice',
                        skip_empty_patches=False,
                        mask_heart=False,
                        optim_type='adam',
                        debug=False,
                        fast_val=False,
                        **kwargs):
        super().__init__(*args, **kwargs)
        self.crit = self.get_loss_func(loss_type)
        self.lr = lr
        self.skip_empty_patches = skip_empty_patches
        self.mask_heart = mask_heart

        self.debug = debug
        self.fast_val = fast_val

        if optim_type in ['adam', 'sgd']:
            self.optim_type = optim_type
        else:
            raise ValueError(f'Unsupported optimizer type: {optim_type}')

    @property
    def ds_meta(self):
        return self._ds_meta

    @ds_meta.setter
    def ds_meta(self, ds_meta):
        self._ds_meta = ds_meta

    def parse_padding(self, padding, kernel_size):
        if isinstance(padding, int):
            return tuple(3*[padding])
        elif isinstance(padding, Iterable) and len(padding) == 3 and all(type(p)==int for p in padding):
            return tuple(padding)
        elif padding == 'same':
            return tuple(3*[kernel_size // 2])
        else:
            raise ValueError(f'Parameter padding must be int, tuple, or "same. Given: {padding}')

    @staticmethod
    def get_loss_func(name):
        if name == 'bce':
            return BCEWrappedLoss()
        elif name == 'dice':
            return DiceLoss()
        elif name == 'gdice':
            return DiceLoss(generalized=True)
        elif name == 'dicebce':
            return DiceBCELoss()
        elif name == 'gdicebce':
            return DiceBCELoss(generalized=True)
        elif name == 'dicebceohnm':
            return DiceBCE_OHNMLoss()
        elif name == 'gdicebceohnm':
            return DiceBCE_OHNMLoss(generalized=True)
        else:
            raise ValueError(f'Unknown loss type: {name}')

    def crop_data(self, data):
        data = data[..., # [batch_size, n_channels]
                     self.crop[0]:-self.crop[0], # x
                     self.crop[1]:-self.crop[1], # y
                     self.crop[2]:-self.crop[2]] # z

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
        x, targs, (vol_ids, patch_ids) = batch
        targs = self.crop_data(targs)
        if self.skip_empty_patches and split == 'train':
            non_empty = self.get_empty_patch_mask(targs)
            x, targs = x[non_empty], targs[non_empty]

        return x, targs, (vol_ids, patch_ids)

    def apply_nonlinearity(self, preds):
        if isinstance(preds, torch.Tensor):
            preds = torch.sigmoid(preds)
        else:
            preds = torch.sigmoid(preds[-1])
        preds = preds.round()
        return preds

    def training_step(self, batch, batch_idx):
        self.train_iter += 1
        self.log(f'train/iter', self.train_iter)
        x, targs, (vol_ids, patch_ids) = self.prepare_batch(batch, 'train')
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
                loss = losses.mean()

        with torch.no_grad():
            losses_per_patch = [ self.crit(preds[i].unsqueeze(0), targs[i].unsqueeze(0)).item() for i in range(len(preds)) ]

        return {
            'vol_ids': vol_ids,
            'patch_ids': patch_ids,
            'losses_per_patch': losses_per_patch,
            'loss': loss,
            'preds': preds,
            'targs': targs
        }

    def training_step_end(self, outs):
        if outs is None: return
        preds, targs = outs['preds'], outs['targs']
        preds = self.apply_nonlinearity(preds)

        self.log(f'train/loss', outs['loss'].item())
        self.log(f'train/dice', dice_score(preds, targs).item(), prog_bar=True)
        return {
            'vol_ids': outs['vol_ids'].detach().cpu().numpy(),
            'patch_ids': outs['patch_ids'].detach().cpu().numpy(),
            'losses_per_patch': outs['losses_per_patch'],
            'loss': outs['loss'],
        }

    def training_epoch_end(self, outs):
        if outs is None: return
        for out in outs: del out['loss']
        if dist.is_initialized():
            outs = self.gather_outs(outs)

        losses = defaultdict(dict)
        if not dist.is_initialized() or dist.get_rank() == 0:
            for batch in outs:
                for i, patch_id in enumerate(batch['patch_ids']):
                    vol_id = batch['vol_ids'][i]
                    loss = batch['losses_per_patch'][i]
                    losses[vol_id][patch_id] = loss

        if dist.is_initialized():
            package = [losses]
            dist.broadcast_object_list(package, 0)
            losses = package[0]

        train_sampler = self.get_sampler('train')
        train_sampler.update_patch_weights(losses)

        if isinstance(self.trainer.logger, pl.loggers.WandbLogger) and (not dist.is_initialized() or dist.get_rank() == 0):
            self.trainer.logger.experiment.log({
                'weights': wandb.Histogram([w for vol in train_sampler.vol_meta.values() for w in vol['weights']]),
                'epoch': self.trainer.current_epoch,
            })

    def validation_step(self, batch, batch_idx):
        self.valid_iter += 1
        self.log(f'valid/iter', self.valid_iter)
        x, targs, (vol_ids, patch_ids) = self.prepare_batch(batch, 'valid')
        preds = self(x)
        if isinstance(preds, torch.Tensor):
            preds = self.crop_preds(preds, targs)
        else:
            preds = self.crop_preds(preds[-1], targs)
        loss = self.crit(preds, targs)

        if self.fast_val:
            sum_dims = list(range(2, len(preds.shape))) if len(preds.shape) == 5 else []
            inter = torch.sum(preds * targs, dim=sum_dims)
            denom = preds.sum(dim=sum_dims) + targs.sum(dim=sum_dims)
            return { 'loss': loss.item(), 'inter': inter.item(), 'denom': denom.item() }
        else:
            return {
                'vol_ids': vol_ids,
                'patch_ids': patch_ids,
                'loss': loss.item(),
                'preds': preds,
                'targs': targs
            }

    def validation_step_end(self, outs):
        if self.fast_val: return outs

        outs['vol_ids'] = outs['vol_ids'].cpu().numpy().astype(np.uint8)
        outs['patch_ids'] = outs['patch_ids'].cpu().numpy().astype(np.int64)
        outs['preds'] = self.apply_nonlinearity(outs['preds'])
        outs['preds'] = outs['preds'].cpu().numpy().astype(np.uint8)
        outs['targs'] = outs['targs'].cpu().numpy().astype(np.uint8)
        return outs

    def validation_epoch_end(self, outs):
        if dist.is_initialized():
            outs = self.gather_outs(outs)

        if not dist.is_initialized() or dist.get_rank() == 0:
            if self.fast_val:
                metrics = self.validate_fast(outs)
            else:
                metrics = self.validate_full(outs)
        else:
            metrics = {}

        if dist.is_initialized():
            package = [metrics]
            dist.broadcast_object_list(package, 0)
            metrics = package[0]

        self.log(f'valid/loss', metrics['valid/loss'])
        self.log(f'valid/dice', metrics['valid/dice'])
        self.log(f'valid/hd95', metrics['valid/hd95'])
        self.log('lr', self.get_lr())

    def gather_outs(self, outs):
        # gather predictions from all gpus/nodes in distributed mode
        output = [ None for _ in range(dist.get_world_size())  ]
        dist.barrier()
        dist.all_gather_object(output, outs)

        outs = []
        for out in output:
            outs.extend(out)
        return outs

    def validate_fast(self, outs):
        inter = np.sum([ out['inter'] for out in outs ]) + 1e-10
        denom = np.sum([ out['denom'] for out in outs]) + 1e-10
        dice = 2 * inter / denom
        loss = np.mean([ out['loss'] for out in outs ])
        return {
            'valid/loss': loss,
            'valid/dice': dice,
            'valid/hd95': np.inf,
        }

    def validate_full(self, outs):
        assert hasattr(self, 'ds_meta'), 'Provide volume shapes to use full validation'
        # skip if validation sanity check
        n = 2 if not dist.is_initialized() else 2 * dist.get_world_size()
        if len(outs) <= n: return

        # gather all patches per volume
        preds = defaultdict(dict)
        targs = defaultdict(dict)
        for batch in outs:
            for i, patch_id in enumerate(batch['patch_ids']):
                vol_id = batch['vol_ids'][i]
                preds[vol_id][patch_id] = batch['preds'][i]
                targs[vol_id][patch_id] = batch['targs'][i]
        
        preds = { vol_id: [ patch for _, patch in sorted(patches.items()) ] for vol_id, patches in preds.items() }
        targs = { vol_id: [ patch for _, patch in sorted(patches.items()) ] for vol_id, patches in targs.items() }

        preds = { vol_id: np.concatenate(batches) for vol_id, batches in preds.items() }
        targs = { vol_id: np.concatenate(targs[vol_id]) for vol_id in preds }
        vol_metas = { vol_id: self.ds_meta['vol_meta'][str(vol_id)] for vol_id in preds }

        with ProcessPoolExecutor(max_workers=4, mp_context=get_context('spawn')) as exec:
            # build prediction volumes from patches
            pred_vols = dict(tqdm(
                exec.map(get_volume, preds.keys(), preds.values(), vol_metas.values(), repeat(self.ds_meta['stride']), repeat(self.ds_meta['stride'])),
                total=len(preds), position=2, leave=False, desc='Building preds')
            )
            # build target volumes from patches
            targ_vols = dict(tqdm(
                exec.map(get_volume, targs.keys(), targs.values(), vol_metas.values(), repeat(self.ds_meta['stride']), repeat(self.ds_meta['stride'])),
                total=len(targs), position=2, leave=False, desc='Building targs')
            )

            # gather all the data needed to compute the metrics
            # ensure that the order of volumes is correct for preds, targs and spacings
            data = [ (vol_id, pred_vols[vol_id], targ_vols[vol_id], vol_metas[vol_id]['orig_spacing'] )
                        for vol_id in pred_vols ]

            # compute metrics for each volume
            metrics = list(tqdm(
                exec.map(compute_vol_metrics, data),
                total=len(data), position=2, leave=False, desc='Computing metrics')
            )

        # finally aggregate accross volumes
        return {
            'valid/loss': np.mean([batch['loss'] for batch in outs]),
            'valid/dice': np.mean([v['dice'] for v in metrics]),
            'valid/hd95': np.mean([v['hd95'] for v in metrics]),
        }

    def configure_optimizers(self):
        if self.optim_type == 'adam':
            optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        elif self.optim_type == 'sgd':
            optimizer = SGD(self.parameters(), lr=self.lr, momentum=0.99, nesterov=True, weight_decay=1e-5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': LambdaLR(optimizer, lambda epoch: (1 - epoch/self.trainer.max_epochs)**0.9),
            }
        }

