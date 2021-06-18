import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from loss import DiceBCELoss, BCEWrappedLoss, DiceLoss, DiceBCE_OHNMLoss
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from tqdm import tqdm
from functools import partial
from metrics import dice_score, hausdorff_95
from collections import defaultdict
from data_utils.helpers import get_volume_pred
import wandb
from itertools import repeat

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


class Base(BasePL):
    def __init__(self, *args,
                        lr=1e-3,
                        loss_type='dice',
                        skip_empty_patches=False,
                        mask_heart=False,
                        optim_type='adam',
                        ds_meta=None,
                        debug=False,
                        fast_val=False,
                        **kwargs):
        super().__init__(*args, **kwargs)
        self.crit = self.get_loss_func(loss_type)
        self.lr = lr
        self.skip_empty_patches = skip_empty_patches
        self.mask_heart = mask_heart

        assert ds_meta is not None
        self.ds_meta = ds_meta
        self.debug = debug
        self.fast_val = fast_val

        if optim_type in ['adam', 'sgd']:
            self.optim_type = optim_type
        else:
            raise ValueError(f'Unsupported optimizer type: {optim_type}')

    def on_train_epoch_start(self):
        train_sampler = self.trainer.train_dataloader.sampler
        valid_sampler = self.trainer.val_dataloaders[0].sampler

        if dist.is_initialized():
            # ensure that each process validates on the same subset of files in ddp
            # otherwise because a sampler is initialized in each process and ddp distributes
            # the data further we end up with partial predictions for e.g. 4 volumes instead of
            # full predictions (all patches) for the 2 required volumes

            # wrapped in distributed sampler
            train_sampler = train_sampler.sampler
            valid_sampler = valid_sampler.sampler
            if dist.get_rank() == 0:
                file_ids = valid_sampler.sample_ids()
                for dest_rank in range(1, dist.get_world_size()):
                    dist.send(torch.tensor(file_ids, device='cuda', dtype=torch.uint8), dest_rank)
            else:
                file_ids = torch.empty(len(valid_sampler.sample_ids()), device='cuda', dtype=torch.uint8)
                dist.recv(file_ids, src=0)
                file_ids = file_ids.tolist()
        else:
            file_ids = valid_sampler.sample_ids()
        valid_sampler.file_ids = file_ids

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
        self.log(f'train/iter', self.train_iter)
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
        self.log(f'train/dice', dice_score(preds, targs).item(), prog_bar=True)
        return outs['loss']

    def validation_step(self, batch, batch_idx):
        self.valid_iter += 1
        self.log(f'valid/iter', self.valid_iter)
        vol_ids, x, targs, h_masks = self.prepare_batch(batch, 'valid')
        preds = self(x)
        if isinstance(preds, torch.Tensor):
            preds = self.crop_preds(preds, targs)
        else:
            preds = self.crop_preds(preds[-1], targs)
        loss = self.crit(preds, targs)
        if self.mask_heart: preds = preds.masked_fill(h_masks==0, -10)

        if self.fast_val:
            sum_dims = list(range(2, len(preds.shape))) if len(preds.shape) == 5 else []
            inter = torch.sum(preds * targs, dim=sum_dims)
            denom = preds.sum(dim=sum_dims) + targs.sum(dim=sum_dims)
            return { 'loss': loss.item(), 'inter': inter.item(), 'denom': denom.item() }
        else:
            return { 'vol_ids': vol_ids, 'loss': loss.item(), 'preds': preds, 'targs': targs }

    def validation_step_end(self, outs):
        if self.fast_val: return outs

        outs['vol_ids'] = outs['vol_ids'].cpu().numpy().astype(np.uint8)
        outs['preds'] = self.apply_nonlinearity(outs['preds'])
        outs['preds'] = outs['preds'].cpu().numpy().astype(np.uint8)
        outs['targs'] = outs['targs'].cpu().numpy().astype(np.uint8)
        return outs

    def gather_preds(self, outs):
        # gather predictions from all gpus/nodes in distributed mode
        output = [ None for _ in range(dist.get_world_size())  ]
        dist.barrier()
        dist.all_gather_object(output, outs)

        outs = []
        for out in output:
            outs.extend(out)
        return outs

    def validation_epoch_end(self, outs):
        if dist.is_initialized():
            outs = self.gather_preds(outs)

            if dist.get_rank() != 0:
                self.log('valid/loss', np.inf)
                return

        if self.fast_val:
            self.validate_fast(outs)
        else:
            self.validate_full(outs)
        self.log('lr', self.get_lr())

    def validate_fast(self, outs):
        inter = np.sum([ out['inter'] for out in outs ]) + 1e-10
        denom = np.sum([ out['denom'] for out in outs]) + 1e-10
        dice = 2 * inter / denom
        loss = np.mean([ out['loss'] for out in outs ])
        self.log('valid/dice', dice)
        self.log('valid/loss', loss)

    def validate_full(self, outs):
        # skip if validation sanity check
        if len(outs) < 10: return

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

        with ProcessPoolExecutor(max_workers=4, mp_context=get_context('spawn')) as exec:
            # build prediction volumes from patches
            pred_vols = dict(tqdm(
                exec.map(get_volume, preds.keys(), preds.values(), vol_metas.values(), repeat(self.ds_meta['stride']), repeat(self.ds_meta['stride'])),
                total=len(preds), position=2, desc='Building preds')
            )
            # build target volumes from patches
            targ_vols = dict(tqdm(
                exec.map(get_volume, targs.keys(), targs.values(), vol_metas.values(), repeat(self.ds_meta['stride']), repeat(self.ds_meta['stride'])),
                total=len(targs), position=2, desc='Building targs')
            )

            # gather all the data needed to compute the metrics
            # ensure that the order of volumes is correct for preds, targs and spacings
            data = [ (vol_id, pred_vols[vol_id], targ_vols[vol_id], vol_metas[vol_id]['orig_spacing'] )
                        for vol_id in pred_vols ]

            # compute metrics for each volume
            metrics = list(tqdm(
                exec.map(compute_vol_metrics, data),
                total=len(data), position=2, desc='Computing metrics')
            )

        # finally aggregate accross volumes and log
        self.log(f'valid/loss', np.mean([batch['loss'] for batch in outs]))
        self.log(f'valid/dice', np.mean([v['dice'] for v in metrics]))
        self.log(f'valid/hd95', np.mean([v['hd95'] for v in metrics]))

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


class Baseline3DCNN(Base):
    def __init__(self, *args, kernel_size=5, arch='default', **kwargs):
        super().__init__(*args, **kwargs)

        common_params = {
            'kernel_size': kernel_size,
            'bias': False,
        }

        if arch == 'default':
            block_params = [
                {'in_channels': 1, 'out_channels': 60 },
                {'in_channels': 60, 'out_channels': 120 },
                {'in_channels': 120, 'out_channels': 120 },
                {'in_channels': 120, 'out_channels': 60 },
            ]
        elif arch == 'strided':
            block_params = [
                {'in_channels': 1, 'out_channels': 180 , 'stride': 2 },
                {'in_channels': 180, 'out_channels': 360  },
                {'in_channels': 360, 'out_channels': 720 },
                {'in_channels': 720, 'out_channels': 720 },
                {'in_channels': 720, 'out_channels': 720 },
                {'in_channels': 720, 'out_channels': 120 },
            ]
        elif arch == 'patch64':
            block_params = [
                {'in_channels': 1, 'out_channels': 120 },
                {'in_channels': 120, 'out_channels': 120 },
                {'in_channels': 120, 'out_channels': 360 },
                {'in_channels': 360, 'out_channels': 120 },
            ]

        blocks = [
            nn.Sequential(
                nn.Conv3d(**b_params, **common_params),
                nn.InstanceNorm3d(b_params['out_channels'], affine=True),
                nn.ReLU(inplace=True),
            ) for b_params in block_params
        ]

        if arch == 'strided': blocks.append(nn.Upsample(scale_factor=2))
        blocks.append(
            nn.Conv3d(block_params[-1]['out_channels'], 1, kernel_size=1)
        )

        if arch == 'strided':
            self.crop = 11
        else:
            self.crop = (common_params['kernel_size']//2) * len(blocks[:-1]) # last layer doesn't affect crop

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)

