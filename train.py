import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from data_utils import AsocaDataModule
from models import Baseline3DCNN, UNet, MobileNetV2, SteerableCNN, CubeRegCNN, IcoRegCNN, EquivUNet
from collections import defaultdict
from pathlib import Path
import numpy as np
import argparse
import json
import yaml
import time
import wandb

import warnings
warnings.filterwarnings('ignore', 'Setting attributes on ParameterDict is not supported')
warnings.filterwarnings('ignore', 'training_step returned None')

def get_logger(hparams, run_name):
    # disable logging in debug mode
    if hparams['debug']: return False

    logger = TensorBoardLogger('logs', name=run_name, default_hp_metric=False)
    # log hparams to tensorboard
    logger.log_hyperparams(hparams, {
        'train_f1': 0,
        'train_loss': float('inf'),
        'valid_dice': 0,
        'valid_f1': 0,
        'valid_iou': 0,
        'valid_loss': float('inf'),
    })

    return logger

def combine_config(wandb_config, hparams):
    if len(wandb_config.keys()) == 0: return hparams

    res = defaultdict(dict)

    for key, val in wandb_config.items():
        group, param = key.split('.')
        res[group][param] = val

    for group in hparams:
        if not isinstance(hparams[group], dict):
            res[group] = hparams[group]
            continue
        for key, val in hparams[group].items():
            if key not in res[group]:
                res[group][key] = val

    if not res['dataset']['normalize']:
        res['dataset']['data_clip_range'] = 'None'
        wandb_config.update({'dataset.data_clip_range': 'None'})

    res['dataset']['patch_stride'] = (np.array(res['dataset']['patch_size']) - 20).tolist()

    return dict(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training on ASOCA dataset')
    parser.add_argument('--debug', type=bool, default=False, choices=[True, False])
    parser.add_argument('--config_path', type=str, default='config/config.yml')
    hparams = vars(parser.parse_args())

    with open(hparams['config_path'], 'r') as f:
        hparams = { **hparams, **yaml.safe_load(f) }

    if not hparams['debug']:
        wandb.init(allow_val_change=True)
        hparams = combine_config(wandb.config, hparams)
        wandb.config.update(hparams)
        run_name = wandb.run.name
    else:
        run_name = None

    print(json.dumps(hparams, indent=2))

    tparams = { 'debug': hparams['debug'], **hparams['train']}

    asoca_dm = AsocaDataModule(
        batch_size=hparams['train']['batch_size'],
        distributed=tparams['gpus'] > 1,
        perc_per_epoch_train=0.25,
        perc_per_epoch_val=1,
        **hparams['dataset'])

    asoca_dm.prepare_data()

    kwargs = { param: tparams[param] for param in ['loss_type', 'lr', 'kernel_size', 'skip_empty_patches', 'fast_val'] }
    with open(Path(asoca_dm.data_dir, 'dataset.json'), 'r') as f:
        ds_meta = json.load(f)
    kwargs['ds_meta'] = ds_meta
    kwargs['debug'] = hparams['debug']
    if tparams['model'] in ['mobilenet', 'cubereg', 'icoreg', 'scnn', 'eunet']:
        kwargs.update({'initialize': not tparams['debug']})

    if tparams['model'] == 'cnn':
        model = Baseline3DCNN(**kwargs)
    elif tparams['model'] == 'unet':
        model = UNet(**{**kwargs, **tparams['unet']})
    elif tparams['model'] == 'mobilenet':
        model = MobileNetV2(**kwargs)
    elif tparams['model'] == 'cubereg':
        model = CubeRegCNN(**kwargs)
    elif tparams['model'] == 'icoreg':
        model = IcoRegCNN(**kwargs)
    elif tparams['model'] == 'scnn':
        model = SteerableCNN(**kwargs)
    elif tparams['model'] == 'eunet':
        model = EquivUNet(**kwargs)

    if tparams['model'] in ['mobilenet', 'cubereg', 'icoreg', 'scnn', 'eunet'] and not kwargs['initialize'] :
        model.init()

    trainer_kwargs = {
        'gpus': tparams['gpus'],
        'accelerator': 'ddp' if tparams['gpus'] > 1 else None,
        'max_epochs': tparams['n_epochs'],
        # disable logging in debug mode
        'checkpoint_callback': not tparams['debug'],
        'logger': get_logger(tparams, run_name),
        'auto_lr_find': tparams['auto_lr_find'],
        'gradient_clip_val': 12,
        'callbacks': [ ModelCheckpoint(monitor='valid/loss', mode='min') ],
        'plugins': DDPPlugin(find_unused_parameters=False) if tparams['gpus'] > 1 else None,
    }
    if tparams['debug']:
        del trainer_kwargs['callbacks']

    trainer = pl.Trainer(**trainer_kwargs)

    if tparams['auto_lr_find']: trainer.tune(model, asoca_dm)

    trainer.fit(model, asoca_dm.train_dataloader() if hparams['debug'] else asoca_dm)

