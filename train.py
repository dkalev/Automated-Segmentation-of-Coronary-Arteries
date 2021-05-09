import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from data_utils import AsocaDataModule
from models import Baseline3DCNN, UNet
from collections import defaultdict
import argparse
import json
import yaml
import time
import wandb

import warnings
warnings.filterwarnings('ignore', 'Setting attributes on ParameterDict is not supported')
warnings.filterwarnings('ignore', 'training_step returned None')

def get_logger(hparams):
    # disable logging in debug mode
    if hparams['debug']: return False

    logger = TensorBoardLogger('logs', name=f"{hparams['model']}-{int(time.time())}", default_hp_metric=False)
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

    return dict(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training on ASOCA dataset')
    parser.add_argument('--debug', type=bool, default=False, choices=[True, False])
    parser.add_argument('--config_path', type=str, default='config.yml')
    hparams = vars(parser.parse_args())

    with open(hparams['config_path'], 'r') as f:
        hparams = { **hparams, **yaml.safe_load(f) }
    
    wandb.init()

    hparams = combine_config(wandb.config, hparams)
    print(json.dumps(hparams, indent=2))

    tparams = { 'debug': hparams['debug'], **hparams['train']}

    asoca_dm = AsocaDataModule(
        batch_size=hparams['train']['batch_size'],
        distributed=tparams['gpus'] > 1,
        **hparams['dataset'])

    kwargs = { param: tparams[param] for param in ['loss_type', 'lr', 'kernel_size', 'skip_empty_patches'] }
    if tparams['model'] == 'cnn':
        model = Baseline3DCNN(**kwargs)
    elif tparams['model'] == 'unet':
        model = UNet(**{**kwargs, **tparams['unet']})

    trainer_kwargs = {
        'gpus': tparams['gpus'],
        'accelerator': 'ddp' if tparams['gpus'] > 1 else None,
        'max_epochs': tparams['n_epochs'],
        # disable logging in debug mode
        'checkpoint_callback': not tparams['debug'],
        'logger': get_logger(tparams),
        'auto_lr_find': tparams['auto_lr_find'],
    }

    trainer = pl.Trainer(**trainer_kwargs)

    if tparams['auto_lr_find']: trainer.tune(model, asoca_dm)

    trainer.fit(model, asoca_dm)
