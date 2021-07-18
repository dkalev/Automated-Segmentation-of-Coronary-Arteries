import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from data_utils import AsocaDataModule
from models import Baseline3DCNN, UNet, MobileNetV2, SteerableCNN, SteerableFTCNN, CubeUNet, IcoUNet, GatedUNet, FTUNet
from pathlib import Path
import numpy as np
from copy import deepcopy
import argparse
import json
import yaml
import re

import warnings
warnings.filterwarnings('ignore', 'Setting attributes on ParameterDict is not supported')
warnings.filterwarnings('ignore', 'training_step returned None')


def update_dict(source, target):
    res = deepcopy(target)
    for key, val in source.items():
        if isinstance(val, dict) and key in res:
            res[key] = update_dict(val, res[key])
        elif val is not None:
            res[key] = val
    return res

def parse_dict(flat_dict, delim='.'):
    res = {}
    for key, val in flat_dict.items():
        if delim not in key:
            res[key] = val
        else:
            path = key
            cur_d = res
            while delim in path:
                k, path = path.split(delim)[0], delim.join(path.split(delim)[1:])
                if k not in cur_d: cur_d[k] = {}
                cur_d = cur_d[k]
            cur_d[path] = val
    return res

def check_patch_config(hparams, model):
    psize = hparams['dataset']['patch_size']
    pstride  = hparams['dataset']['patch_stride']

    has_correct_stride = np.allclose(
        hparams['dataset']['patch_size'],
        np.array(hparams['dataset']['patch_stride']) + 2*model.crop
    )

    if not has_correct_stride:
        if hparams['debug']:
            raise ValueError(f"patch_size: {psize}, stride: {pstride}, crop: {model.crop}, expected stride: {psize[0] - 2*model.crop}")
        else:
            hparams['dataset']['patch_stride'] = psize - 2 * model.crop

    return hparams['dataset']['patch_stride']

def get_model(tparams):
    model_params = ['loss_type', 'lr', 'kernel_size', 'skip_empty_patches', 'fast_val', 'debug']
    equivariant_models = ['mobilenet', 'cubereg', 'icoreg', 'scnn', 'sftcnn', 'eunet']

    kwargs = { param: tparams[param] for param in model_params }

    if tparams['model'] in equivariant_models:
        kwargs.update({'initialize': not tparams['debug']})

    if tparams['model'] == 'cnn':
        model = Baseline3DCNN(**{**kwargs, **tparams['cnn']})
    elif tparams['model'] == 'unet':
        model = UNet(**{**kwargs, **tparams['unet']})
    elif tparams['model'] == 'mobilenet':
        model = MobileNetV2(**kwargs)
    elif tparams['model'] == 'cubereg':
        model = CubeUNet(**{**kwargs, **tparams['cubereg']})
    elif tparams['model'] == 'icoreg':
        model = IcoUNet(**{**kwargs, **tparams['icoreg']})
    elif tparams['model'] == 'ftunet':
        model = FTUNet(**{**kwargs, **tparams['steerable'], **tparams['ft']})
    elif tparams['model'] == 'scnn':
        model = SteerableCNN(**{**kwargs, **tparams['steerable']})
    elif tparams['model'] == 'sftcnn':
        model = SteerableFTCNN(**{**kwargs, **tparams['steerable']})
    elif tparams['model'] == 'eunet':
        model = GatedUNet(**kwargs)

    if tparams['model'] in equivariant_models and not kwargs['initialize']:
        model.init()

    return model

if __name__ == '__main__':
    bool_type = lambda x: x.lower() == 'true'
    list_type = lambda x: [ int(d) for d in re.findall('\d+', x) ]
    int_or_list_type = lambda x: int(x) if re.match('\d+$', x) else list_type(x)
    parser = argparse.ArgumentParser('Training on ASOCA dataset')
    parser.add_argument('--debug', type=bool_type, default=False, choices=[True, False])
    parser.add_argument('--config_path', type=str, default='config/config.yml')

    parser.add_argument('--dataset.patch_size', type=list_type)
    parser.add_argument('--dataset.patch_stride', type=list_type)
    parser.add_argument('--dataset.normalize', type=bool_type)
    parser.add_argument('--dataset.data_clip_range')
    parser.add_argument('--dataset.num_workers', type=int)
    parser.add_argument('--dataset.resample_vols', type=bool_type)
    parser.add_argument('--dataset.oversample', type=bool_type)
    parser.add_argument('--dataset.crop_empty', type=bool_type)
    parser.add_argument('--dataset.perc_per_epoch_train', type=float)
    parser.add_argument('--dataset.perc_per_epoch_val', type=float)
    parser.add_argument('--dataset.weight_update_step', type=float)
    parser.add_argument('--dataset.sample_every_epoch', type=bool_type)
    parser.add_argument('--dataset.data_dir')
    parser.add_argument('--dataset.sourcepath')

    parser.add_argument('--train.model')
    parser.add_argument('--train.gpus', type=int_or_list_type)
    parser.add_argument('--train.n_epochs', type=int)
    parser.add_argument('--train.batch_size', type=int)
    parser.add_argument('--train.lr', type=float)
    parser.add_argument('--train.loss_type')
    parser.add_argument('--train.fully_conv', type=bool_type)
    parser.add_argument('--train.fast_val', type=bool_type)
    parser.add_argument('--train.skip_empty_patches', type=bool_type)
    parser.add_argument('--train.mask_heart', type=bool_type)
    parser.add_argument('--train.optim_type')
    parser.add_argument('--train.kernel_size', type=int)

    # # model specific params
    parser.add_argument('--train.unet.deep_supervision')
    parser.add_argument('--train.cnn.arch')
    parser.add_argument('--train.cubereg.arch')
    parser.add_argument('--train.icoreg.arch')
    parser.add_argument('--train.steerable.repr_type')
    parser.add_argument('--train.ft.grid_kwargs.type')

    hparams = vars(parser.parse_args())
    hparams = parse_dict(hparams)

    with open(hparams['config_path'], 'r') as f:
        hparams = update_dict(hparams, yaml.safe_load(f))
        print(json.dumps(hparams, indent=2))

    multigpu = (isinstance(hparams['train']['gpus'], int) and hparams['train']['gpus'] > 1) \
	or (isinstance(hparams['train']['gpus'], list) and len(hparams['train']['gpus']) > 1)

    tparams = { 'debug': hparams['debug'], **hparams['train']}

    model = get_model(tparams)

    hparams['dataset']['patch_stride'] = check_patch_config(hparams, model)

    asoca_dm = AsocaDataModule( batch_size=hparams['train']['batch_size'], **hparams['dataset'])

    asoca_dm.prepare_data()

    with open(Path(hparams['dataset']['data_dir'], 'dataset.json'), 'r') as f:
        model.ds_meta = json.load(f)

    logger = WandbLogger() if not tparams['debug'] else None
    trainer_kwargs = {
        'gpus': tparams['gpus'],
        'accelerator': 'ddp' if multigpu else None,
        'max_epochs': tparams['n_epochs'],
        # disable logging in debug mode
        'checkpoint_callback': True if not tparams['debug'] else False,
        'logger': logger,
        'auto_lr_find': tparams['auto_lr_find'],
        'gradient_clip_val': 12,
        'callbacks': None if tparams['debug'] else [ ModelCheckpoint(monitor='valid/loss', mode='min') ],
        'plugins': DDPPlugin(find_unused_parameters=False) if multigpu else None,
        'replace_sampler_ddp': False,
        'num_sanity_val_steps': 0,
        'reload_dataloaders_every_epoch': True if multigpu else False,
    }
    if tparams['debug']: del trainer_kwargs['callbacks']

    trainer = pl.Trainer(**trainer_kwargs)

    if logger:
        logger.log_hyperparams(hparams)
        logger.watch(model)

    trainer.fit(model, asoca_dm)
