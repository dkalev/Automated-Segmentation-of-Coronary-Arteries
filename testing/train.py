import sys
sys.path.append('..')
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from models import Baseline3DCNN, UNet
from dataset import MnistDataset
from data_utils import AsocaDataModule
import argparse
import json
import yaml
import wandb

import warnings
warnings.filterwarnings('ignore', 'Setting attributes on ParameterDict is not supported')
warnings.filterwarnings('ignore', 'training_step returned None')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training on ASOCA dataset')
    parser.add_argument('--config_path', type=str, default='config.yml')
    hparams = vars(parser.parse_args())

    with open(hparams['config_path'], 'r') as f:
        hparams = { **hparams, **yaml.safe_load(f) }
    
    wandb.init(config=hparams)

    print(json.dumps(hparams, indent=2))

    tparams = hparams['train']
    
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
        'checkpoint_callback': False,
        'logger': False,
    }

    trainer = pl.Trainer(**trainer_kwargs, gradient_clip_val=12)

    asoca_dm = AsocaDataModule(
        batch_size=hparams['train']['batch_size'],
        distributed=tparams['gpus'] > 1,
        data_dir='../dataset/3d_mnist_128',
        **hparams['dataset'])

    trainer.fit(model, asoca_dm)

    # ds_train = MnistDataset('../dataset/3d_mnist_64.h5', split='train')
    # ds_test = MnistDataset('../dataset/3d_mnist_64.h5', split='test')

    # dl_train = DataLoader(ds_train, shuffle=True, batch_size=hparams['train']['batch_size'], pin_memory=True, num_workers=12)
    # dl_test = DataLoader(ds_test, batch_size=batch_size=hparams['train']['batch_size'], pin_memory=True, num_workers=12)
    # trainer.fit(model, dl_train, dl_test)

