import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from data_utils import AsocaDataModule
from models.base import Baseline3DCNN
from models.unet import UNet
from models.e3nn_models import e3nnCNN
from models.nn_unet import NNUNet
import yaml
import time

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
        'valid_loss': float('inf'),
    })

    return logger

if __name__ == '__main__':
    with open('config.yml', 'r') as f:
        hparams = yaml.safe_load(f)

    asoca_dm = AsocaDataModule(batch_size=hparams['train']['batch_size'], **hparams['dataset'])

    tparams = hparams['train']

    kwargs = { param: tparams[param] for param in ['loss_type', 'lr', 'kernel_size', 'skip_empty_patches'] }
    if tparams['model'] == 'cnn':
        model = Baseline3DCNN(**kwargs)
    elif tparams['model'] == 'unet':
        model = UNet(**kwargs)
    elif tparams['model'] == 'e3nn_cnn':
        model = e3nnCNN(**kwargs)
    elif tparams['model'] == 'nnunet':
        model = NNUNet(**kwargs)

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