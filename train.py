import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from dataset import AsocaDataModule
from models.base import Baseline3DCNN
from models.unet import UNet
from models.e3nn_models import e3nnCNN
import argparse
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
    parser = argparse.ArgumentParser('Train model on ASOCA dataset')
    parser.add_argument('--model', type=str, required=True, choices=['cnn', 'unet', 'e3nn_cnn'])
    parser.add_argument('--debug', type=bool, default=False, choices=[True, False])
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--patch_stride', type=int, default=28)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--auto_lr_find', type=bool, default=False)
    parser.add_argument('--loss_type', type=str, default='dicebce', choices=['dice', 'bce', 'dicebce'])
    hparams = vars(parser.parse_args())

    asoca_dm = AsocaDataModule(batch_size=hparams['batch_size'],
                               patch_size=hparams['patch_size'],
                               stride=hparams['patch_stride'])

    kwargs = { 'loss_type': hparams['loss_type'] }
    if hparams['model'] == 'cnn':
        model = Baseline3DCNN(**kwargs)
    elif hparams['model'] == 'unet':
        model = UNet(**kwargs)
    elif hparams['model'] == 'e3nn_cnn':
        model = e3nnCNN(**kwargs)

    trainer_kwargs = {
        'gpus': hparams['gpus'],
        'accelerator': 'ddp' if hparams['gpus'] > 1 else None,
        'max_epochs': hparams['n_epochs'],
        # disable logging in debug mode
        'checkpoint_callback': not hparams['debug'],
        'logger': get_logger(hparams),
        'auto_lr_find': hparams['auto_lr_find'],
    }

    
    trainer = pl.Trainer(**trainer_kwargs)

    if hparams['auto_lr_find']: trainer.tune(model, asoca_dm)

    trainer.fit(model, asoca_dm)