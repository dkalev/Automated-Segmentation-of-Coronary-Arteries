import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from pathlib import Path
from data_utils import AsocaDataModule
from models.base import Baseline3DCNN
import time
import pickle as pkl

import warnings
warnings.filterwarnings('ignore', 'Setting attributes on ParameterDict is not supported')
warnings.filterwarnings('ignore', 'training_step returned None')

def train_tune(hparams, checkpoint_dir=None):

    hparams['patch_stride'] = { 32: 28, 64: 60, 128: 120 }[hparams['patch_size']]

    rdm = AsocaDataModule(
        output_dir='/home/dkalev/Automated-Segmentation-of-Coronary-Arteries/dataset',
        patch_size=hparams['patch_size'],
        stride=hparams['patch_stride'])

    model = Baseline3DCNN(kernel_size=hparams['kernel_size'], lr=hparams['lr'])

    logger = TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version=".", default_hp_metric=False)
    logger.log_hyperparams(hparams, {
        'train_loss': 0,
        'valid_bce': 0,
        'valid_f1': 0,
        'valid_loss': 0,
    })

    accelerator = 'ddp' if hparams['gpus'] > 1 else None

    trainer = pl.Trainer(
        max_epochs=hparams['n_epochs'],
        gpus=hparams['gpus'],
        accelerator=accelerator,
        logger=logger,
        checkpoint_callback=False,
        progress_bar_refresh_rate=0,
        callbacks=[
            TuneReportCallback(
                ['valid_dice', 'valid_f1', 'valid_loss'],
                on="validation_end")
        ])
    trainer.fit(model, rdm)

def grid_search(hparams):
    scheduler = ASHAScheduler(
        max_t=hparams['n_epochs'],
        grace_period=1,
        reduction_factor=2)
    
    reporter = CLIReporter(
        parameter_columns=hparams['param_cols'],
        metric_columns=['valid_dice', 'valid_f1', 'valid_loss'])

    ray.init()
    output_dir = Path(hparams['output_dir'], 'ray_tune')
    run_name = f"tune_{hparams['model']}_ASOCA-{int(time.time())}"
    analysis = tune.run(
        train_tune,
        # tune.with_parameters(train_tune, rdm=rdm),
        resources_per_trial={ "cpu": 3, "gpu":hparams['gpus'] },
        metric="valid_dice",
        mode="max",
        config=hparams,
        local_dir=output_dir,
        num_samples=2,
        scheduler=scheduler,
        progress_reporter=reporter,
        name=run_name)

    print("Best hyperparameters found were: ", analysis.best_config)

    with open(Path(output_dir, run_name, 'analysis.pkl'), 'wb') as f:
        pkl.dump(analysis, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Grid search")
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'unet', 'e3nn_cnn'])
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--n_epochs', type=int, default=10)
    hparams = vars(parser.parse_args())

    hparams['patch_size'] = tune.grid_search([32, 64, 128])
    hparams['batch_size'] = tune.grid_search([4, 8, 16, 32])
    hparams['loss_type']  = tune.grid_search(['dice', 'bce', 'dicebce'])
    hparams['kernel_size'] = tune.grid_search([3,5])
    # hparams['lr'] = tune.loguniform(1e-5, 1e-1)

    # hparams['batch_size'] = 4
    # hparams['loss_type']  = 'dice'
    # hparams['kernel_size'] = 3
    hparams['lr'] = 1e-3


    param_cols = ['batch_size', 'patch_size', 'loss_type']
    hparams['param_cols'] = param_cols

    grid_search(hparams)