import yaml
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
from models import Baseline3DCNN, e3nnCNN, UNet
import time
import pickle as pkl

import warnings
warnings.filterwarnings('ignore', 'Setting attributes on ParameterDict is not supported')
warnings.filterwarnings('ignore', 'training_step returned None')

def train_tune(hparams, checkpoint_dir=None):

    kwargs = { param: hparams[param] for param in ['loss_type', 'lr', 'kernel_size', 'skip_empty_patches'] }
    if hparams['model'] == 'cnn':
        model = Baseline3DCNN(**kwargs)
    elif hparams['model'] == 'unet':
        model = UNet(**kwargs, deep_supervision= hparams['unet']['deep_supervision'])
    elif hparams['model'] == 'e3nn_cnn':
        model = e3nnCNN(**kwargs)

    rdm = AsocaDataModule(
                batch_size=hparams['batch_size'],
                patch_size=hparams['patch_size'],
                patch_stride=hparams['patch_stride'],
                normalize=hparams['normalize'],
                data_clip_range=hparams['data_clip_range'],
                resample_vols=hparams['resample_vols'],
                crop_empty=hparams['crop_empty'],
                oversample=hparams['oversample'],
                data_dir='/home/dkalev/ASOCA/dataset/processed')

    logger = TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version=".", default_hp_metric=False)
    logger.log_hyperparams(hparams, {
        'train_loss': 0,
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
                ['valid_dice', 'valid_loss'],
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
        metric_columns=['valid_dice', 'valid_loss'])

    ray.init()
    output_dir = Path(hparams['output_dir'], 'ray_tune')
    run_name = f"tune_{hparams['model']}_ASOCA-{int(time.time())}"
    analysis = tune.run(
        train_tune,
        resources_per_trial={ "cpu": 3, "gpu":hparams['gpus'] },
        metric="valid_dice",
        mode="max",
        config=hparams,
        local_dir=output_dir,
        num_samples=50,
        scheduler=scheduler,
        progress_reporter=reporter,
        name=run_name)

    print("Best hyperparameters found were: ", analysis.best_config)

    with open(Path(output_dir, run_name, 'analysis.pkl'), 'wb') as f:
        pkl.dump(analysis, f)

def flatten_hparams(hparams):
    res = []
    for k, v in hparams.items():
        if isinstance(v, dict):
            res.extend(v.items())
        else:
            res.append((k,v))
    return dict(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Grid search")
    parser.add_argument('--config_path', type=str, default='config.yml')
    parser.add_argument('--output_dir', type=str, default='.')
    hparams = vars(parser.parse_args())

    with open(hparams['config_path'], 'r') as f:
        hparams = { **hparams, **yaml.safe_load(f) }
        hparams = flatten_hparams(hparams)

    hparams['loss_type']  = tune.choice(['dice', 'dicebce', 'dicebceohnm'])
    hparams['kernel_size'] = tune.choice([3,5])
    hparams['skip_empty_patches'] = tune.choice([True, False])
    hparams['lr'] = tune.loguniform(1e-5, 1e-1)

    param_cols = ['loss_type', 'kernel_size', 'skip_empty_batches', 'lr']
    hparams['param_cols'] = param_cols

    grid_search(hparams)
