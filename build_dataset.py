import yaml
import json
import argparse
from data_utils.datamodule import AsocaClassificationDataModule, AsocaDataModule

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Build and preprocess ASOCA dataset')
    parser.add_argument('--rebuild', type=bool, default=False, choices=[True, False])
    parser.add_argument('--config_path', type=str, default='config/config.yaml')
    parser.add_argument('--type', type=str, choices=['class', 'seg'],  default='seg')
    hparams = vars(parser.parse_args())

    if hparams['type'] == 'seg':
        with open(hparams['config_path'], 'r') as f:
            hparams = { 'rebuild': hparams['rebuild'], **yaml.safe_load(f)['dataset'] }
            print(json.dumps(hparams, indent=2))

        asoca_dm = AsocaDataModule(**hparams)
        asoca_dm.prepare_data()
        asoca_dm.setup()
        train_dl = asoca_dm.train_dataloader()
        valid_dl = asoca_dm.val_dataloader()
        print(next(iter(train_dl))[0].shape)
        print(next(iter(valid_dl))[0].shape)

    elif hparams['type'] == 'class':
        acdm = AsocaClassificationDataModule(
            patch_size=68,
            n_patches=100000,
            data_dir='dataset/classification',
            sourcepath='dataset/ASOCA2020Data.zip',
        )

        acdm.prepare_data()
        train_dl = acdm.train_dataloader()
        valid_dl = acdm.val_dataloader()

        train_batch = next(iter(acdm.train_dataloader()))
        valid_batch = next(iter(acdm.val_dataloader()))
        print(train_batch[0].shape, train_batch[1:])
        print(valid_batch[0].shape, valid_batch[1:])