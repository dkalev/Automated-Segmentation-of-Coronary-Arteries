import yaml
import json
import argparse
from data_utils import AsocaDataModule

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Build and preprocess ASOCA dataset')
    parser.add_argument('--rebuild', type=bool, default=False, choices=[True, False])
    parser.add_argument('--config_path', type=str, default='config/config.yml')
    hparams = vars(parser.parse_args())

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