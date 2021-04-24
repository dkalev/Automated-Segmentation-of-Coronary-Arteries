import yaml
from data_utils import AsocaDataModule

if __name__ == '__main__':
    with open('config.yml', 'r') as f:
        hparams = yaml.safe_load(f)['dataset']

    asoca_dm = AsocaDataModule(**hparams)
    asoca_dm.prepare_data()
    asoca_dm.setup()
    train_dl = asoca_dm.train_dataloader()
    valid_dl = asoca_dm.val_dataloader()
    print(next(iter(train_dl))[0].shape)
    print(next(iter(valid_dl))[0].shape)