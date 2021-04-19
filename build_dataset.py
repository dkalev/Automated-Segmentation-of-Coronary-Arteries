import argparse
from data_utils import AsocaDataModule

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Build ASOCA dataset')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--stride', type=int, default=28)
    parser.add_argument('--normalize', type=bool, default=False)
    parser.add_argument('--norm_type', type=str, default='global', choices=['global', 'patch-wise'])
    hparams = vars(parser.parse_args())

    asoca_dm = AsocaDataModule(**hparams)
    asoca_dm.prepare_data()
    asoca_dm.setup()
    train_dl = asoca_dm.train_dataloader()
    valid_dl = asoca_dm.val_dataloader()
    print(next(iter(train_dl))[0].shape)
    print(next(iter(valid_dl))[0].shape)