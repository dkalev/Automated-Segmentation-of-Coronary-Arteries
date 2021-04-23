import argparse
from data_utils import AsocaDataModule

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Build ASOCA dataset')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--stride', type=int, default=28)
    parser.add_argument('--normalize', type=str, choices=['global', 'patch-wise'])
    parser.add_argument('--clip_range_low', type=float)
    parser.add_argument('--clip_range_high', type=float)
    hparams = vars(parser.parse_args())

    if hparams['clip_range_low'] is not None and hparams['clip_range_high'] is not None:
        hparams['data_clip_range'] = (hparams['clip_range_low'], hparams['clip_range_high'])
    del hparams['clip_range_low']
    del hparams['clip_range_high']

    asoca_dm = AsocaDataModule(**hparams)
    asoca_dm.prepare_data()
    asoca_dm.setup()
    train_dl = asoca_dm.train_dataloader()
    valid_dl = asoca_dm.val_dataloader()
    print(next(iter(train_dl))[0].shape)
    print(next(iter(valid_dl))[0].shape)