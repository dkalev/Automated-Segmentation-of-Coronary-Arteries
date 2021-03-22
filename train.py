import pytorch_lightning as pl
from models.base import Baseline3DCNN
from models.unet import UNet

if __name__ == '__main__':
    from dataset import AsocaDataModule
    asoca_dm = AsocaDataModule(batch_size=8, patch_size=64)

    model = Baseline3DCNN()
    # model = UNet()
    trainer = pl.Trainer(gpus=4, max_epochs=100, accelerator='ddp')
    # trainer = pl.Trainer(gpus=1, max_epochs=100)
    trainer.fit(model, asoca_dm)