import torch
import torch.nn as nn
from .base import Base
from collections import OrderedDict

class UNet(Base):
    def __init__(self, *args, n_features=4, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder1 = self.get_encoder(1, n_features, 'encoder1')
        self.encoder2 = self.get_encoder(n_features, n_features*2, 'encoder2')
        self.encoder3 = self.get_encoder(n_features*2, n_features*4, 'encoder3')

        self.bottleneck = self.get_bottleneck(n_features*4)

        self.decoder1 = self.get_decoder(n_features*4, n_features*2, 'decoder1')
        self.decoder2 = self.get_decoder(n_features*2, n_features, 'decoder2')
        self.decoder3 = self.get_decoder(n_features, 1, 'decoder3')

    @staticmethod
    def get_encoder(in_channels, out_channels, name):
        return nn.Sequential(OrderedDict({
            f'{name}-conv1': nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            f'{name}-bn1': nn.BatchNorm3d(in_channels),
            f'{name}-relu1': nn.ReLU(inplace=True),
            f'{name}-conv2': nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            f'{name}-bn2': nn.BatchNorm3d(out_channels),
            f'{name}-relu2': nn.ReLU(inplace=True),
        }))

    @staticmethod
    def get_decoder(in_channels, out_channels, name):
        return nn.Sequential(OrderedDict({
            f'{name}-deconv1': nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=2, output_padding=1, bias=False),
            f'{name}-bn1': nn.BatchNorm3d(out_channels),
            f'{name}-relu1': nn.ReLU(inplace=True),
            f'{name}-deconv2': nn.Conv3d(out_channels, out_channels, kernel_size=3, bias=False),
            f'{name}-bn2': nn.BatchNorm3d(out_channels),
            f'{name}-relu2': nn.ReLU(inplace=True),
        }))

    @staticmethod
    def get_bottleneck(channels, name='bottleneck'):
        return nn.Sequential(OrderedDict({
            f'{name}-conv1': nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            f'{name}-bn1': nn.BatchNorm3d(channels),
            f'{name}-relu1': nn.ReLU(inplace=True),
            f'{name}-conv2': nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            f'{name}-bn2': nn.BatchNorm3d(channels),
            f'{name}-relu2': nn.ReLU(inplace=True),
        }))

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)

        x = self.bottleneck(enc3)

        x = self.decoder1(x + enc3)
        x = self.decoder2(x + enc2)
        x = self.decoder3(x + enc1)

        return x

