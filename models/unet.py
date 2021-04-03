import torch
import torch.nn as nn
from .base import Base
from collections import OrderedDict

class UNet(Base):
    def __init__(self, *args, n_features=32, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder1 = self.get_encoder(1, n_features, 'encoder1', mid_channels=32)
        self.encoder2 = self.get_encoder(n_features, n_features*2, 'encoder2')
        self.encoder3 = self.get_encoder(n_features*2, n_features*4, 'encoder3')

        self.bottleneck = self.get_bottleneck(n_features*4, n_features*8)

        self.decoder1 = self.get_decoder(n_features*4+n_features*8, n_features*4, 'decoder1')
        self.decoder2 = self.get_decoder(n_features*2+n_features*4, n_features*2, 'decoder2')
        self.decoder3 = self.get_decoder(n_features+n_features*2, n_features, 'decoder3')
        self.final = nn.Conv3d(n_features, 1, kernel_size=3)


    @staticmethod
    def get_encoder(in_channels, out_channels, name, mid_channels=None):
        if mid_channels is None: mid_channels = in_channels
        return nn.Sequential(OrderedDict({
            f'{name}-conv1': nn.Conv3d(in_channels, mid_channels, kernel_size=3, bias=False),
            f'{name}-bn1': nn.BatchNorm3d(mid_channels),
            f'{name}-relu1': nn.ReLU(inplace=True),
            f'{name}-conv2': nn.Conv3d(mid_channels, out_channels, kernel_size=3, bias=False),
            f'{name}-bn2': nn.BatchNorm3d(out_channels),
            f'{name}-relu2': nn.ReLU(inplace=True),
            f'{name}-maxpool': nn.MaxPool3d(2, stride=2)
        }))

    @staticmethod
    def get_decoder(in_channels, out_channels, name):
        return nn.Sequential(OrderedDict({
            f'{name}-conv1': nn.Conv3d(in_channels, out_channels, kernel_size=3, bias=False),
            f'{name}-bn1': nn.BatchNorm3d(out_channels),
            f'{name}-relu1': nn.ReLU(inplace=True),
            f'{name}-deconv1': nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2, bias=False),
            f'{name}-bn2': nn.BatchNorm3d(out_channels),
            f'{name}-relu2': nn.ReLU(inplace=True),
        }))

    @staticmethod
    def get_bottleneck(in_channels, out_channels, name='bottleneck'):
        return nn.Sequential(OrderedDict({
            f'{name}-conv1': nn.Conv3d(in_channels, out_channels, kernel_size=3, bias=False),
            f'{name}-bn1': nn.BatchNorm3d(out_channels),
            f'{name}-relu1': nn.ReLU(inplace=True),
            f'{name}-deconv1': nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2, bias=False),
            f'{name}-bn2': nn.BatchNorm3d(out_channels),
            f'{name}-relu2': nn.ReLU(inplace=True),
        }))

    def mask_targs(self, targs):
        targs = targs[..., # [batch_size, n_channels]
                     31:-31, # x
                     31:-31, # y
                     31:-31, ] # z

        mask = targs.sum(dim=[1,2,3,4]) > 0
        return targs[mask,...], mask

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)

        x = self.bottleneck(enc3)

        x = self.decoder1(torch.cat([x[...,4:-4,4:-4,4:-4] ,enc3], dim=1))
        x = self.decoder2(torch.cat([x ,enc2[...,4:-5,4:-5,4:-5]], dim=1))
        x = self.decoder3(torch.cat([x ,enc1[...,13:-13,13:-13,13:-13]], dim=1))
        x = self.final(x)

        return x

