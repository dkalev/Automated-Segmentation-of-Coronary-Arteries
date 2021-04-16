import torch
import torch.nn as nn
from .base import Base
from collections import OrderedDict

class UNet(Base):
    def __init__(self, *args, kernel_size=3, n_features=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.crop = self.padding * 2 * 3 # 2 conv per encoder

        self.encoder1 = self.get_encoder(1, n_features, 'encoder1', mid_channels=n_features//2)
        self.encoder2 = self.get_encoder(n_features, n_features*2, 'encoder2')
        self.encoder3 = self.get_encoder(n_features*2, n_features*4, 'encoder3')

        self.bottleneck = self.get_bottleneck(n_features*4, n_features*8)

        self.decoder1 = self.get_decoder(n_features*4+n_features*8, n_features*4, 'decoder1')
        self.decoder2 = self.get_decoder(n_features*2+n_features*4, n_features*2, 'decoder2')
        self.decoder3 = self.get_decoder(n_features+n_features*2, n_features, 'decoder3')
        self.final = nn.Conv3d(n_features, 1, kernel_size=self.kernel_size, padding=self.padding)

    def get_encoder(self, in_channels, out_channels, name, mid_channels=None):
        if mid_channels is None: mid_channels = in_channels
        return nn.Sequential(OrderedDict({
            f'{name}-conv1': nn.Conv3d(in_channels, mid_channels, kernel_size=self.kernel_size, padding=self.padding, bias=False),
            f'{name}-bn1': nn.BatchNorm3d(mid_channels),
            f'{name}-relu1': nn.ReLU(inplace=True),
            f'{name}-conv2': nn.Conv3d(mid_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding, bias=False),
            f'{name}-bn2': nn.BatchNorm3d(out_channels),
            f'{name}-relu2': nn.ReLU(inplace=True),
            f'{name}-maxpool': nn.MaxPool3d(2, stride=2)
        }))

    def get_decoder(self, in_channels, out_channels, name):
        return nn.Sequential(OrderedDict({
            f'{name}-deconv1': nn.ConvTranspose3d(in_channels, out_channels, kernel_size=self.kernel_size, stride=2, output_padding=1, bias=False),
            f'{name}-bn1': nn.BatchNorm3d(out_channels),
            f'{name}-relu1': nn.ReLU(inplace=True),
            f'{name}-conv1': nn.Conv3d(out_channels, out_channels, kernel_size=self.kernel_size, bias=False),
            f'{name}-bn2': nn.BatchNorm3d(out_channels),
            f'{name}-relu2': nn.ReLU(inplace=True),
        }))

    def get_bottleneck(self, in_channels, out_channels, name='bottleneck'):
        return nn.Sequential(OrderedDict({
            f'{name}-conv1': nn.Conv3d(in_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding, bias=False),
            f'{name}-bn1': nn.BatchNorm3d(out_channels),
            f'{name}-relu1': nn.ReLU(inplace=True),
            f'{name}-deconv1': nn.Conv3d(out_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding, bias=False),
            f'{name}-bn2': nn.BatchNorm3d(out_channels),
            f'{name}-relu2': nn.ReLU(inplace=True),
        }))

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)

        x = self.bottleneck(enc3)

        x = self.decoder1(torch.cat([x ,enc3], dim=1))
        x = self.decoder2(torch.cat([x ,enc2], dim=1))
        x = self.decoder3(torch.cat([x ,enc1], dim=1))
        x = self.final(x)

        x = x[...,
            self.crop:-self.crop, # x
            self.crop:-self.crop, # y
            self.crop:-self.crop] # z
        return x

