import torch
import torch.nn as nn
from .base import Base
from collections import OrderedDict

class NNUNet(Base):
    def __init__(self, *args, kernel_size=3, n_features=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.encoders = nn.ModuleDict({
            'encoder1': self.get_encoder(1, 32, stride=1),
            'encoder2': self.get_encoder(32, 64),
            'encoder3': self.get_encoder(64, 128),
            'encoder4': self.get_encoder(128, 256),
            'encoder5': self.get_encoder(256, 320),
        })

        self.bottleneck = self.get_bottleneck(320)

        self.decoders = nn.ModuleDict({
            'decoder1': nn.ModuleDict({
                'upsampler': self.get_upsampler(320, 320, kernel_size=(1,2,2), stride=(1,2,2)),
                'decoder': self.get_decoder(640, 320)
            }),
            'decoder2': nn.ModuleDict({
                'upsampler': self.get_upsampler(320, 256),
                'decoder': self.get_decoder(512, 256)
            }),
            'decoder3': nn.ModuleDict({
                'upsampler': self.get_upsampler(256, 128),
                'decoder': self.get_decoder(256, 128)
            }),
            'decoder4': nn.ModuleDict({
                'upsampler': self.get_upsampler(128, 64),
                'decoder': self.get_decoder(128, 64)
            }),
            'decoder5': nn.ModuleDict({
                'upsampler': self.get_upsampler(64, 32),
                'decoder': self.get_decoder(64, 32)
            }),
        })

        self.final = nn.Conv3d(n_features, 1, kernel_size=1)

        self.crop = len(self.encoders) * 2 * self.padding # 2 conv per encoder
    
    def get_encoder(self, in_channels, out_channels, stride=2):
        return nn.Sequential(OrderedDict({
            'conv1': nn.Conv3d(in_channels, out_channels, kernel_size=self.kernel_size, stride=stride, padding=self.padding, bias=False),
            'instnorm': nn.InstanceNorm3d(out_channels),
            'lrelu': nn.LeakyReLU(inplace=True),
            'conv2': nn.Conv3d(out_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding, bias=False),
            'instnorm': nn.InstanceNorm3d(out_channels),
            'lrelu': nn.LeakyReLU(inplace=True),
        }))

    def get_decoder(self, in_channels, out_channels):
        return nn.Sequential(OrderedDict({
            'conv1': nn.Conv3d(in_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding, bias=False),
            'instnorm': nn.InstanceNorm3d(out_channels),
            'lrelu': nn.LeakyReLU(inplace=True),
            'conv2': nn.Conv3d(out_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding, bias=False),
            'instnorm': nn.InstanceNorm3d(out_channels),
            'lrelu': nn.LeakyReLU(inplace=True),
        }))

    def get_upsampler(self, in_channels, out_channels, kernel_size=2, stride=2):
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False)

    def get_bottleneck(self, channels):
        return nn.Sequential(OrderedDict({
            'conv1': nn.Conv3d(channels, channels, kernel_size=self.kernel_size, stride=(1,2,2), padding=self.padding, bias=False),
            'instnorm': nn.InstanceNorm3d(channels),
            'lrelu': nn.LeakyReLU(inplace=True),
            'conv2': nn.Conv3d(channels, channels, kernel_size=self.kernel_size, padding=self.padding, bias=False),
            'instnorm': nn.InstanceNorm3d(channels),
            'lrelu': nn.LeakyReLU(inplace=True),
        }))

    def forward(self, x):
        skip_cons = []
        for encoder in self.encoders.values():
            x = encoder(x)
            skip_cons.append(x)

        x = self.bottleneck(x)

        for i, block in enumerate(self.decoders.values()):
            x = block['upsampler'](x)
            x = torch.cat([x, skip_cons[-(i+1)]], dim=1)
            x = block['decoder'](x)

        x = self.final(x)

        x = x[...,
            self.crop:-self.crop, # x
            self.crop:-self.crop, # y
            self.crop:-self.crop] # z
        return x

