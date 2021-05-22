import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import Base
from collections import OrderedDict

class UNet(Base):
    def __init__(self, *args, kernel_size=3, n_features=32, deep_supervision=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.deep_supervision = deep_supervision

        self.encoders = nn.ModuleList([
            self.get_encoder(1, 32, stride=1),
            self.get_encoder(32, 64),
            self.get_encoder(64, 128),
            self.get_encoder(128, 256),
            self.get_encoder(256, 320),
        ])

        self.bottleneck = self.get_bottleneck(320)

        self.decoders = nn.ModuleList([
            nn.ModuleDict({
                'upsampler': self.get_upsampler(320, 320, kernel_size=(1,2,2), stride=(1,2,2)),
                'decoder': self.get_decoder(640, 320)
            }),
            nn.ModuleDict({ 'upsampler': self.get_upsampler(320, 256), 'decoder': self.get_decoder(512, 256) }),
            nn.ModuleDict({ 'upsampler': self.get_upsampler(256, 128), 'decoder': self.get_decoder(256, 128) }),
            nn.ModuleDict({ 'upsampler': self.get_upsampler(128, 64), 'decoder': self.get_decoder(128, 64) }),
            nn.ModuleDict({ 'upsampler': self.get_upsampler(64, 32), 'decoder': self.get_decoder(64, 32) }),
        ])


        if self.deep_supervision:
            self.heads = nn.ModuleList([
                nn.Conv3d(320, 1, kernel_size=1, bias=False),
                nn.Conv3d(256, 1, kernel_size=1, bias=False),
                nn.Conv3d(128, 1, kernel_size=1, bias=False),
                nn.Conv3d(64, 1, kernel_size=1, bias=False),
                nn.Conv3d(32, 1, kernel_size=1, bias=False),
            ])
        else:
            self.final = nn.Conv3d(n_features, 1, kernel_size=1, bias=False)

        self.register_buffer('ds_weight',
            torch.FloatTensor([0., 0.06666667, 0.13333333, 0.26666667, 0.53333333])
        )

        self.crop = len(self.encoders) * 2 * self.padding # 2 conv per encoder

    def get_encoder(self, in_channels, out_channels, stride=2):
        return nn.Sequential(OrderedDict({
            'conv1': nn.Conv3d(in_channels, out_channels, kernel_size=self.kernel_size, stride=stride, padding=self.padding),
            'instnorm1': nn.InstanceNorm3d(out_channels, affine=True),
            'lrelu1': nn.LeakyReLU(inplace=True),
            'conv2': nn.Conv3d(out_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding),
            'instnorm2': nn.InstanceNorm3d(out_channels, affine=True),
            'lrelu2': nn.LeakyReLU(inplace=True),
        }))

    def get_decoder(self, in_channels, out_channels):
        return nn.Sequential(OrderedDict({
            'conv1': nn.Conv3d(in_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding),
            'instnorm1': nn.InstanceNorm3d(out_channels, affine=True),
            'lrelu1': nn.LeakyReLU(inplace=True),
            'conv2': nn.Conv3d(out_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding),
            'instnorm2': nn.InstanceNorm3d(out_channels, affine=True),
            'lrelu2': nn.LeakyReLU(inplace=True),
        }))

    def get_upsampler(self, in_channels, out_channels, kernel_size=2, stride=2):
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False)

    def get_bottleneck(self, channels):
        return nn.Sequential(OrderedDict({
            'conv1': nn.Conv3d(channels, channels, kernel_size=self.kernel_size, stride=(1,2,2), padding=self.padding),
            'instnorm1': nn.InstanceNorm3d(channels, affine=True),
            'lrelu1': nn.LeakyReLU(inplace=True),
            'conv2': nn.Conv3d(channels, channels, kernel_size=self.kernel_size, padding=self.padding),
            'instnorm2': nn.InstanceNorm3d(channels, affine=True),
            'lrelu2': nn.LeakyReLU(inplace=True),
        }))

    def forward(self, x):
        targ_size = x.shape[2:]

        skip_cons = []
        for encoder in self.encoders:
            x = encoder(x)
            skip_cons.append(x)

        x = self.bottleneck(x)

        outputs = []
        for i, block in enumerate(self.decoders):
            x = block['upsampler'](x)
            x = torch.cat([x, skip_cons[-(i+1)]], dim=1)
            x = block['decoder'](x)
            if self.deep_supervision:
                outputs.append(x)

        if self.deep_supervision:
            outputs = [ head(out) for head, out in zip(self.heads, outputs) ]
            outputs = [ F.interpolate(out, size=targ_size) for out in outputs ]
            return outputs
        else:
            return self.final(x)
