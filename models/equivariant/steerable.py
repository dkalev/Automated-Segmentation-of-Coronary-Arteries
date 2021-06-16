import e3cnn.nn as enn
from e3cnn import gspaces
from e3cnn.group import directsum
import torch
import torch.nn as nn
from typing import Tuple
from .base import BaseEquiv, GatedFieldType, FTNonLinearity


class SteerableCNN(BaseEquiv):
    def __init__(self, in_channels=1, ouqt_channels=1, max_freq=2, kernel_size=3, padding=0, initialize=True, **kwargs):
        gspace = gspaces.rot3dOnR3()
        super().__init__(gspace, in_channels, kernel_size, padding, **kwargs)

        small_type = GatedFieldType.build(gspace, 60)
        mid_type = GatedFieldType.build(gspace, 240)
        final_type = GatedFieldType.build(gspace, 240, max_freq=1)

        common_kwargs = {
                'kernel_size': kernel_size,
                'padding': self.padding,
                'initialize': initialize,
        }

        blocks = [
            self.get_block(self.input_type, small_type, **common_kwargs, stride=2),
            self.get_block(small_type.no_gates(), small_type, **common_kwargs),
            self.get_block(small_type.no_gates(), mid_type, **common_kwargs),
            self.get_block(mid_type.no_gates(), final_type, **common_kwargs),
        ]

        self.model = enn.SequentialModule(*blocks)
        self.pool = enn.NormPool(blocks[-1].out_type)
        pool_out = self.pool.out_type.size
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(pool_out, out_channels, kernel_size=1)
        )
        # input layer + crop of each block
        self.crop = len(blocks) * (kernel_size // 2 - self.padding)
        self.crop = 7

    def get_block(self, in_type, out_type, **kwargs):
        layers = []

        layers.append( enn.R3Conv(in_type, out_type, **kwargs) )
        layers.append( enn.IIDBatchNorm3d(out_type) )
        layers.append(
            enn.MultipleModule(out_type,
                labels=[
                    *( len(out_type.trivials) * ['trivial'] + (len(out_type.gated) + len(out_type.gates)) * ['gate'] )
                ],
                modules=[
                    (enn.ELU(out_type.trivials, inplace=True), 'trivial'),
                    (enn.GatedNonLinearity1(out_type.gated+out_type.gates,
                                            len(out_type.gated)*['gated']+len(out_type.gates)*['gate']), 'gate')
                ]
            )
        )
        return enn.SequentialModule(*layers)

    def forward(self, x):
        x = self.pre_forward(x)
        x = self.model(x)
        x = self.pool(x)
        x = x.tensor
        x = self.final(x)
        return x


class SteerableFTCNN(BaseEquiv):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 max_freq=2,
                 kernel_size=3,
                 padding=0,
                 type='spherical',
                 initialize=True, **kwargs):
        gspace = gspaces.rot3dOnR3()
        super().__init__(gspace, in_channels, kernel_size, padding, **kwargs)

        self.type = type

        common_kwargs = {
                'kernel_size': kernel_size,
                'padding': self.padding,
                'initialize': initialize,
        }

        params = [
            {'max_freq': 2, 'out_channels': 120, **common_kwargs, 'stride': 2},
            {'max_freq': 2, 'out_channels': 240, **common_kwargs},
            {'max_freq': 2, 'out_channels': 480, **common_kwargs},
            {'max_freq': 2, 'out_channels': 960, **common_kwargs},
            {'max_freq': 2, 'out_channels': 240, **common_kwargs},
        ]

        blocks = []
        in_type = self.input_type
        for param in params:
            block, in_type = self.get_block(in_type, **param)
            blocks.append(block)

        self.model = enn.SequentialModule(*blocks)
        self.pool = enn.NormPool(blocks[-1].out_type)
        pool_out = self.pool.out_type.size
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(pool_out, out_channels, kernel_size=1)
        )

        # input layer + crop of each block
        self.crop = len(blocks) * (kernel_size // 2 - self.padding)
        self.crop = 9

    @staticmethod
    def get_dim(max_freq, spherical=False):
        if spherical:
            return sum([2*l+1 for l in range(max_freq+1)])
        else:
            return sum([(2*l+1)**2 for l in range(max_freq+1)])

    def get_block(self, in_type, out_channels, max_freq=2, **kwargs):
        if self.type == 'trivial':
            return self._get_block_trivial(in_type, out_channels, **kwargs)
        elif self.type in ['spherical', 'so3']:
            return self._get_block_non_trivial(in_type, out_channels, max_freq=max_freq, **kwargs)

    def _get_block_trivial(self, in_type, channels, **kwargs):
        out_type = enn.FieldType(self.gspace, channels*[self.gspace.trivial_repr])
        return enn.SequentialModule(
            enn.R3Conv(in_type, out_type, **kwargs),
            enn.IIDBatchNorm3d(out_type),
            enn.ELU(out_type, inplace=True)
        ), out_type

    def _get_block_non_trivial(self, in_type, out_channels, max_freq=2, **kwargs):
        dim = self.get_dim(max_freq, spherical=self.type=='spherical')
        channels = max(1, out_channels // dim)
        ft_nonlin = FTNonLinearity(max_freq, channels, 'cube', spherical=self.type=='spherical')
        mid_type, out_type = ft_nonlin.in_type, ft_nonlin.out_type
        return enn.SequentialModule(
            enn.R3Conv(in_type, mid_type, **kwargs),
            enn.IIDBatchNorm3d(mid_type),
            ft_nonlin
        ), out_type

    def forward(self, x):
        x = self.pre_forward(x)
        x = self.model(x)
        x = self.pool(x)
        x = x.tensor
        x = self.final(x)
        return x


