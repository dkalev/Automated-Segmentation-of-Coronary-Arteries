import e3cnn.nn as enn
from e3cnn import gspaces
from e3cnn.group import directsum
import torch
import torch.nn as nn
from typing import Tuple
from .base import BaseEquiv, GatedFieldType


class SteerableCNN(BaseEquiv):
    def __init__(self, in_channels=1, out_channels=1, max_freq=2, kernel_size=3, padding=0, initialize=True, **kwargs):
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

