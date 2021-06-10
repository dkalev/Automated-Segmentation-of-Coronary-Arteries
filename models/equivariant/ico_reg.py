import e3cnn.nn as enn
from e3cnn import gspaces
from e3cnn.group import directsum
import torch
import torch.nn as nn
from typing import Tuple
from functools import partial
from .base import BaseEquiv


class IcoRegCNN(BaseEquiv):
    def __init__(self, in_channels=1, out_channels=1, max_freq=2, kernel_size=3, padding=0, initialize=True, **kwargs):
        gspace = gspaces.icoOnR3()
        super().__init__(gspace, in_channels, kernel_size, padding, **kwargs)

        small_type = enn.FieldType(self.gspace, 4*[self.gspace.regular_repr])
        mid_type   = enn.FieldType(self.gspace, 32*[self.gspace.regular_repr])
        final_type = enn.FieldType(self.gspace, 16*[self.gspace.fibergroup.ico_vertices_representation])

        R3Conv = partial(enn.R3Conv, kernel_size=kernel_size, padding=self.padding, bias=False, initialize=initialize)

        self.model = enn.SequentialModule(
            R3Conv(self.input_type, small_type, stride=2),
            enn.IIDBatchNorm3d(small_type),
            enn.ELU(small_type, inplace=True),
            R3Conv(small_type, small_type),
            enn.IIDBatchNorm3d(small_type),
            enn.ELU(small_type, inplace=True),
            R3Conv(small_type, mid_type),
            enn.IIDBatchNorm3d(mid_type),
            enn.ELU(mid_type, inplace=True),
            R3Conv(mid_type, final_type),
            enn.IIDBatchNorm3d(final_type),
            enn.ELU(final_type, inplace=True),
        )
        self.pool = enn.GroupPooling(final_type)
        pool_out = len(final_type.representations)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(pool_out, out_channels, kernel_size=1)
        )

        # input layer + crop of each block
        self.crop = 4 * (kernel_size // 2 - self.padding)
        self.crop = 7

    def forward(self, x):
        x = self.pre_forward(x)
        x = self.model(x)
        x = self.pool(x)
        x = x.tensor
        x = self.final(x)
        return x

