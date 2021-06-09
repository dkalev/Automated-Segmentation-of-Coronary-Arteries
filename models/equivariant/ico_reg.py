import e3cnn.nn as enn
from e3cnn import gspaces
from e3cnn.group import directsum
import torch
import torch.nn as nn
from typing import Tuple
from .base import BaseEquiv


class IcoRegCNN(BaseEquiv):
    def __init__(self, in_channels=1, out_channels=1, max_freq=2, kernel_size=3, padding=0, initialize=True, **kwargs):
        gspace = gspaces.icoOnR3()
        super().__init__(gspace, in_channels, kernel_size, padding, **kwargs)

        small_type = enn.FieldType(self.gspace, 4*[self.gspace.regular_repr])
        mid_type   = enn.FieldType(self.gspace, 8*[self.gspace.regular_repr])
        large_type = enn.FieldType(self.gspace, 16*[self.gspace.regular_repr])

        self.model = enn.SequentialModule(
            enn.R3Conv(self.input_type, small_type, kernel_size=kernel_size, padding=self.padding, bias=False, initialize=initialize),
            enn.IIDBatchNorm3d(small_type),
            enn.ELU(small_type, inplace=True),
            enn.R3Conv(small_type, mid_type, kernel_size=kernel_size, padding=self.padding, bias=False, initialize=initialize),
            enn.IIDBatchNorm3d(mid_type),
            enn.ELU(mid_type, inplace=True),
            enn.R3Conv(mid_type, large_type, kernel_size=kernel_size, padding=self.padding, bias=False, initialize=initialize),
            enn.IIDBatchNorm3d(large_type),
            enn.ELU(large_type, inplace=True),
            enn.R3Conv(large_type, small_type, kernel_size=kernel_size, padding=self.padding, bias=False, initialize=initialize),
            enn.IIDBatchNorm3d(small_type),
            enn.ELU(small_type, inplace=True),
        )
        self.pool = enn.NormPool(small_type)
        pool_out = len(small_type.representations)
        self.final = nn.Conv3d(pool_out, out_channels, kernel_size=1)

        # input layer + crop of each block
        self.crop = 4 * (kernel_size // 2 - self.padding)

    def forward(self, x):
        x = self.pre_forward(x)
        x = self.model(x)
        x = self.pool(x)
        x = x.tensor
        x = self.final(x)
        return x

