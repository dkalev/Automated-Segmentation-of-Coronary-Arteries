import e3cnn.nn as enn
from e3cnn import gspaces
from e3cnn.group import directsum
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from functools import partial
from .base import BaseEquiv


class IcoRegCNN(BaseEquiv):
    def __init__(self, in_channels=1, out_channels=1, max_freq=2, kernel_size=3, padding=0, initialize=True, **kwargs):
        gspace = gspaces.icoOnR3()
        super().__init__(gspace, in_channels, kernel_size, padding, **kwargs)

        type3 = enn.FieldType(self.gspace, 3*[self.gspace.regular_repr])
        type6 = enn.FieldType(self.gspace, 6*[self.gspace.regular_repr])
        final_type = enn.FieldType(self.gspace, 36*[self.gspace.fibergroup.ico_vertices_representation])

        get_block = partial(self.get_block, kernel_size=kernel_size, padding=self.padding, bias=False, initialize=initialize)

        blocks = [
            get_block(self.input_type, type3, stride=2),
            get_block(type3, type6),
            get_block(type6, type6),
            get_block(type6, type6),
            get_block(type6, type6),
            get_block(type6, final_type),
        ]

        self.model = enn.SequentialModule( *blocks )
        self.pool = enn.GroupPooling(final_type)
        pool_out = len(final_type.representations)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(pool_out, out_channels, kernel_size=1)
        )

        # FIXME hardcoded assuming input of shape [128,128,128], stride=2 in first conv and upsampling after
        crop_per_layer = kernel_size // 2
        input_shape = np.array([128,128,128])
        output_shape = ((input_shape - 2*crop_per_layer) // 2 - len(blocks[1:]) * 2 * crop_per_layer) * 2
        self.crop = (input_shape - output_shape) // 2

    @staticmethod
    def get_block(in_type, out_type, **kwargs):
        return enn.SequentialModule(
            enn.R3Conv(in_type, out_type, **kwargs),
            enn.IIDBatchNorm3d(out_type),
            enn.ELU(out_type, inplace=True),
        )

    def forward(self, x):
        x = self.pre_forward(x)
        x = self.model(x)
        x = self.pool(x)
        x = x.tensor
        x = self.final(x)
        return x
