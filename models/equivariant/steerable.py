import e3cnn.nn as enn
from e3cnn import gspaces
from e3cnn.group import directsum
import torch
import torch.nn as nn
from typing import Tuple
from .base import BaseEquiv


class SteerableCNN(BaseEquiv):
    def __init__(self, in_channels=1, out_channels=1, max_freq=2, kernel_size=3, padding=0, initialize=True, **kwargs):
        gspace = gspaces.rot3dOnR3()
        super().__init__(gspace, in_channels, kernel_size, padding, **kwargs)

        small_type = self.get_field_type(4)
        mid_type = self.get_field_type(10)

        common_kwargs = {
                'kernel_size': kernel_size,
                'padding': self.padding,
                'initialize': initialize,
        }
        blocks = [
            self.get_block(self.input_type, small_type, **common_kwargs),
            self.get_block(small_type[0], small_type, **common_kwargs),
            self.get_block(small_type[0], mid_type, **common_kwargs),
            self.get_block(mid_type[0], small_type, **common_kwargs),
        ]

        self.model = enn.SequentialModule(*blocks)
        self.pool = enn.NormPool(blocks[-1].out_type)
        pool_out = self.pool.out_type.size
        self.final = nn.Conv3d(pool_out, out_channels, kernel_size=1)
        # input layer + crop of each block
        self.crop = len(blocks) * (kernel_size // 2 - self.padding)

    def get_field_type(self, channels, max_freq=2):
        field_type_tr = enn.FieldType(self.gspace, channels*[self.gspace.trivial_repr])
        field_type_gated = enn.FieldType(self.gspace, channels*[directsum([self.gspace.irrep(i) for i in range(1,max_freq+1)])])
        field_type_gates = enn.FieldType(self.gspace, channels*[self.gspace.trivial_repr])
        field_type = field_type_tr + field_type_gates + field_type_gated
        return field_type, (field_type_tr, field_type_gates, field_type_gated)

    def get_block(self, in_type, out_type, **kwargs):
        out_type, (out_type_tr, out_type_gates, out_type_gated) = out_type

        layers = []

        layers.append( enn.R3Conv(in_type, out_type, **kwargs) )
        layers.append( enn.IIDBatchNorm3d(out_type) )
        layers.append(
            enn.MultipleModule(out_type,
                labels=[
                    *( len(out_type_tr) * ['trivial'] + (len(out_type_gates) + len(out_type_gated)) * ['gate'] )
                ],
                modules=[
                    (enn.ELU(out_type_tr, inplace=True), 'trivial'),
                    (enn.GatedNonLinearity1(out_type_gates+out_type_gated, drop_gates=False), 'gate')
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

