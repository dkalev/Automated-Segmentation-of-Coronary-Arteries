import e3cnn.nn as enn
from e3cnn import gspaces
from e3cnn.nn.field_type import FieldType
from e3cnn.group import directsum
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from .base import Base


class BaseEquiv(Base):
    def __init__(self, gspace, in_channels=1, kernel_size=3, padding=0, initialize=False, **kwargs):
        super().__init__(**kwargs)

        self.gspace = gspace
        self.padding = self.parse_padding(padding, kernel_size)
        self.input_type = enn.FieldType(self.gspace, in_channels*[self.gspace.trivial_repr])

    def parse_padding(self, padding, kernel_size):
        if isinstance(padding, int):
            return padding
        elif isinstance(padding, tuple) and len(padding) == 3 and all(type(p)==int for p in padding):
            return padding
        elif padding == 'same':
            return kernel_size // 2
        else:
            raise ValueError(f'Parameter padding must be int, tuple, or "same. Given: {padding}')

    def init(self):
        # FIXME initialize the rest of the modules when starting to use them
        for m in self.modules():
            if isinstance(m, enn.R3Conv):
                m.weights.data = torch.randn_like(m.weights)

    def pre_forward(self, x):
        if isinstance(x, torch.Tensor):
            x = enn.GeometricTensor(x, self.input_type)
        assert x.type == self.input_type
        return x

    def get_memory_req_est(self, batch_shape):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        buffers = sum(n.numel() for b in self.buffers())
        total = 2 * trainable_params + non_trainable_params + buffers
        return total

    def evaluate_output_shape(self, input_shape):
        return input_shape[..., self.crop:-self.crop, self.crop:-self.crop, self.crop: -self.crop]


class CubeRegCNN(BaseEquiv):
    def __init__(self, in_channels=1, out_channels=1, max_freq=2, kernel_size=3, padding=0, initialize=True, **kwargs):
        gspace = gspaces.octaOnR3()
        super().__init__(gspace, in_channels, kernel_size, padding, **kwargs)

        small_type = enn.FieldType(self.gspace, 2*[self.gspace.regular_repr])
        mid_type   = enn.FieldType(self.gspace, 8*[self.gspace.regular_repr])

        self.model = enn.SequentialModule(
            enn.R3Conv(self.input_type, small_type, kernel_size=kernel_size, padding=self.padding, bias=False, initialize=initialize),
            enn.IIDBatchNorm3d(small_type),
            enn.ELU(small_type, inplace=True),
            enn.R3Conv(small_type, small_type, kernel_size=kernel_size, padding=self.padding, bias=False, initialize=initialize),
            enn.IIDBatchNorm3d(small_type),
            enn.ELU(small_type, inplace=True),
            enn.R3Conv(small_type, mid_type, kernel_size=kernel_size, padding=self.padding, bias=False, initialize=initialize),
            enn.IIDBatchNorm3d(mid_type),
            enn.ELU(mid_type, inplace=True),
            enn.R3Conv(mid_type, small_type, kernel_size=kernel_size, padding=self.padding, bias=False, initialize=initialize),
            enn.IIDBatchNorm3d(small_type),
            enn.ELU(small_type, inplace=True),
        )
        self.pool = enn.GroupPooling(small_type)
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


class IcoRegCNN(BaseEquiv):
    def __init__(self, in_channels=1, out_channels=1, max_freq=2, kernel_size=3, padding=0, initialize=True, **kwargs):
        gspace = gspaces.icoOnR3()
        super().__init__(gspace, in_channels, kernel_size, padding, **kwargs)

        small_type = enn.FieldType(self.gspace, 1*[self.gspace.regular_repr])
        mid_type   = enn.FieldType(self.gspace, 4*[self.gspace.regular_repr])

        self.model = enn.SequentialModule(
            enn.R3Conv(self.input_type, small_type, kernel_size=kernel_size, padding=self.padding, bias=False, initialize=initialize),
            enn.IIDBatchNorm3d(small_type),
            enn.ELU(small_type, inplace=True),
            enn.R3Conv(small_type, small_type, kernel_size=kernel_size, padding=self.padding, bias=False, initialize=initialize),
            enn.IIDBatchNorm3d(small_type),
            enn.ELU(small_type, inplace=True),
            enn.R3Conv(small_type, mid_type, kernel_size=kernel_size, padding=self.padding, bias=False, initialize=initialize),
            enn.IIDBatchNorm3d(mid_type),
            enn.ELU(mid_type, inplace=True),
            enn.R3Conv(mid_type, small_type, kernel_size=kernel_size, padding=self.padding, bias=False, initialize=initialize),
            enn.IIDBatchNorm3d(small_type),
            enn.ELU(small_type, inplace=True),
        )
        self.pool = enn.GroupPooling(small_type)
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

