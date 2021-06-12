import e3cnn.nn as enn
import torch
from collections import Counter
from e3cnn.group import directsum
from typing import Tuple
from ..base import Base


class GatedFieldType(enn.FieldType):
    def __init__(self, gspace, trivials, gated, gates):
        self.trivials = trivials
        self.gated = gated
        self.gates = gates

        super().__init__(gspace, (self.trivials + self.gated + self.gates).representations)

    def no_gates(self):
        return enn.FieldType(self.gspace, (self.trivials + self.gated).representations)

    @classmethod
    def build(cls, gspace, channels, max_freq=2):
        dim = sum([2*l+1 for l in range(max_freq+1)]) + 1 # + 1 for gate per directsum of higher frequencies
        n_irreps, n_rem = channels // dim, channels % dim
        n_triv = n_irreps + n_rem

        trivials = enn.FieldType(gspace, n_triv*[gspace.trivial_repr])
        gated = enn.FieldType(gspace, n_irreps*[directsum([gspace.irrep(i) for i in range(1,max_freq+1)])])
        gates = enn.FieldType(gspace, n_irreps*[gspace.trivial_repr])

        return cls(gspace, trivials, gated, gates)

    def __add__(self, other: 'GatedFieldType') -> 'GatedFieldType':
        assert self.gspace == other.gspace
        return GatedFieldType(self.gspace, self.trivials + other.trivials, self.gated + other.gated, self.gates + other.gates)


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

