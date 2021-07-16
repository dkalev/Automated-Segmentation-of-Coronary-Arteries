from abc import abstractmethod
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseEquiv
from collections import OrderedDict
import e3cnn.nn as enn
from typing import Iterable


class EquivUNet(BaseEquiv):
    def __init__(self, *args,
            kernel_size=3,
            padding='same',
            deep_supervision=False,
            initialize=True,
            **kwargs):
        gspace = self.gspace
        super().__init__(gspace, *args, kernel_size=kernel_size, padding=padding, **kwargs)

        self.kernel_size = kernel_size
        self.deep_supervision = deep_supervision
        self.initialize = initialize

    @property
    def crop(self):
        if not hasattr(self, '_crop'):
            crop = len(self.encoders) * 2 * (self.kernel_size//2) # 2 conv per encoder
            self._crop = np.array(3*[crop])
        return self._crop
    
    @staticmethod
    @abstractmethod
    def get_nonlin(ftype): pass

    @staticmethod
    @abstractmethod
    def get_pooling(ftype): pass

    def build_model(self, layer_params):
        self.encoders = nn.ModuleList([
            self.get_encoder(p['in_type'], p['out_type'], mid_type=p.get('mid_type'), stride=p['stride'])
            for p in layer_params['encoders']
        ])

        self.bottleneck = self.get_bottleneck(layer_params['bottleneck']['in_type'], layer_params['bottleneck']['out_type'])

        self.decoders = nn.ModuleList([
            nn.ModuleDict({
                'upsampler': self.get_upsampler(p['in_type'], p['mid_type']),
                'decoder': self.get_decoder(p['mid_type']+p['mid_type'], p['out_type'], mid_type=p.get('mid_type'))
            })
            for p in layer_params['decoders']
        ])

        if self.deep_supervision:
            self.register_buffer('ds_weight',
                torch.FloatTensor([0., 0.06666667, 0.13333333, 0.26666667, 0.53333333])
            )
            self.pools = [ self.get_pooling(decoder.out_type) for decoder in self.decoders ]
            self.heads = nn.ModuleList([ nn.Conv3d(pool.out_type.size, 1, kernel_size=1) for pool in self.pools ])
        else:
            self.pool = self.get_pooling(self.decoders[-1]['decoder'].out_type)
            pool_out = self.pool.out_type.size
            self.final = nn.Conv3d(pool_out, 1, kernel_size=1)

    def get_encoder(self, input_type, out_type, mid_type=None, stride=2):
        if mid_type is None: mid_type = out_type
        return enn.SequentialModule(OrderedDict({
            'conv1': enn.R3Conv(input_type, out_type, kernel_size=self.kernel_size, stride=stride, padding=self.padding, bias=False, initialize=self.initialize),
            'bnorm1': enn.IIDBatchNorm3d(out_type),
            'nonlin1': self.get_nonlin(out_type),
            'conv2': enn.R3Conv(mid_type, out_type, kernel_size=self.kernel_size, padding=self.padding, bias=False, initialize=self.initialize),
            'bnorm2': enn.IIDBatchNorm3d(out_type),
            'nonlin2': self.get_nonlin(out_type),
        }))

    def get_decoder(self, input_type, out_type, mid_type=None):
        if mid_type is None: mid_type = out_type
        return enn.SequentialModule(OrderedDict({
            'conv1': enn.R3Conv(input_type, out_type, kernel_size=self.kernel_size, padding=self.padding, bias=False, initialize=self.initialize),
            'bnorm1': enn.IIDBatchNorm3d(out_type),
            'nonlin1': self.get_nonlin(out_type),
            'conv2': enn.R3Conv(mid_type, out_type, kernel_size=self.kernel_size, padding=self.padding, bias=False, initialize=self.initialize),
            'bnorm2': enn.IIDBatchNorm3d(out_type),
            'nonlin2': self.get_nonlin(out_type),
        }))

    def get_upsampler(self, input_type, out_type):
        return enn.SequentialModule(
            enn.R3Upsampling(input_type, scale_factor=2),
            enn.R3Conv(input_type, out_type, kernel_size=1, initialize=self.initialize),
        )

    def get_bottleneck(self, in_type, out_type):
        return enn.SequentialModule(OrderedDict({
            'conv1': enn.R3Conv(in_type, out_type, kernel_size=self.kernel_size, stride=2, padding=self.padding, bias=False, initialize=self.initialize),
            'bnorm1': enn.IIDBatchNorm3d(out_type),
            'nonlin1': self.get_nonlin(out_type),
            'conv2': enn.R3Conv(in_type, out_type, kernel_size=self.kernel_size, padding=self.padding, bias=False, initialize=self.initialize),
            'bnorm2': enn.IIDBatchNorm3d(out_type),
            'nonlin2': self.get_nonlin(out_type),
        }))

    @staticmethod
    def cat(gtensors:Iterable[enn.GeometricTensor], *args, **kwargs) -> enn.GeometricTensor:
        tensors = [ t.tensor for t in gtensors ]
        tensors_cat = torch.cat(tensors, *args, **kwargs)
        feature_type = sum([t.type for t in gtensors[1:]], start=gtensors[0].type)
        return enn.GeometricTensor(tensors_cat, feature_type)

    def forward(self, x):
        x = self.pre_forward(x)
        targ_size = x.shape[2:]

        skip_cons = []
        for encoder in self.encoders:
            x = encoder(x)
            print(x.shape)
            skip_cons.append(x)

        x = self.bottleneck(x)

        outputs = []
        for i, block in enumerate(self.decoders):
            x = block['upsampler'](x)
            # GatedFieldType is lost in the MultipleModule used in the nonlinearities
            # assign the correct one from the input x
            skip = skip_cons[-(i+1)]
            skip.type = x.type
            x = self.cat([x, skip], dim=1)
            x = block['decoder'](x)
            print(x.shape)
            if self.deep_supervision:
                outputs.append(x)

        if self.deep_supervision:
            outputs = [ pool(out).tensor for pool, out in zip(self.pools, outputs) ]
            outputs = [ head(out) for head, out in zip(self.heads, outputs) ]
            outputs = [ F.interpolate(out, size=targ_size) for out in outputs ]
            return outputs
        else:
            x = self.pool(x)
            x = x.tensor
            return self.final(x)

