import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseEquiv
from collections import OrderedDict
import e3cnn.nn as enn
import e3cnn.gspaces as gspaces
from typing import Iterable


class CubeUNet(BaseEquiv):
    def __init__(self, *args,
            kernel_size=3,
            padding='same',
            arch='mixed',
            deep_supervision=False,
            initialize=True,
            **kwargs):
        gspace = gspaces.octaOnR3()
        super().__init__(gspace, *args, kernel_size=kernel_size, padding=padding, **kwargs)

        self.kernel_size = kernel_size
        self.deep_supervision = deep_supervision
        self.initialize = initialize

        vert_repr = gspace.fibergroup.cube_vertices_representation
        if arch == 'mixed':
            type32  = enn.FieldType(gspace, 4*[vert_repr])
            type64  = enn.FieldType(gspace, 2*[vert_repr]+2*[gspace.regular_repr])
            type128 = enn.FieldType(gspace, 1*[vert_repr]+5*[gspace.regular_repr])
            type256 = enn.FieldType(gspace, 2*[vert_repr]+10*[gspace.regular_repr])
            type320 = enn.FieldType(gspace, 2*[vert_repr]+13*[gspace.regular_repr])
        elif arch == 'vert':
            type32  = enn.FieldType(gspace, 4*[vert_repr])
            type64  = enn.FieldType(gspace, 8*[vert_repr])
            type128 = enn.FieldType(gspace, 16*[vert_repr])
            type256 = enn.FieldType(gspace, 32*[vert_repr])
            type320 = enn.FieldType(gspace, 40*[vert_repr])
        elif arch == 'regular':
            type32  = enn.FieldType(gspace, 2*[gspace.regular_repr])
            type64  = enn.FieldType(gspace, 3*[gspace.regular_repr])
            type128 = enn.FieldType(gspace, 6*[gspace.regular_repr])
            type256 = enn.FieldType(gspace, 11*[gspace.regular_repr])
            type320 = enn.FieldType(gspace, 14*[gspace.regular_repr])


        self.encoders = nn.ModuleList([
            self.get_encoder(self.input_type, type32, stride=1),
            self.get_encoder(type32, type64),
            self.get_encoder(type64, type128),
            self.get_encoder(type128, type256),
            self.get_encoder(type256, type320),
        ])

        self.bottleneck = self.get_bottleneck(type320)

        self.decoders = nn.ModuleList([
            nn.ModuleDict({
                'upsampler': self.get_upsampler(type320, type320),
                'decoder': self.get_decoder(type320+type320, type320) }),
            nn.ModuleDict({
                'upsampler': self.get_upsampler(type320, type256),
                'decoder': self.get_decoder(type256+type256, type256) }),
            nn.ModuleDict({
                'upsampler': self.get_upsampler(type256, type128),
                'decoder': self.get_decoder(type128+type128, type128) }),
            nn.ModuleDict({
                'upsampler': self.get_upsampler(type128, type64),
                'decoder': self.get_decoder(type64+type64, type64) }),
            nn.ModuleDict({
                'upsampler': self.get_upsampler(type64, type32),
                'decoder': self.get_decoder(type32+type32, type32) }),
        ])


        if self.deep_supervision:
            self.pools = [ enn.GroupPooling(decoder.out_type) for decoder in self.decoders ]
            self.heads = nn.ModuleList([ nn.Conv3d(pool.out_type.size, 1, kernel_size=1) for pool in self.pools ])
        else:
            self.pool = enn.GroupPooling(self.decoders[-1]['decoder'].out_type)
            pool_out = self.pool.out_type.size
            self.final = nn.Conv3d(pool_out, 1, kernel_size=1)

        self.register_buffer('ds_weight',
            torch.FloatTensor([0., 0.06666667, 0.13333333, 0.26666667, 0.53333333])
        )

        self.crop = len(self.encoders) * 2 * (self.kernel_size//2) # 2 conv per encoder
        self.crop = np.array([self.crop, self.crop, self.crop])

    @staticmethod
    def get_nonlin(ftype):
        return enn.ELU(ftype, inplace=True)

    def get_encoder(self, input_type, out_type, stride=2):
        return enn.SequentialModule(OrderedDict({
            'conv1': enn.R3Conv(input_type, out_type, kernel_size=self.kernel_size, stride=stride, padding=self.padding, bias=False, initialize=self.initialize),
            'bnorm1': enn.IIDBatchNorm3d(out_type),
            'nonlin1': enn.ELU(out_type, inplace=True),
            'conv2': enn.R3Conv(out_type, out_type, kernel_size=self.kernel_size, padding=self.padding, bias=False, initialize=self.initialize),
            'bnorm2': enn.IIDBatchNorm3d(out_type),
            'nonlin2': enn.ELU(out_type, inplace=True),
        }))

    def get_decoder(self, input_type, out_type):
        return enn.SequentialModule(OrderedDict({
            'conv1': enn.R3Conv(input_type, out_type, kernel_size=self.kernel_size, padding=self.padding, bias=False, initialize=self.initialize),
            'bnorm1': enn.IIDBatchNorm3d(out_type),
            'nonlin1': enn.ELU(out_type, inplace=True),
            'conv2': enn.R3Conv(out_type, out_type, kernel_size=self.kernel_size, padding=self.padding, bias=False, initialize=self.initialize),
            'bnorm2': enn.IIDBatchNorm3d(out_type),
            'nonlin2': enn.ELU(out_type, inplace=True),
        }))

    def get_upsampler(self, input_type, out_type):
        return enn.SequentialModule(
            enn.R3Upsampling(input_type, scale_factor=2),
            enn.R3Conv(input_type, out_type, kernel_size=1, initialize=self.initialize),
        )

    def get_bottleneck(self, ftype):
        return enn.SequentialModule(OrderedDict({
            'conv1': enn.R3Conv(ftype, ftype, kernel_size=self.kernel_size, stride=2, padding=self.padding, bias=False, initialize=self.initialize),
            'bnorm1': enn.IIDBatchNorm3d(ftype),
            'nonlin1': enn.ELU(ftype, inplace=True),
            'conv2': enn.R3Conv(ftype, ftype, kernel_size=self.kernel_size, padding=self.padding, bias=False, initialize=self.initialize),
            'bnorm2': enn.IIDBatchNorm3d(ftype),
            'nonlin2': enn.ELU(ftype, inplace=True),
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

