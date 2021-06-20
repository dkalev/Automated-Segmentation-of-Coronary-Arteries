import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseEquiv, GatedFieldType
from collections import OrderedDict
import e3cnn.nn as enn
import e3cnn.gspaces as gspaces
from typing import Iterable


class EquivUNet(BaseEquiv):
    def __init__(self, *args, kernel_size=3, padding='same', deep_supervision=False, type='spherical', initialize=True, **kwargs):
        gspace = gspaces.rot3dOnR3()
        super().__init__(gspace, *args, kernel_size=kernel_size, padding=padding, **kwargs)

        self.kernel_size = kernel_size
        self.deep_supervision = deep_supervision
        self.initialize = initialize
        print('type', type)

        type32  = GatedFieldType.build(gspace, 32, type=type)
        type64  = GatedFieldType.build(gspace, 64, type=type)
        type128 = GatedFieldType.build(gspace, 128, type=type)
        type256 = GatedFieldType.build(gspace, 256, type=type)
        type320 = GatedFieldType.build(gspace, 320, type=type)

        self.encoders = nn.ModuleList([
            self.get_encoder(self.input_type, type32, stride=1),
            self.get_encoder(type32.no_gates(), type64),
            self.get_encoder(type64.no_gates(), type128),
            self.get_encoder(type128.no_gates(), type256),
            self.get_encoder(type256.no_gates(), type320),
        ])

        self.bottleneck = self.get_bottleneck(type320)

        self.decoders = nn.ModuleList([
            nn.ModuleDict({
                'upsampler': self.get_upsampler(type320, type320),
                'decoder': self.get_decoder(type320.no_gates()+type320.no_gates(), type320) }),
            nn.ModuleDict({
                'upsampler': self.get_upsampler(type320, type256),
                'decoder': self.get_decoder(type256.no_gates()+type256.no_gates(), type256) }),
            nn.ModuleDict({
                'upsampler': self.get_upsampler(type256, type128),
                'decoder': self.get_decoder(type128.no_gates()+type128.no_gates(), type128) }),
            nn.ModuleDict({
                'upsampler': self.get_upsampler(type128, type64),
                'decoder': self.get_decoder(type64.no_gates()+type64.no_gates(), type64) }),
            nn.ModuleDict({
                'upsampler': self.get_upsampler(type64, type32),
                'decoder': self.get_decoder(type32.no_gates()+type32.no_gates(), type32) }),
        ])

        self.pool = enn.NormPool(self.decoders[-1]['decoder'].out_type)
        pool_out = self.pool.out_type.size

        if self.deep_supervision:
            self.heads = nn.ModuleList([
                nn.Conv3d(320, 1, kernel_size=1, bias=False),
                nn.Conv3d(256, 1, kernel_size=1, bias=False),
                nn.Conv3d(128, 1, kernel_size=1, bias=False),
                nn.Conv3d(64, 1, kernel_size=1, bias=False),
                nn.Conv3d(32, 1, kernel_size=1, bias=False),
            ])
        else:
            self.final = nn.Conv3d(pool_out, 1, kernel_size=1, bias=False)

        self.register_buffer('ds_weight',
            torch.FloatTensor([0., 0.06666667, 0.13333333, 0.26666667, 0.53333333])
        )

        self.crop = len(self.encoders) * 2 * (self.kernel_size//2) # 2 conv per encoder
        self.crop = np.array([self.crop, self.crop, self.crop])

    @staticmethod
    def get_nonlin(ftype):
        if ftype.gated and ftype.gates:
            return enn.MultipleModule(ftype,
                labels=[
                    *( len(ftype.trivials) * ['trivial'] + (len(ftype.gated) + len(ftype.gates)) * ['gate'] )
                ],
                modules=[
                    (enn.ELU(ftype.trivials, inplace=True), 'trivial'),
                    (enn.GatedNonLinearity1(ftype.gated+ftype.gates, len(ftype.gated)*['gated']+len(ftype.gates)*['gate']), 'gate')
                ]
            )
        else:
            print('trivial nonlin')
            return enn.ELU(ftype, inplace=True)

    def get_encoder(self, input_type, out_type, stride=2):
        return enn.SequentialModule(OrderedDict({
            'conv1': enn.R3Conv(input_type, out_type, kernel_size=self.kernel_size, stride=stride, padding=self.padding, initialize=self.initialize),
            'bnorm1': enn.IIDBatchNorm3d(out_type),
            'nonlin1': self.get_nonlin(out_type),
            'conv2': enn.R3Conv(out_type.no_gates(), out_type, kernel_size=self.kernel_size, padding=self.padding, initialize=self.initialize),
            'bnorm2': enn.IIDBatchNorm3d(out_type),
            'nonlin2': self.get_nonlin(out_type),
        }))

    def get_decoder(self, input_type, out_type):
        return enn.SequentialModule(OrderedDict({
            'conv1': enn.R3Conv(input_type, out_type, kernel_size=self.kernel_size, padding=self.padding, initialize=self.initialize),
            'bnorm1': enn.IIDBatchNorm3d(out_type),
            'nonlin1': self.get_nonlin(out_type),
            'conv2': enn.R3Conv(out_type.no_gates(), out_type, kernel_size=self.kernel_size, padding=self.padding, initialize=self.initialize),
            'bnorm2': enn.IIDBatchNorm3d(out_type),
            'nonlin2': self.get_nonlin(out_type),
        }))

    def get_upsampler(self, input_type, out_type):
        return enn.SequentialModule(
            enn.R3Upsampling(input_type.no_gates(), scale_factor=2),
            enn.R3Conv(input_type.no_gates(), out_type.no_gates(), kernel_size=1, initialize=self.initialize),
        )

    def get_bottleneck(self, ftype):
        return enn.SequentialModule(OrderedDict({
            'conv1': enn.R3Conv(ftype.no_gates(), ftype, kernel_size=self.kernel_size, stride=2, padding=self.padding, initialize=self.initialize),
            'bnorm1': enn.IIDBatchNorm3d(ftype),
            'nonlin1': self.get_nonlin(ftype),
            'conv2': enn.R3Conv(ftype.no_gates(), ftype, kernel_size=self.kernel_size, padding=self.padding, initialize=self.initialize),
            'bnorm2': enn.IIDBatchNorm3d(ftype),
            'nonlin2': self.get_nonlin(ftype),
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

        x = self.pool(x)
        x = x.tensor
        if self.deep_supervision:
            outputs = [ head(out) for head, out in zip(self.heads, outputs) ]
            outputs = [ F.interpolate(out, size=targ_size) for out in outputs ]
            return outputs
        else:
            return self.final(x)

