import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseEquiv
from collections import OrderedDict
import e3cnn.nn as enn
import e3cnn.gspaces as gspaces
from e3cnn.group import directsum
from typing import Iterable


class EquivUNet(BaseEquiv):
    def __init__(self, *args, kernel_size=3, deep_supervision=False, initialize=True, **kwargs):
        gspace = gspaces.rot3dOnR3()
        super().__init__(gspace, *args, **kwargs)

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.deep_supervision = deep_supervision
        self.initialize = initialize

        type32  = self.get_field_type(32)
        type64  = self.get_field_type(64)
        type128 = self.get_field_type(128)
        type256 = self.get_field_type(256)
        type320 = self.get_field_type(320)
        type512 = self.get_field_type(512)
        type640 = self.get_field_type(640)
        type256_dec = self.get_field_type(256, dec=True)
        type512_dec = self.get_field_type(512, dec=True)
        self.types = [type32, type64, type128, type256, type512, type640]


        self.encoders = nn.ModuleList([
            self.get_encoder((self.input_type,), type32, stride=1),
            self.get_encoder(type32, type64),
            self.get_encoder(type64, type128),
            self.get_encoder(type128, type256),
            self.get_encoder(type256, type320),
        ])

        self.bottleneck = self.get_bottleneck(type320)

        self.decoders = nn.ModuleList([
            nn.ModuleDict({ 'upsampler': self.get_upsampler(type320, type320), 'decoder': self.get_decoder(type640, type320) }),
            nn.ModuleDict({ 'upsampler': self.get_upsampler(type320, type256), 'decoder': self.get_decoder(type512_dec, type256) }),
            nn.ModuleDict({ 'upsampler': self.get_upsampler(type256, type128), 'decoder': self.get_decoder(type256_dec, type128) }),
            nn.ModuleDict({ 'upsampler': self.get_upsampler(type128, type64), 'decoder': self.get_decoder(type128, type64) }),
            nn.ModuleDict({ 'upsampler': self.get_upsampler(type64, type32), 'decoder': self.get_decoder(type64, type32) }),
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

        self.crop = len(self.encoders) * 2 * self.padding # 2 conv per encoder

    def get_field_type(self, channels, max_freq=2, dec=False):
        """get_field_type.

        :param channels:
        :param max_freq:
        :param dec:
        """
        dim = sum([2*l+1 for l in range(max_freq+1)]) + 1 # + 1 for gate per directsum of higher frequencies
        if dec:
            n_irreps = ((channels//2) // dim)*2
            n_rem = ((channels//2) % dim )*2
        else:
            n_irreps, n_rem = channels // dim, channels % dim
        n_triv = n_irreps + n_rem
        field_type_tr = enn.FieldType(self.gspace, n_triv*[self.gspace.trivial_repr])
        field_type_gated = enn.FieldType(self.gspace, n_irreps*[directsum([self.gspace.irrep(i) for i in range(1,max_freq+1)])])
        field_type_gates = enn.FieldType(self.gspace, n_irreps*[self.gspace.trivial_repr])
        field_type = field_type_tr + field_type_gates + field_type_gated
        return field_type, (field_type_tr, field_type_gates, field_type_gated)

    @staticmethod
    def get_nonlin(ftype):
        ftype, (ftype_tr, ftype_gates, ftype_gated) = ftype
        return enn.MultipleModule(ftype,
            labels=[
                *( len(ftype_tr) * ['trivial'] + (len(ftype_gates) + len(ftype_gated)) * ['gate'] )
            ],
            modules=[
                (enn.ELU(ftype_tr, inplace=True), 'trivial'),
                (enn.GatedNonLinearity1(ftype_gates+ftype_gated, drop_gates=False), 'gate')
            ]
        )

    def get_encoder(self, input_type, out_type, stride=2):
        return enn.SequentialModule(OrderedDict({
            'conv1': enn.R3Conv(input_type[0], out_type[0], kernel_size=self.kernel_size, stride=stride, padding=self.padding, initialize=self.initialize),
            'bnorm1': enn.IIDBatchNorm3d(out_type[0]),
            'nonlin1': self.get_nonlin(out_type),
            'conv2': enn.R3Conv(out_type[0], out_type[0], kernel_size=self.kernel_size, padding=self.padding, initialize=self.initialize),
            'bnorm2': enn.IIDBatchNorm3d(out_type[0]),
            'nonlin2': self.get_nonlin(out_type),
        }))

    def get_decoder(self, input_type, out_type):
        return enn.SequentialModule(OrderedDict({
            'conv1': enn.R3Conv(input_type[0], out_type[0], kernel_size=self.kernel_size, padding=self.padding, initialize=self.initialize),
            'bnorm1': enn.IIDBatchNorm3d(out_type[0]),
            'nonlin1': self.get_nonlin(out_type),
            'conv2': enn.R3Conv(out_type[0], out_type[0], kernel_size=self.kernel_size, padding=self.padding, initialize=self.initialize),
            'bnorm2': enn.IIDBatchNorm3d(out_type[0]),
            'nonlin2': self.get_nonlin(out_type),
        }))

    def get_upsampler(self, input_type, out_type):
        return enn.SequentialModule(
            enn.R3Upsampling(input_type[0], scale_factor=2),
            enn.R3Conv(input_type[0], out_type[0], kernel_size=1, initialize=self.initialize),
        )

    def get_bottleneck(self, ftype):
        return enn.SequentialModule(OrderedDict({
            'conv1': enn.R3Conv(ftype[0], ftype[0], kernel_size=self.kernel_size, stride=2, padding=self.padding, initialize=self.initialize),
            'bnorm1': enn.IIDBatchNorm3d(ftype[0]),
            'nonlin1': self.get_nonlin(ftype),
            'conv2': enn.R3Conv(ftype[0], ftype[0], kernel_size=self.kernel_size, padding=self.padding, initialize=self.initialize),
            'bnorm2': enn.IIDBatchNorm3d(ftype[0]),
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
            x = self.cat([x, skip_cons[-(i+1)]], dim=1)
            # print('concat', x.shape, 'x: ', x.type, '\n', skip_cons[-(i+1)].type,  '\n expected: ', self.types[-(i+1)][0])
            x = block['decoder'](x)
            # print('dec', x.shape, x.type)
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

