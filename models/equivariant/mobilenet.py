import e3cnn.nn as enn
from e3cnn import gspaces
from e3cnn.nn.field_type import FieldType
from e3cnn.group import directsum
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from .base import BaseEquiv, FTNonLinearity, kernel_so3, kernel_sphere


class FTGPool(enn.EquivariantModule):
    def __init__(self, in_type: enn.FieldType, *grid_args, **grid_kwargs):
        f"""
            Sample SO(3)'s irreps -> Take maximum over the samples

            features = (max frequency, number of channels)
            *grid_args, **grid_kwargs = argument to generate the grids over SO(3); see `SO3.grid()`

        """
        super().__init__()

        gspace = gspaces.rot3dOnR3()
        max_freq = max([irr[0] for irr in in_type.irreps])
        channels = len(in_type)
        if all('quotient[(False, -1)]' in rep.name for rep in in_type.representations):
            rho = gspace.fibergroup.bl_quotient_representation(max_freq, (False, -1))
            kernel = kernel_sphere(gspace, max_freq)
        elif all('regular' in rep.name for rep in in_type.representations):
            rho = gspace.fibergroup.bl_regular_representation(max_freq)
            kernel = kernel_so3(max_freq)
        grid = gspace.fibergroup.grid(*grid_args, **grid_kwargs)

        self.dim = rho.size
        self.channels = channels
        self.in_type = enn.FieldType(gspace, channels*[rho])
        self.out_type = enn.FieldType(gspace, self.channels*[gspace.trivial_repr])

        # sensing matrix
        A = np.stack([ kernel @ rho(g) for g in grid ])
        A /= np.sqrt(len(A))

        self.register_buffer('A', torch.tensor(A, dtype=torch.get_default_dtype()))

    def forward(self, x: enn.GeometricTensor):
        assert x.type == self.in_type

        in_shape = (x.shape[0], self.channels, self.dim, *x.shape[2:])
        out_shape = (x.shape[0], self.channels, *x.shape[2:])

        x = x.tensor.view(in_shape)
        _x = torch.einsum( 'gi,bfi...->bfg...', self.A, x)
        y, _ = torch.max(_x, dim=2)
        y = y.reshape(out_shape)

        return enn.GeometricTensor(y, self.out_type)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape[:1] + (self.out_type.size, ) + input_shape[2:]

class SphericalFTElu(enn.EquivariantModule):

    def __init__(self, features: Tuple[int, int], *grid_args, BL: int = None, moorepenrose: bool = True, **grid_kwargs):
        f"""
            Sample spherical harmonics -> Apply ELU -> Fourier Transform

            features = (max frequency, number of channels)
            BL = bandlimit for output (the final fourier transform); if None (by default), use the max_frequency in `features`
            *grid_args, **grid_kwargs = argument to generate the grids over the sphere; see `SO3.sphere_grid()`
            moorepenrose = keep it True

        """
        super(SphericalFTElu, self).__init__()

        F, channels = features
        self.gs = gspaces.rot3dOnR3()
        G: SO3 = self.gs.fibergroup
        rho = G.bl_quotient_representation(F, (False, -1))
        self._C = channels
        self._size = rho.size
        self.in_type = enn.FieldType(self.gs, [rho]*channels)

        kernel = self._kernel_sphere(F).reshape(1, -1)

        if BL is None:
            BL = F

        rho_bl = self.gs.fibergroup.bl_quotient_representation(BL, (False, -1))
        self.out_type = enn.FieldType(self.gs, [rho_bl]*channels)

        grid = self.gs.fibergroup.sphere_grid(*grid_args, **grid_kwargs)

        # sensing matrix
        A = [
            kernel @ rho(g).T for g in grid
        ]

        A = np.concatenate(A, axis=0) / np.sqrt(len(A))

        # reconstruction matrix
        kernel_bl = self._kernel_sphere(BL).reshape(1, -1)
        Abl = [
            kernel_bl @ rho_bl(g).T for g in grid
        ]
        Abl = np.concatenate(Abl, axis=0) / np.sqrt(len(Abl))
        eps = 1e-8
        if moorepenrose:
            A_inv = np.linalg.inv(Abl.T @ Abl + eps * np.eye(Abl.shape[1])) @ Abl.T
        else:
            A_inv = Abl.T

        self.register_buffer('A', torch.tensor(A, dtype=torch.get_default_dtype()))
        self.register_buffer('Ainv', torch.tensor(A_inv, dtype=torch.get_default_dtype()))

    def _kernel_sphere(self, F: int):

        sphere = self.gs.fibergroup.homspace((False, -1))

        basis = []
        for l in range(F+1):
            basis.append(
                sphere.basis(self.gs.fibergroup.identity, (l,), (0,)).flatten() * np.sqrt(2*l+1)
            )
        return np.concatenate(basis)

    def forward(self, x: enn.GeometricTensor):

        assert x.type == self.in_type

        shape = x.shape
        x = x.tensor.view(shape[0], self._C, self._size, *shape[2:])

        _x = torch.einsum(
            'gi,bfi...->bfg...', self.A, x
        )

        _y = torch.nn.functional.elu(_x)

        y = torch.einsum(
            'ig,bfg...->bfi...', self.Ainv, _y
        )

        outshape = (shape[0], self.Ainv.shape[0] * self._C, *shape[2:])
        y = y.reshape(outshape)

        return enn.GeometricTensor(y, self.out_type)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        shape = [*input_shape]
        shape[1] = shape[1] // self.A.shape[1] * self.Ainv.shape[0]
        return tuple(shape)

    def export(self):
        raise NotImplementedError()

class ResBlock(enn.EquivariantModule):
    def __init__(self,
                in_type: FieldType,
                channels: int,
                out_type: FieldType = None,
                kernel_size: int = 3,
                padding: int = 0,
                stride: int = 1,
                initialize=False):
        super().__init__()

        self.in_type = in_type
        self.out_type = out_type or in_type

        # non-linear layer samples the features on the elements of the Icosahedral group
        ftelu = FTNonLinearity(2, channels, 'ico')
        res_type = ftelu.in_type

        self.res_block = enn.SequentialModule(
            enn.R3Conv(in_type, res_type, kernel_size=kernel_size, padding=padding, bias=False, initialize=initialize),
            enn.IIDBatchNorm3d(res_type, affine=True),
            ftelu,
            enn.R3Conv(res_type, self.out_type, kernel_size=kernel_size, padding=padding, stride=stride, bias=False, initialize=initialize),
        )

        if stride > 1:
            self.downsample = enn.PointwiseAvgPoolAntialiased3D(in_type, .33, 2, 1)
        else:
            self.downsample = lambda x: x

        if self.in_type != self.out_type:
            self.skip = enn.R3Conv(self.in_type, self.out_type, kernel_size=1, bias=False, initialize=initialize)
        else:
            self.skip = lambda x: x

    def forward(self, x: enn.GeometricTensor):
        assert x.type == self.in_type
        return self.skip(self.downsample(x)) + self.res_block(x)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if self.in_type != self.out_type:
            return input_shape[:1] + (self.out_type.size, ) + input_shape[2:]
        else:
            return input_shape

    def export(self):
        raise NotImplementedError()


class MobileNetV2(BaseEquiv):
    def __init__(self, in_channels=1, out_channels=1, bb_freq=2, kernel_size=3, padding='same', initialize=True, **kwargs):
        gspace = gspaces.rot3dOnR3()
        super().__init__(gspace, in_channels, kernel_size, padding, **kwargs)

        layer_params = [
                {'type': enn.FieldType(self.gspace, [self.bl_sphere(bb_freq)] * 2),  'channels': 4,    'stride': 1},
                {'type': enn.FieldType(self.gspace, [self.bl_sphere(bb_freq)] * 3),  'channels': 4,    'stride': 1},
                {'type': enn.FieldType(self.gspace, [self.bl_sphere(bb_freq)] * 6), 'channels': 8,    'stride': 1},
                {'type': enn.FieldType(self.gspace, [self.bl_sphere(bb_freq)] * 6), 'channels': 16,   'stride': 1},
                {'type': enn.FieldType(self.gspace, [self.bl_sphere(bb_freq)] * 8), 'channels': 8,    'stride': 1},
                {'type': enn.FieldType(self.gspace, [self.bl_sphere(bb_freq)] * 8), 'channels': 16,   'stride': 1},
                {'type': enn.FieldType(self.gspace, [self.bl_sphere(bb_freq)] * 8), 'channels': None, 'stride': 1},
        ]

        blocks = [
            enn.R3Conv(self.input_type,
                       layer_params[0]['type'],
                       kernel_size=kernel_size,
                       padding=self.padding,
                       stride=2,
                       bias=False,
                       initialize=initialize)
        ]

        for in_params, out_params in zip(layer_params, layer_params[1:]):
            blocks.append( ResBlock(in_params['type'],
                                    in_params['channels'],
                                    out_params['type'],
                                    stride=in_params['stride'],
                                    padding=self.padding,
                                    initialize=initialize) )

        self.model = enn.SequentialModule(*blocks)
        # self.pool = SphericalFTElu((2, 128), 'ico', BL=0)
        self.pool = FTGPool(blocks[-1].out_type, 'cube')
        # self.pool = FTNonLinearity(bb_freq, 8, 'cube', max_freq_out=0)
        pool_out = self.pool.out_type.size
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(pool_out, 1, kernel_size=1)
        )

        self.crop = np.array([32,32,32])

    def bl_sphere(self, max_freq):
        # S^2 = SO(3)/SO(2)
        return self.gspace.fibergroup.bl_quotient_representation(max_freq, (False, -1))

    def forward(self, x):
        x = self.pre_forward(x)
        x = self.model(x)
        x = self.pool(x)
        x = x.tensor
        x = self.final(x)
        return x

