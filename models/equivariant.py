import e3cnn.nn as enn
from e3cnn import gspaces
from e3cnn.nn.field_type import FieldType
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from .base import Base

def kernel_so3(L: int):
    dim = sum( (2 * l + 1) ** 2 for l in range(L+1))

    V = np.zeros((dim, 1))
    
    p = 0
    for l in range(L + 1):
        d = (2 * l + 1)
        V[p:p + d ** 2, 0] = np.eye(d).reshape(-1) * d
        p += d ** 2

    V /= np.linalg.norm(V.reshape(-1))
    return V


class FTNonLinearity(enn.EquivariantModule):
    def __init__(self, 
                 F,
                 channels,
                 *grid_args, 
                 non_linearity='elu',
                 BL: int = None, 
                 moorepenrose: bool = True, 
                 **grid_kwargs):
        f"""
            Sample SO(3)'s irreps -> Apply ELU -> Fourier Transform

            features = (max frequency, number of channels)
            BL = bandlimit for output (the final fourier transform); if None (by default), use the max_frequency in `features`
            *grid_args, **grid_kwargs = argument to generate the grids over SO(3); see `SO3.grid()`
            moorepenrose = keep it True

        """
        super().__init__()

        gs = gspaces.rot3dOnR3()
        G = gs.fibergroup
        rho = G.bl_regular_representation(F)
        self._C = channels
        self.in_type = enn.FieldType(gs, [rho]*channels)

        if hasattr(torch.nn.functional, non_linearity):
            self.non_linearity = getattr(torch.nn.functional, non_linearity)
        else:
            raise ValueError(f'Unsupported non-linearity type: {non_linearity}')

        kernel = kernel_so3(F).T

        if BL is None: BL = F

        self._size = rho.size

        rho_bl = gs.fibergroup.bl_regular_representation(BL)
        self.out_type = enn.FieldType(gs, [rho_bl]*self._C)
        grid = gs.fibergroup.grid(*grid_args, **grid_kwargs)

        # sensing matrix
        A = [
            kernel @ rho(g).T for g in grid
        ]

        A = np.concatenate(A, axis=0) / np.sqrt(len(A))

        # reconstruction matrix
        kernel_bl = kernel_so3(BL).T
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

    def forward(self, x: enn.GeometricTensor):
        assert x.type == self.in_type

        shape = x.shape
        x = x.tensor.view(shape[0], self._C, self._size, *shape[2:])

        _x = torch.einsum( 'gi,bfi...->bfg...', self.A, x)

        _y = self.non_linearity(_x, inplace=True)

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


class ConvBlock3D(enn.EquivariantModule):
    def __init__(self, in_type: enn.FieldType,
                       out_type: enn.FieldType,
                       kernel_size:int=3,
                       padding:int=0,
                       sample_type='ico',
                       nonlin_type= 'elu',
                       **kwargs):
        super().__init__()
        channels = len(out_type.representations)
        self.in_type = in_type
        self.block = enn.SequentialModule(
            enn.R3Conv(in_type, out_type, kernel_size=kernel_size, padding=padding, **kwargs),
            enn.IIDBatchNorm3d(out_type),
            FTNonLinearity(2, channels, non_linearity=nonlin_type, type=sample_type)
        )
        
        self.crop = kernel_size // 2 - padding
        self.out_channels = out_type.size
        self.out_type = out_type # needs to be there when used in enn.SequentialModule
    
    def forward(self, x):
        assert x.type == self.in_type
        return self.block(x)
    
    def evaluate_output_shape(self, input_shape):
        bs, _, w, h, d = input_shape
        return bs, self.out_channels, w-2*self.crop, h-2*self.crop, d-2*self.crop
        

class FTGPool(enn.EquivariantModule):
    def __init__(self, F: int, channels: int, *grid_args, **grid_kwargs):
        f"""
            Sample SO(3)'s irreps -> Take maximum over the samples

            features = (max frequency, number of channels)
            *grid_args, **grid_kwargs = argument to generate the grids over SO(3); see `SO3.grid()`

        """
        super().__init__()

        gs = gspaces.rot3dOnR3()
        rho = gs.fibergroup.bl_regular_representation(F)
        self._C = channels
        self.in_type = enn.FieldType(gs, channels*[rho])

        kernel = kernel_so3(F).T

        self._size = rho.size
        self.out_type = enn.FieldType(gs, self._C*[gs.trivial_repr])

        grid = gs.fibergroup.grid(*grid_args, **grid_kwargs)

        # sensing matrix
        A = [ kernel @ rho(g) for g in grid ]
        A = np.concatenate(A, axis=0) / np.sqrt(len(A))

        self.register_buffer('A', torch.tensor(A, dtype=torch.get_default_dtype()))

    def forward(self, x: enn.GeometricTensor):
        assert x.type == self.in_type

        shape = x.shape
        x = x.tensor.view(shape[0], self._C, self._size, *shape[2:])

        _x = torch.einsum(
            'gi,bfi...->bfg...', self.A, x
        )

        y, _ = torch.max(_x, dim=2)

        y = y.reshape(shape[0], self._C, *shape[2:])

        return enn.GeometricTensor(y, self.out_type)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape[:1] + (self.out_type.size, ) + input_shape[2:]


# class BaselineRegularCNN(Base, enn.EquivariantModule):
class BaselineRegularCNN(Base):
    def __init__(self, in_channels=1, out_channels=1, max_freq=2, kernel_size=3, padding=0, **kwargs):
        super().__init__(**kwargs)

        if isinstance(padding, int):
            self.padding = padding
        elif isinstance(padding, tuple) and len(padding) == 3 and all(type(p)==int for p in padding):
            self.padding = padding
        elif padding == 'same':
            self.padding = kernel_size // 2
        else:
            raise ValueError(f'Parameter padding must be int, tuple, or "same. Given: {padding}')

        self.gspace = gspaces.octaOnR3()
        self.input_type = enn.FieldType(self.gspace, in_channels*[self.gspace.trivial_repr]) 

        small_type = enn.FieldType(self.gspace, 1*[self.gspace.regular_repr])
        mid_type   = enn.FieldType(self.gspace, 2*[self.gspace.regular_repr])
        
        params = [
            { 'in_type': small_type, 'out_type': small_type, 'kernel_size': kernel_size, 'padding': self.padding, 'bias': False, 'initialize': False },
            { 'in_type': small_type, 'out_type': small_type, 'kernel_size': kernel_size, 'padding': self.padding, 'bias': False, 'initialize': False },
            { 'in_type': small_type, 'out_type': small_type, 'kernel_size': kernel_size, 'padding': self.padding, 'bias': False, 'initialize': False },
            # { 'in_type': small_type, 'out_type': mid_type,   'kernel_size': kernel_size, 'padding': self.padding, 'bias': False, 'initialize': False },
            # { 'in_type': mid_type,   'out_type': small_type, 'kernel_size': kernel_size, 'padding': self.padding, 'bias': False, 'initialize': False },
        ]
        
        blocks = [ enn.R3Conv(self.input_type,
                              small_type,
                              kernel_size=kernel_size,
                              padding=self.padding,
                              initialize=False) ]

        for param in params:
            blocks.append(ConvBlock3D(**param))
        
        self.model = enn.SequentialModule(*blocks)
        pool_out = len(params[-1]['out_type'].representations)
        self.pool = FTGPool(max_freq, pool_out, 'ico')
        self.final = nn.Conv3d(pool_out, out_channels, kernel_size=kernel_size, padding=self.padding)

        self.crop = 2* (kernel_size // 2 - self.padding) + sum(b.crop for b in blocks[1:])
        
    def init(self):
        for m in self.modules():
            if isinstance(m, enn.R3Conv):
                m.weights.data = torch.randn_like(m.weights)
        
    def forward(self, x):
        x = enn.GeometricTensor(x, self.input_type)
        x = self.model(x)
        x = self.pool(x)
        x = x.tensor
        x = self.final(x)
        return x
    
    def evaluate_output_shape(self, input_shape):
        return input_shape[..., self.crop:-self.crop, self.crop:-self.crop, self.crop: -self.crop]


class BaselineSteerableCNN(Base):
    def __init__(self, in_channels=1, out_channels=1, max_freq=2, kernel_size=3, padding=0, **kwargs):
        super().__init__(**kwargs)

        if isinstance(padding, int):
            self.padding = padding
        elif isinstance(padding, tuple) and len(padding) == 3 and all(type(p)==int for p in padding):
            self.padding = padding
        elif padding == 'same':
            self.padding = kernel_size // 2
        else:
            raise ValueError(f'Parameter padding must be int, tuple, or "same. Given: {padding}')

        self.gspace = gspaces.rot3dOnR3()
        self.input_type = enn.FieldType(self.gspace, in_channels*[self.gspace.trivial_repr]) 

        small_type = enn.FieldType(self.gspace, 4*[self.gspace.irrep(l) for l in range(max_freq+1)])
        mid_type   = enn.FieldType(self.gspace, 16*[self.gspace.irrep(l) for l in range(max_freq+1)])
        
        params = [
            { 'in_type': small_type, 'out_type': small_type, 'kernel_size': kernel_size, 'padding': self.padding, 'bias': False, 'initialize': False },
            { 'in_type': small_type, 'out_type': small_type, 'kernel_size': kernel_size, 'padding': self.padding, 'bias': False, 'initialize': False },
            { 'in_type': small_type, 'out_type': small_type, 'kernel_size': kernel_size, 'padding': self.padding, 'bias': False, 'initialize': False },
            { 'in_type': small_type, 'out_type': mid_type,   'kernel_size': kernel_size, 'padding': self.padding, 'bias': False, 'initialize': False },
            { 'in_type': mid_type,   'out_type': small_type, 'kernel_size': kernel_size, 'padding': self.padding, 'bias': False, 'initialize': False },
        ]
        
        blocks = [ enn.R3Conv(self.input_type,
                              small_type,
                              kernel_size=kernel_size,
                              padding=self.padding,
                              initialize=False) ]

        for param in params:
            blocks.append(ConvBlock3D(**param))
        
        self.model = enn.SequentialModule(*blocks)
        pool_out = len(params[-1]['out_type'].representations)
        self.pool = FTGPool(max_freq, pool_out, 'ico')
        self.final = nn.Conv3d(pool_out, out_channels, kernel_size=kernel_size, padding=self.padding)

        self.crop = 2* (kernel_size // 2 - self.padding) + sum(b.crop for b in blocks[1:])
        
    def init(self):
        for m in self.modules():
            if isinstance(m, enn.R3Conv):
                m.weights.data = torch.randn_like(m.weights)
        
    def forward(self, x):
        x = enn.GeometricTensor(x, self.input_type)
        x = self.model(x)
        x = self.pool(x)
        x = x.tensor
        x = self.final(x)
        return x
    
    def evaluate_output_shape(self, input_shape):
        return input_shape[..., self.crop:-self.crop, self.crop:-self.crop, self.crop: -self.crop]

