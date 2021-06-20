import e3cnn.nn as enn
import e3cnn.gspaces as gspaces
from e3cnn.gspaces import GSpace3D
import torch
import numpy as np
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


def kernel_so3(L: int):
    dims = [ 2 * l + 1 for l in range(L+1)]
    V = np.concatenate([np.eye(d).flatten() * d for d in dims])
    V /= np.linalg.norm(V)
    return V

def kernel_sphere(gspace: GSpace3D, L: int):
    sphere = gspace.fibergroup.homspace((False, -1))
    identity = gspace.fibergroup.identity

    return np.concatenate([
        sphere.basis(identity, (l,), (0,)).flatten() * np.sqrt(2*l+1)
        for l in range(L+1)
    ])


class FTNonLinearity(enn.EquivariantModule):
    def __init__(self,
                 max_freq_in,
                 channels,
                 *grid_args,
                 non_linearity='elu',
                 max_freq_out: int = None,
                 moorepenrose: bool = True,
                 spherical: bool = False,
                 **grid_kwargs):
        f"""
            Sample SO(3)'s irreps -> Apply ELU -> Fourier Transform

            features = (max frequency, number of channels)
            max_freq_out = bandlimit for output (the final fourier transform); if None (by default), use the max_frequency in `features`
            *grid_args, **grid_kwargs = argument to generate the grids over SO(3); see `SO3.grid()`
            moorepenrose = keep it True

        """
        super().__init__()
        self.channels = channels
        self.spherical = spherical
        if hasattr(torch.nn.functional, non_linearity):
            self.non_linearity = getattr(torch.nn.functional, non_linearity)
        else:
            raise ValueError(f'Unsupported non-linearity type: {non_linearity}')

        max_freq_out = max_freq_out or max_freq_in

        self.gspace = gspaces.rot3dOnR3()
        rho = self.get_representation(max_freq_in)
        rho_bl = self.get_representation(max_freq_out)

        self.dim = rho.size
        self.in_type  = enn.FieldType(self.gspace, [rho]*channels)
        self.out_type = enn.FieldType(self.gspace, [rho_bl]*channels)

        grid = self.get_grid(*grid_args, **grid_kwargs)

        # sensing matrix
        kernel = self.get_kernel(max_freq_in)
        A = np.stack([ kernel @ rho(g).T for g in grid ])
        A /= np.sqrt(len(A))

        # reconstruction matrix
        kernel_bl = self.get_kernel(max_freq_out)
        Abl = np.stack([ kernel_bl @ rho_bl(g).T for g in grid ])
        Abl /= np.sqrt(len(Abl))

        eps = 1e-8
        if moorepenrose:
            A_inv = np.linalg.inv(Abl.T @ Abl + eps * np.eye(Abl.shape[1])) @ Abl.T
        else:
            A_inv = Abl.T

        self.register_buffer('A', torch.tensor(A, dtype=torch.get_default_dtype()))
        self.register_buffer('Ainv', torch.tensor(A_inv, dtype=torch.get_default_dtype()))

    def get_representation(self, max_freq):
        if self.spherical:
            return self.gspace.fibergroup.bl_quotient_representation(max_freq, (False, -1))
        else:
            return self.gspace.fibergroup.bl_regular_representation(max_freq)

    def get_grid(self, *grid_args, **grid_kwargs):
        if self.spherical:
            return self.gspace.fibergroup.sphere_grid(*grid_args, **grid_kwargs)
        else:
            return self.gspace.fibergroup.grid(*grid_args, **grid_kwargs)
        
    def get_kernel(self, max_freq):
        if self.spherical:
            return kernel_sphere(self.gspace, max_freq)
        else:
            return kernel_so3(max_freq)

    def forward(self, x: enn.GeometricTensor):
        assert x.type == self.in_type

        in_shape = (x.shape[0], self.channels, self.dim, *x.shape[2:])
        out_shape = (x.shape[0], len(self.Ainv) * self.channels, *x.shape[2:])

        # IFT
        x = x.tensor.view(in_shape)
        _x = torch.einsum( 'gi,bfi...->bfg...', self.A, x)

        _y = self.non_linearity(_x, inplace=True)

        # FT
        y = torch.einsum( 'ig,bfg...->bfi...', self.Ainv, _y)
        y = y.reshape(out_shape)
        return enn.GeometricTensor(y, self.out_type)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        shape = [*input_shape]
        shape[1] = shape[1] // self.A.shape[1] * self.Ainv.shape[0]
        return tuple(shape)


class BaseEquiv(Base):
    def __init__(self, gspace, in_channels=1, kernel_size=3, padding=0, initialize=False, **kwargs):
        super().__init__(**kwargs)

        self.gspace = gspace
        self.padding = self.parse_padding(padding, kernel_size)
        self.input_type = enn.FieldType(self.gspace, in_channels*[self.gspace.trivial_repr])

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

