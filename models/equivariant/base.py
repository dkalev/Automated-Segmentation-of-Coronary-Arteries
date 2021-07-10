import e3cnn.nn as enn
import e3cnn.gspaces as gspaces
import torch
import numpy as np
from e3cnn.group import directsum
from typing import Tuple
from ..base import Base


class GatedFieldType(enn.FieldType):
    def __init__(self, gspace: gspaces.GSpace3D,
                       trivials: enn.FieldType,
                       gated: enn.FieldType,
                       gates: enn.FieldType):
        """ Extends the standard enn.FieldType to provide additional features
            useful for working with gated nonlinearities

        Args:
            gspace (gspaces.GSpace3D): [description]
            trivials (enn.FieldType): field type containing irreps of freq 0
            gated (enn.FieldType): field type containing higher order freq
            gates (enn.FieldType): field type containing scalar gates for the higher order freq nonlinearities
        """
        self.trivials = trivials
        self.gated = gated
        self.gates = gates

        combined = self.trivials
        if self.gated and self.gates:
            combined += self.gated
            combined += self.gates
        super().__init__(gspace, (combined).representations)

    def no_gates(self):
        """ Returns a new field type with irreps only without the gates

        Returns:
            enn.FieldType: the direct sum of trivials and higher freq irreps
        """
        combined = self.trivials
        if self.gated: combined += self.gated
        return enn.FieldType(self.gspace, combined.representations)

    @classmethod
    def build(cls,
            gspace: gspaces.GSpace3D,
            channels: int,
            type: str = 'spherical',
            max_freq: int = 2,
            direct_sum: bool = True,
            channels_hard_limit: bool = False) -> 'GatedFieldType':
        """ Creates a gated field type with the specified number of channels

        Args:
            gspace (gspaces.GSpace3D): group space w.r.t which the features will be equivariant
            channels (int): total number of channels (check also channels_hard_limit)
            type (str, optional): Feature type, can be one of [trivial, spherical, so3]. Defaults to 'spherical'.
            max_freq (int, optional): Maximum frequency for the irreps. Defaults to 2.
            direct_sum (bool, optional): If set to true irreps of frequency greater than 0
                will be combined in a directsum and a single gate will be used for them. Otherwise each irrep
                gets a separate gate. Defaults to True.
            channels_hard_limit (bool, optional): If set to true all the features including the gates
                must not exceed the value specified in channels, otherwise as many features as possible are
                fit within the specified channels and the gates are added as extra channels. Defaults to False.

        Raises:
            ValueError: if the type is not one of the following: [trivial, spherical, so3]
            ValueError: if the dimensionality of the requested type exceeds the number of 
                requested channels

        Returns:
            GatedFieldType: instance of the class that is a wrapper over enn.FieldType of the direct sum
                of trivials, gated and gates with the additional components as attributes and some additional
                functionality
        """

        if type == 'trivial':
            trivials = enn.FieldType(gspace, channels*[gspace.trivial_repr])
            gated, gates = None, None
            return cls(gspace, trivials, gated, gates)

        if type == 'spherical':
            dim = sum([2*l+1 for l in range(max_freq+1)])
        elif type == 'so3':
            dim = sum([(2*l+1)**2 for l in range(max_freq+1)])
        else:
            raise ValueError(f'type must be on of ["trivial", "spherical", "so3"], given: {type}')
        
        if channels_hard_limit:
            dim += 1 if direct_sum else max_freq # 1 gate if using directsum, otherwise gate per freq greater than 0
        
        if dim > channels:
            raise ValueError(f'Cannot build field of type {type}. The field requires at least {dim} channels, {channels} channels specified')

        n_irreps, n_rem = channels // dim, channels % dim
        n_triv = n_irreps + n_rem

        higher_freqs = [gspace.irrep(i) for i in range(1,max_freq+1)]
        higher_freqs = [directsum(higher_freqs)] if direct_sum else higher_freqs

        trivials = enn.FieldType(gspace, n_triv*[gspace.trivial_repr])
        gated = enn.FieldType(gspace, n_irreps*higher_freqs)
        gates = enn.FieldType(gspace, n_irreps*len(higher_freqs)*[gspace.trivial_repr])

        return cls(gspace, trivials, gated, gates)

    def __add__(self, other: 'GatedFieldType') -> 'GatedFieldType':
        assert self.gspace == other.gspace
        if self.gated and other.gated:
            gated = self.gated + other.gated
        elif self.gated:
            gated = self.gated
        elif other.gated:
            gated = other.gated
        else:
            gated = None
        if self.gates and other.gates:
            gates = self.gates + other.gates
        elif self.gates:
            gates = self.gates
        elif other.gates:
            gates = other.gates
        else:
            gates = None
        return GatedFieldType(self.gspace, self.trivials + other.trivials, gated, gates)


def kernel_so3(L: int):
    dims = [ 2 * l + 1 for l in range(L+1)]
    V = np.concatenate([np.eye(d).flatten() * d for d in dims])
    V /= np.linalg.norm(V)
    return V

def kernel_sphere(gspace: gspaces.GSpace3D, L: int):
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
    def __init__(self, gspace, in_channels=1, kernel_size=3, padding=0, **kwargs):
        super().__init__(**kwargs)

        self.gspace = gspace
        self.padding = self.parse_padding(padding, kernel_size)
        self.input_type = enn.FieldType(self.gspace, in_channels*[self.gspace.trivial_repr])

    def init(self):
        # FIXME initialize the rest of the modules when starting to use them
        for m in self.modules():
            if isinstance(m, enn.R3Conv):
                enn.init.generalized_he_init(m.weights.data, m.basisexpansion, cache=True)


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

