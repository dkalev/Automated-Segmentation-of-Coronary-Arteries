import e3cnn.nn as enn
import e3cnn.gspaces as gspaces
from e3cnn.group.representation import Representation
import torch
import numpy as np
from typing import Tuple, List, Callable


def kernel_so3(L: int) -> np.ndarray:
    dims = [ 2 * l + 1 for l in range(L+1)]
    V = np.concatenate([np.eye(d).flatten() * d for d in dims])
    V /= np.linalg.norm(V)
    return V

def kernel_sphere(gspace: gspaces.GSpace3D, L: int) -> np.ndarray:
    sphere = gspace.fibergroup.homspace((False, -1))
    identity = gspace.fibergroup.identity

    return np.concatenate([
        sphere.basis(identity, (l,), (0,)).flatten() * np.sqrt(2*l+1)
        for l in range(L+1)
    ])


class FTNonLinearity(enn.EquivariantModule):
    def __init__(self,
                 max_freq_in: int,
                 channels: int,
                 *grid_args,
                 non_linearity: str = 'elu',
                 max_freq_out: int = None,
                 moorepenrose: bool = True,
                 repr_type: str = 'spherical',
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
        if repr_type not in ['spherical', 'so3']:
            raise ValueError(f'repr_type must be one of [spherical, so3], given: {repr_type}')
        self.repr_type = repr_type
        self.non_linearity = self.get_nonlin(non_linearity)
        max_freq_out = max_freq_out or max_freq_in

        self.gspace = gspaces.rot3dOnR3()
        rho    = self.get_representation(max_freq_in)
        rho_bl = self.get_representation(max_freq_out)

        self.dim = rho.size
        self.in_type  = enn.FieldType(self.gspace, [rho]*channels)
        self.out_type = enn.FieldType(self.gspace, [rho_bl]*channels)

        grid  = self.get_grid(*grid_args, **grid_kwargs)
        A     = self.build_sensing_matrix(rho, grid, max_freq_in)
        A_inv = self.build_reconstruction_matrix(rho_bl, grid, max_freq_out, moorepenrose=moorepenrose)

        self.register_buffer('A', torch.tensor(A, dtype=torch.get_default_dtype()))
        self.register_buffer('Ainv', torch.tensor(A_inv, dtype=torch.get_default_dtype()))
    
    @staticmethod
    def get_inv(x:np.ndarray, moorepenrose:bool=False, eps:float=1e-8) -> np.ndarray:
        if moorepenrose:
            return np.linalg.inv(x.T @ x + eps * np.eye(x.shape[1])) @ x.T
        else:
            return x.T
    
    @staticmethod
    def get_nonlin(nonlin_type:str) -> Callable:
        if hasattr(torch.nn.functional, nonlin_type):
            return getattr(torch.nn.functional, nonlin_type)
        else:
            raise ValueError(f'Unsupported non-linearity type: {nonlin_type}')

    def get_representation(self, max_freq:int) -> Representation:
        if self.repr_type == 'spherical':
            return self.gspace.fibergroup.bl_quotient_representation(max_freq, (False, -1))
        else:
            return self.gspace.fibergroup.bl_regular_representation(max_freq)

    def get_grid(self, *grid_args, **grid_kwargs) -> List:
        if self.repr_type == 'spherical':
            return self.gspace.fibergroup.sphere_grid(*grid_args, **grid_kwargs)
        else:
            return self.gspace.fibergroup.grid(*grid_args, **grid_kwargs)
        
    def get_kernel(self, max_freq:int) -> np.ndarray:
        if self.repr_type == 'spherical':
            return kernel_sphere(self.gspace, max_freq)
        else:
            return kernel_so3(max_freq)

    def build_sensing_matrix(self, rho:Representation, grid:List, max_freq:int) -> np.ndarray:
        kernel = self.get_kernel(max_freq)
        A = np.stack([ kernel @ rho(g).T for g in grid ])
        A /= np.sqrt(len(A))
        return A

    def build_reconstruction_matrix(self, rho:Representation, grid:List, max_freq:int, moorepenrose:bool=False) -> np.ndarray:
        kernel_bl = self.get_kernel(max_freq)
        Abl = np.stack([ kernel_bl @ rho(g).T for g in grid ])
        Abl /= np.sqrt(len(Abl))

        A_inv = self.get_inv(Abl, moorepenrose=moorepenrose)
        return A_inv

    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        assert x.type == self.in_type

        in_shape  = (x.shape[0], self.channels, self.dim, *x.shape[2:])
        out_shape = (x.shape[0], len(self.Ainv) * self.channels, *x.shape[2:])

        # IFT
        x = torch.einsum( 'gi,bfi...->bfg...', self.A, x.tensor.view(in_shape))
        y = self.non_linearity(x, inplace=True)
        # FT
        y = torch.einsum('ig,bfg...->bfi...', self.Ainv, y)
        y = y.reshape(out_shape)
        return enn.GeometricTensor(y, self.out_type)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        shape = [*input_shape]
        shape[1] = shape[1] // self.A.shape[1] * self.Ainv.shape[0]
        return tuple(shape)
