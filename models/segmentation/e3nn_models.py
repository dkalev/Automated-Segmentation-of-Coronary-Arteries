import torch
import torch.nn as nn

from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, Linear
from e3nn.nn.batchnorm import BatchNorm
from e3nn.nn.gate import Gate
from e3nn.math import soft_one_hot_linspace
from .base import Base

class Convolution(torch.nn.Module):
    r"""convolution on voxels
    Parameters
    ----------
    irreps_in : `Irreps`
    irreps_out : `Irreps`
    irreps_sh : `Irreps`
        set typically to ``o3.Irreps.spherical_harmonics(lmax)``
    size : int
    steps : tuple of int
    """
    def __init__(self, irreps_in, irreps_out, irreps_sh, size, steps=(1, 1, 1), **kwargs):
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.size = size
        self.num_rbfs = self.size
        self.crop = size // 2
        
        # self-connection
        self.sc = Linear(self.irreps_in, self.irreps_out)
        # R^3 = angles S^2 x radial R^+ union with origin
        # 3x3 8 values ^, 1 coming from Linear

        # connection with neighbors
        r = torch.linspace(-1, 1, self.size)
        x = r * steps[0] / min(steps)
        x = x[x.abs() <= 1]
        y = r * steps[1] / min(steps)
        y = y[y.abs() <= 1]
        z = r * steps[2] / min(steps)
        z = z[z.abs() <= 1]
        lattice = torch.stack(torch.meshgrid(x, y, z), dim=-1)  # [x, y, z, R^3]
        self.register_buffer('d', lattice.norm(dim=-1))

        emb = soft_one_hot_linspace(
            x=self.d,
            start=0.0,
            end=1.0,
            number=self.num_rbfs,
            basis='smooth_finite',
            endpoint=True,
        ) # [d.shape[0] x self.num_rbfs]
        self.register_buffer('emb', emb)

        sh = o3.spherical_harmonics(self.irreps_sh, lattice, True, 'component')  # [x, y, z, irreps_sh.dim]
        self.register_buffer('sh', sh)

        self.tp = FullyConnectedTensorProduct(self.irreps_in, self.irreps_sh, self.irreps_out, shared_weights=False)

        self.weight = torch.nn.Parameter(
            nn.init.kaiming_normal_(
                torch.randn(self.num_rbfs, self.tp.weight_numel), nonlinearity='relu')
            )

    def forward(self, x):
        r"""
        Parameters
        ----------
        x : `torch.Tensor`
            tensor of shape ``(batch, irreps_in.dim, x, y, z)``
        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(batch, irreps_out.dim, x, y, z)``
        """
        sc = self.sc(x.transpose(1, 4)).transpose(1, 4)
        sc = sc[..., self.crop:-self.crop, self.crop: -self.crop, self.crop:-self.crop] # remove parts outside of receptive field

        weight = self.emb @ self.weight # [d x n_rbfs] @ [n_rbfs x n_tp_weights] => [d x n_tp_weights]
        
        # weight = weight / (self.size ** (3/2)) # normalizing weights to ensure the norm of the weight is 1
        kernel = self.tp.right(self.sh, weight)  # [x, y, z, irreps_in.dim, irreps_out.dim] xw \tp y [w()\tp y](x)
        # input_irreps \tp (weights) base_sp_irreps => output_irreps
        # dim tp = [input_irreps, base_sp_irreps, output_irreps] = tp.right => [input_irreps, output_irreps]
        kernel = torch.einsum('xyzio->oixyz', kernel)

        # return sc + 0.1 * torch.nn.functional.conv3d(x, kernel)
        return sc + torch.nn.functional.conv3d(x, kernel)

class BatchNorm3D(BatchNorm):
    def forward(self, x):
        x = x.transpose(1, 4)
        x = super().forward(x)
        x = x.transpose(1, 4)
        return x

class GatedNonLinearity(Gate):
    def forward(self, x):
        x = x.transpose(1, 4)
        x = super().forward(x)
        x = x.transpose(1, 4)
        return x
    
class e3nnCNN(Base):
    def __init__(self, *args, kernel_size=3, **kwargs):
        super().__init__(*args, **kwargs)
    
        blocks = [
            self.get_conv_block(in_channels=1, out_channels=1, max_freq_in=0, max_freq_out=1, kernel_size=kernel_size),
            self.get_conv_block(in_channels=1, out_channels=2, max_freq_in=1, max_freq_out=2, kernel_size=kernel_size),
            self.get_conv_block(in_channels=2, out_channels=4, max_freq_in=2, max_freq_out=2, kernel_size=kernel_size),
            self.get_conv_block(in_channels=4, out_channels=4, max_freq_in=2, max_freq_out=2, kernel_size=kernel_size),
            Convolution('4x0e+4x1o+4x2e', '1x0e', o3.Irreps.spherical_harmonics(lmax=2), size=kernel_size)
        ]
        self.model = nn.Sequential(*blocks)
        self.crop = (kernel_size//2) * len(blocks)
    
    def forward(self, x):
        return self.model(x)


    def get_conv_block(self, in_channels=1, out_channels=1, max_freq_in=2, max_freq_out=2, kernel_size=5):
        in_type, _ = self.get_field_type(in_channels, max_freq_in, include_gates=False)
        out_type, (gates_type_o, triv_type_o, nontriv_type_o) = self.get_field_type(out_channels, max_freq_out)

        if nontriv_type_o == '':
            activation = nn.ReLU(inplace=True)
        else:
            activation = GatedNonLinearity(triv_type_o,
                            [torch.relu], gates_type_o, [torch.sigmoid], nontriv_type_o)

        return nn.Sequential(
            Convolution(in_type, out_type, o3.Irreps.spherical_harmonics(lmax=2), size=kernel_size),
            BatchNorm3D(out_type),
            activation,
        )

    def get_field_type(self, n_channels, max_freq, include_gates=True):
        irrep_types = [f'{n_channels}x{i}' for i in range(max_freq+1)]
        irrep_types = [irrep_type+'e' if i%2==0 else irrep_type+'o' for i, irrep_type in enumerate(irrep_types)]
        field_type_comps = self.get_field_comps(irrep_types, n_channels, max_freq)
        field_type = self.combine_type_comps(*field_type_comps, include_gates)
        return field_type, field_type_comps
    
    def get_field_comps(self, irrep_types, n_channels, max_freq):
        gates_type = self.get_gates_type(n_channels, max_freq)
        trivial_type = irrep_types[0]
        non_trivial_types = '+'.join(irrep_types[1:])
        return gates_type, trivial_type, non_trivial_types

    @staticmethod
    def combine_type_comps(gates_type, trivial_type, non_trivial_types, include_gates=True):
        if include_gates and gates_type != '':
            res = f'{gates_type}+{trivial_type}'
        else:
            res = trivial_type
        
        if non_trivial_types != '': res = f'{res}+{non_trivial_types}'

        return res

    @staticmethod
    def get_gates_type(n_channels, max_freq):
        n_gates = max_freq * n_channels
        if n_gates == 0: return ''
        return f'{n_gates}x0e'
