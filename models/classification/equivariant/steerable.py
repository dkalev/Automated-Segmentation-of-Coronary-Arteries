import e3cnn.nn as enn
import e3cnn.gspaces as gspaces
import torch.nn as nn
from .base import BaseEquiv, GatedFieldType
from .ft_nonlinearity import FTNonLinearity


class GatedCNN(BaseEquiv):
    def __init__(self, *args, kernel_size=5, repr_type='spherical', direct_sum=True, initialize=True, **kwargs):
        super().__init__(*args, **kwargs)

        type240  = GatedFieldType.build(self.gspace, 240, type=repr_type, max_freq=1, direct_sum=direct_sum)
        type600  = GatedFieldType.build(self.gspace, 600, type=repr_type, max_freq=3, direct_sum=direct_sum)
        type_final = type600.no_gates()

        self.encoder = enn.SequentialModule(
            enn.R3Conv(self.input_type, type240, kernel_size=kernel_size, stride=2, bias=False, initialize=initialize),
            enn.IIDBatchNorm3d(type240),
            self.get_nonlin(type240),

            enn.R3Conv(type240.no_gates(), type600, kernel_size=kernel_size, stride=2, bias=False, initialize=initialize),
            enn.IIDBatchNorm3d(type600),
            self.get_nonlin(type600),

            enn.R3Conv(type600.no_gates(), type600, kernel_size=kernel_size, stride=2, bias=False, initialize=initialize),
            enn.IIDBatchNorm3d(type600),
            self.get_nonlin(type600),
        )
        self.pool = enn.NormPool(type_final)
        pool_out = len(type_final.representations)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(pool_out*5**3),
            nn.Linear(pool_out*5**3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )

    @property
    def gspace(self): return gspaces.rot3dOnR3()

    @staticmethod
    def get_nonlin(ftype):
        if ftype.gated and ftype.gates:
            labels = len(ftype.trivials) * ['trivial'] + (len(ftype.gated) + len(ftype.gates)) * ['gate'] 
            labels_gate = len(ftype.gated)*['gated']+len(ftype.gates)*['gate']
            return enn.MultipleModule(ftype,
                labels=labels,
                modules=[
                    (enn.ELU(ftype.trivials, inplace=True), 'trivial'),
                    (enn.GatedNonLinearity1(ftype.gated+ftype.gates, labels_gate), 'gate')
                ]
            )
        else:
            return enn.ELU(ftype, inplace=True)
    
    def forward(self, x):
        x = self.pre_forward(x)
        x = self.encoder(x)
        x = self.pool(x)
        x = x.tensor
        x = self.head(x)
        return x


class FTCNN(BaseEquiv):
    def __init__(self, *args,
                kernel_size=5,
                initialize=True,
                repr_type='spherical',
                grid_kwargs=None,
                **kwargs):
        super().__init__(*args, **kwargs)
        self.repr_type = repr_type
        self.grid_kwargs = grid_kwargs or {'type': 'cube'}

        common_kwargs = {
                'kernel_size': kernel_size,
                'stride': 2,
                'initialize': initialize,
        }

        params = [
            {'max_freq': 2, 'out_channels': 240, **common_kwargs},
            {'max_freq': 2, 'out_channels': 600, **common_kwargs},
            {'max_freq': 2, 'out_channels': 600, **common_kwargs},
        ]

        blocks = []
        in_type = self.input_type
        for param in params:
            block, in_type = self.get_block(in_type, **param)
            blocks.append(block)

        self.encoder = enn.SequentialModule(*blocks)
        self.pool = enn.NormPool(blocks[-1].out_type)
        pool_out = self.pool.out_type.size

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(pool_out*5**3),
            nn.Linear(pool_out*5**3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )

    @property
    def gspace(self): return gspaces.rot3dOnR3()

    @staticmethod
    def get_dim(max_freq, spherical=False):
        if spherical:
            return sum([2*l+1 for l in range(max_freq+1)])
        else:
            return sum([(2*l+1)**2 for l in range(max_freq+1)])

    def get_block(self, in_type, out_channels, max_freq=2, **kwargs):
        if self.repr_type == 'trivial':
            return self._get_block_trivial(in_type, out_channels, **kwargs)
        elif self.repr_type in ['spherical', 'so3']:
            return self._get_block_non_trivial(in_type, out_channels, max_freq=max_freq, **kwargs)

    def _get_block_trivial(self, in_type, channels, **kwargs):
        out_type = enn.FieldType(self.gspace, channels*[self.gspace.trivial_repr])
        return enn.SequentialModule(
            enn.R3Conv(in_type, out_type, **kwargs),
            enn.IIDBatchNorm3d(out_type),
            enn.ELU(out_type, inplace=True)
        ), out_type

    def _get_block_non_trivial(self, in_type, out_channels, max_freq=2, **kwargs):
        dim = self.get_dim(max_freq, spherical=self.repr_type=='spherical')
        channels = max(1, out_channels // dim)
        ft_nonlin = FTNonLinearity( max_freq,
                                    channels,
                                    type=self.grid_kwargs['type'],
                                    repr_type=self.repr_type,
                                    N=self.grid_kwargs.get('N'),
                                    )
        mid_type, out_type = ft_nonlin.in_type, ft_nonlin.out_type
        return enn.SequentialModule(
            enn.R3Conv(in_type, mid_type, **kwargs),
            enn.IIDBatchNorm3d(mid_type),
            ft_nonlin
        ), out_type
    
    def forward(self, x):
        x = self.pre_forward(x)
        x = self.encoder(x)
        x = self.pool(x)
        x = x.tensor
        x = self.head(x)
        return x
