import e3cnn.nn as enn
import e3cnn.gspaces as gspaces
import torch.nn as nn
from .base import BaseEquiv, GatedFieldType


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
    def __init__(self, *args, kernel_size=5, initialize=True, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def gspace(self): return gspaces.rot3dOnR3()
    
    def forward(self, x):
        x = self.pre_forward(x)
        x = self.encoder(x)
        x = self.pool(x)
        x = x.tensor
        x = self.head(x)
        return x
