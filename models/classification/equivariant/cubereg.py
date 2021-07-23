import torch.nn as nn
import e3cnn.nn as enn
import e3cnn.gspaces as gspaces
from .base import BaseEquiv


class CubeCNN(BaseEquiv):
    def __init__(self, *args, kernel_size=5, initialize=True, **kwargs):
        super().__init__(*args, **kwargs)

        type240 = enn.FieldType(self.gspace, 10*[self.gspace.regular_repr])
        type600 = enn.FieldType(self.gspace, 25*[self.gspace.regular_repr])
        self.encoder = enn.SequentialModule(
            enn.R3Conv(self.input_type, type240, kernel_size=kernel_size, stride=2, bias=False, initialize=initialize),
            enn.IIDBatchNorm3d(type240),
            enn.ELU(type240, inplace=True),

            enn.R3Conv(type240, type600, kernel_size=kernel_size, stride=2, bias=False, initialize=initialize),
            enn.IIDBatchNorm3d(type600),
            enn.ELU(type600, inplace=True),

            enn.R3Conv(type600, type600, kernel_size=kernel_size, stride=2, bias=False, initialize=initialize),
            enn.IIDBatchNorm3d(type600),
            enn.ELU(type600, inplace=True),
        )
        self.pool = enn.GroupPooling(type600)
        pool_out = len(type600.representations)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(pool_out*5**3),
            nn.Linear(pool_out*5**3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )

    @property
    def gspace(self): return gspaces.octaOnR3()
    
    def forward(self, x):
        x = self.pre_forward(x)
        x = self.encoder(x)
        x = self.pool(x)
        x = x.tensor
        x = self.head(x)
        return x