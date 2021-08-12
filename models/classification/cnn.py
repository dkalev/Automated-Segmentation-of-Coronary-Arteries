import torch.nn as nn
from .base import BaseClassification


class Baseline3DClassification(BaseClassification):
    def __init__(self, *args, kernel_size=5, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 240, kernel_size=kernel_size, stride=2, bias=False),
            nn.BatchNorm3d(240),
            nn.ReLU(inplace=True),

            nn.Conv3d(240, 600, kernel_size=kernel_size, stride=2, bias=False),
            nn.BatchNorm3d(600),
            nn.ReLU(inplace=True),

            nn.Conv3d(600, 600, kernel_size=kernel_size, stride=2, bias=False),
            nn.BatchNorm3d(600),
            nn.ReLU(inplace=True),

            nn.Conv3d(600, 300, kernel_size=1, bias=False),
            nn.BatchNorm3d(300),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=1),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(300*3**3),
            nn.Linear(300*3**3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x
