import torch.nn as nn
from .base import BaseSegmentation
import numpy as np


class Baseline3DSegmentation(BaseSegmentation):
    def __init__(self, *args, kernel_size=5, arch='default', **kwargs):
        super().__init__(*args, **kwargs)

        common_params = {
            'kernel_size': kernel_size,
            'bias': False,
        }

        if arch == 'default':
            block_params = [
                {'in_channels': 1, 'out_channels': 60 },
                {'in_channels': 60, 'out_channels': 120 },
                {'in_channels': 120, 'out_channels': 120 },
                {'in_channels': 120, 'out_channels': 60 },
            ]
        elif arch == 'strided':
            block_params = [
                {'in_channels': 1, 'out_channels': 120 , 'stride': 2 },
                {'in_channels': 120, 'out_channels': 360  },
                {'in_channels': 360, 'out_channels': 360 },
                {'in_channels': 360, 'out_channels': 360 },
                {'in_channels': 360, 'out_channels': 360 },
                # {'in_channels': 360, 'out_channels': 360 },
            ]
        elif arch == 'patch64':
            block_params = [
                {'in_channels': 1, 'out_channels': 120 },
                {'in_channels': 120, 'out_channels': 120 },
                {'in_channels': 120, 'out_channels': 360 },
                {'in_channels': 360, 'out_channels': 120 },
            ]
        elif arch == 'fully_conv':
            block_params = [
                {'in_channels': 1, 'out_channels': 120, 'kernel_size': 11 },
                {'in_channels': 120, 'out_channels': 120, 'kernel_size': 7 },
                {'in_channels': 120, 'out_channels': 120, 'kernel_size': 5 },
                {'in_channels': 120, 'out_channels': 120, 'kernel_size': 5 },
                {'in_channels': 120, 'out_channels': 120, 'kernel_size': 5 },
            ]

        for b_params in block_params:
            for k, v in common_params.items():
                if k not in b_params:
                    b_params[k] = v

        blocks = [
            nn.Sequential(
                nn.Conv3d(**b_params),
                nn.InstanceNorm3d(b_params['out_channels'], affine=True),
                nn.ReLU(inplace=True),
            ) for b_params in block_params
        ]

        if arch == 'strided': blocks.append(nn.Upsample(scale_factor=2))
        blocks.append(
            nn.Conv3d(block_params[-1]['out_channels'], 1, kernel_size=1)
        )

        if arch == 'strided':
            self.crop = np.array([18,18,18])
        elif arch == 'fully_conv':
            self.crop = np.sum([ bp['kernel_size']//2 for bp in block_params ])
            self.crop = np.array([self.crop, self.crop, self.crop])
        else:
            self.crop = (common_params['kernel_size']//2) * len(blocks[:-1]) # last layer doesn't affect crop
            self.crop = np.array([self.crop, self.crop, self.crop])

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
