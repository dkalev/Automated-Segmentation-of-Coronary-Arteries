from .base import EquivUNet
import e3cnn.nn as enn
import e3cnn.gspaces as gspaces
from ..base import GatedFieldType


class GatedUNet(EquivUNet):
    def __init__(self, *args, repr_type='spherical', **kwargs):
        super().__init__(*args, **kwargs)
        layer_params = self.get_layer_params(self.gspace, repr_type=repr_type)
        self.build_model(layer_params)

    @property
    def gspace(self): return gspaces.rot3dOnR3()
    
    @staticmethod
    def get_pooling(ftype): return enn.NormPool(ftype)

    def get_layer_params(self, gspace, **kwargs):
        repr_type = kwargs.get('repr_type')
        if repr_type not in ['trivial', 'spherical', 'so3']:
            raise ValueError(f'repr_type must be one of [trivial, spherical, so3], given: {repr_type}')
        
        direct_sum = 'direct_sum' in kwargs and kwargs['direct_sum']

        type32  = GatedFieldType.build(gspace, 32, type=repr_type, max_freq=1, direct_sum=direct_sum)
        type64  = GatedFieldType.build(gspace, 64, type=repr_type, max_freq=3, direct_sum=direct_sum)
        type128 = GatedFieldType.build(gspace, 128, type=repr_type, max_freq=3, direct_sum=direct_sum)
        type256 = GatedFieldType.build(gspace, 256, type=repr_type, max_freq=3, direct_sum=direct_sum)
        type320 = GatedFieldType.build(gspace, 320, type=repr_type, max_freq=3, direct_sum=direct_sum)

        return {
            'encoders': [
                {'in_type': self.input_type, 'mid_type': type32, 'out_type': type32, 'stride': 1},
                {'in_type': type32, 'mid_type': type64, 'out_type': type64, 'stride': 2},
                {'in_type': type64, 'mid_type': type128, 'out_type': type128, 'stride': 2},
                {'in_type': type128, 'mid_type': type256, 'out_type': type256, 'stride': 2},
                {'in_type': type256, 'mid_type': type320, 'out_type': type320, 'stride': 2},
            ],
            'bottleneck': {'in_type': type320, 'out_type': type320},
            'decoders': [
                {'in_type': type320, 'mid_type': type320, 'out_type': type320},
                {'in_type': type320, 'mid_type': type256, 'out_type': type256},
                {'in_type': type256, 'mid_type': type128, 'out_type': type128},
                {'in_type': type128, 'mid_type': type64, 'out_type': type64},
                {'in_type': type64, 'mid_type': type32, 'out_type': type32},
            ],
        }

    @staticmethod
    def get_nonlin(ftype):
        pass

