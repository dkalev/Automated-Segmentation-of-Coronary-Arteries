from .base import EquivUNet
import e3cnn.nn as enn
import e3cnn.gspaces as gspaces
from ..base import FTNonLinearity


class FTUNet(EquivUNet):
    def __init__(self,
                *args,
                repr_type: str = 'spherical',
                max_freq: int = 2,
                grid_kwargs: dict = None,
                **kwargs):
        super().__init__(*args, **kwargs)
        self.repr_type = repr_type
        self.max_freq = max_freq
        self.grid_kwargs = grid_kwargs or {
            'type': 'cube',
        }
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

        if repr_type == 'trivial':
            type32  = enn.FieldType(gspace, 32 *[gspace.trivial_repr])
            type64  = enn.FieldType(gspace, 64 *[gspace.trivial_repr])
            type128 = enn.FieldType(gspace, 128*[gspace.trivial_repr])
            type256 = enn.FieldType(gspace, 256*[gspace.trivial_repr])
            type320 = enn.FieldType(gspace, 320*[gspace.trivial_repr])
        elif repr_type == 'spherical':
            rho_spherical = gspace.fibergroup.bl_quotient_representation(2, (False, -1))
            type32  = enn.FieldType(gspace, 4 *[rho_spherical])
            type64  = enn.FieldType(gspace, 7 *[rho_spherical])
            type128 = enn.FieldType(gspace, 14*[rho_spherical])
            type256 = enn.FieldType(gspace, 28*[rho_spherical])
            type320 = enn.FieldType(gspace, 36*[rho_spherical])
        elif repr_type == 'so3':
            rho_so3 = gspace.fibergroup.bl_regular_representation(2)
            type32  = enn.FieldType(gspace, 1*[rho_so3])
            type64  = enn.FieldType(gspace, 2*[rho_so3])
            type128 = enn.FieldType(gspace, 4*[rho_so3])
            type256 = enn.FieldType(gspace, 7*[rho_so3])
            type320 = enn.FieldType(gspace, 9*[rho_so3])

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

    def get_nonlin(self, ftype):
        if self.repr_type == 'trivial':
            return enn.ELU(ftype, inplace=True)
        elif self.repr_type in ['spherical', 'so3']:
            return FTNonLinearity(
                max_freq_in=self.max_freq,
                channels=len(ftype),
                repr_type=self.repr_type,
                **self.grid_kwargs
            )
