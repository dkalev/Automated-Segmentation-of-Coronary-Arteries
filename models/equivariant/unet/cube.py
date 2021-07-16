from .base import EquivUNet
import e3cnn.nn as enn
import e3cnn.gspaces as gspaces


class CubeUNet(EquivUNet):
    def __init__(self, *args, arch='mixed', **kwargs):
        super().__init__(*args, **kwargs)
        layer_params = self.get_layer_params(self.gspace, arch)
        self.build_model(layer_params)

    @property
    def gspace(self): return gspaces.octaOnR3()

    @staticmethod
    def get_pooling(ftype): return enn.GroupPooling(ftype)

    def get_layer_params(self, gspace, arch):
        vert_repr = gspace.fibergroup.cube_vertices_representation
        if arch == 'mixed':
            type32  = enn.FieldType(gspace, 4*[vert_repr])
            type64  = enn.FieldType(gspace, 2*[vert_repr]+2*[gspace.regular_repr])
            type128 = enn.FieldType(gspace, 1*[vert_repr]+5*[gspace.regular_repr])
            type256 = enn.FieldType(gspace, 2*[vert_repr]+10*[gspace.regular_repr])
            type320 = enn.FieldType(gspace, 2*[vert_repr]+13*[gspace.regular_repr])
        elif arch == 'vert':
            type32  = enn.FieldType(gspace, 4*[vert_repr])
            type64  = enn.FieldType(gspace, 8*[vert_repr])
            type128 = enn.FieldType(gspace, 16*[vert_repr])
            type256 = enn.FieldType(gspace, 32*[vert_repr])
            type320 = enn.FieldType(gspace, 40*[vert_repr])
        elif arch == 'regular':
            type32  = enn.FieldType(gspace, 2*[gspace.regular_repr])
            type64  = enn.FieldType(gspace, 3*[gspace.regular_repr])
            type128 = enn.FieldType(gspace, 6*[gspace.regular_repr])
            type256 = enn.FieldType(gspace, 11*[gspace.regular_repr])
            type320 = enn.FieldType(gspace, 14*[gspace.regular_repr])
        
        return {
            'encoders': [
                {'in_type': self.input_type, 'out_type': type32, 'stride': 1},
                {'in_type': type32, 'out_type': type64, 'stride': 2},
                {'in_type': type64, 'out_type': type128, 'stride': 2},
                {'in_type': type128, 'out_type': type256, 'stride': 2},
                {'in_type': type256, 'out_type': type320, 'stride': 2},
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
        return enn.ELU(ftype, inplace=True)
