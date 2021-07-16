from .base import EquivUNet
import e3cnn.nn as enn
import e3cnn.gspaces as gspaces


class IcoUNet(EquivUNet):
    def __init__(self, *args, arch='mixed', **kwargs):
        super().__init__(*args, **kwargs)
        layer_params = self.get_layer_params(self.gspace, arch)
        self.build_model(layer_params)

    @property
    def gspace(self): return gspaces.icoOnR3()

    @staticmethod
    def get_pooling(ftype): return enn.GroupPooling(ftype)

    def get_layer_params(self, gspace, arch):
        vert_repr = gspace.fibergroup.ico_vertices_representation
        face_repr = gspace.fibergroup.ico_faces_representation
        if arch == 'mixed':
            type32  = enn.FieldType(gspace, 3*[vert_repr])
            type64  = enn.FieldType(gspace, 1*[gspace.regular_repr]+4*[gspace.trivial_repr])
            type128 = enn.FieldType(gspace, 2*[gspace.regular_repr]+8*[gspace.trivial_repr])
            type256 = enn.FieldType(gspace, 4*[gspace.regular_repr]+[vert_repr]+4*[gspace.trivial_repr])
            type320 = enn.FieldType(gspace, 5*[gspace.regular_repr]+1*[face_repr])
        elif arch == 'vert':
            type32  = enn.FieldType(gspace, 3*[vert_repr])
            type64  = enn.FieldType(gspace, 5*[vert_repr]+4*[gspace.trivial_repr])
            type128 = enn.FieldType(gspace, 10*[vert_repr]+8*[gspace.trivial_repr])
            type256 = enn.FieldType(gspace, 21*[vert_repr]+4*[gspace.trivial_repr])
            type320 = enn.FieldType(gspace, 26*[vert_repr]+8*[gspace.trivial_repr])
        elif arch == 'regular':
            type32  = enn.FieldType(gspace, 3*[vert_repr])
            type64  = enn.FieldType(gspace, 1*[gspace.regular_repr]+4*[gspace.trivial_repr])
            type128 = enn.FieldType(gspace, 2*[gspace.regular_repr]+8*[gspace.trivial_repr])
            type256 = enn.FieldType(gspace, 4*[gspace.regular_repr]+16*[gspace.trivial_repr])
            type320 = enn.FieldType(gspace, 5*[gspace.regular_repr]+20*[gspace.trivial_repr])

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
