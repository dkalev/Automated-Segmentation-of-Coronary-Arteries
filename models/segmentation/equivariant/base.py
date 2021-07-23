import e3cnn.nn as enn
import e3cnn.gspaces as gspaces
import torch
from e3cnn.group import directsum
from abc import abstractmethod
from ..base import BaseSegmentation


class GatedFieldType(enn.FieldType):
    def __init__(self, gspace: gspaces.GSpace3D,
                       trivials: enn.FieldType,
                       gated: enn.FieldType,
                       gates: enn.FieldType):
        """ Extends the standard enn.FieldType to provide additional features
            useful for working with gated nonlinearities

        Args:
            gspace (gspaces.GSpace3D): [description]
            trivials (enn.FieldType): field type containing irreps of freq 0
            gated (enn.FieldType): field type containing higher order freq
            gates (enn.FieldType): field type containing scalar gates for the higher order freq nonlinearities
        """
        self.trivials = trivials
        self.gated = gated
        self.gates = gates

        combined = self.trivials
        if self.gated and self.gates:
            combined += self.gated
            combined += self.gates
        super().__init__(gspace, (combined).representations)

    def no_gates(self):
        """ Returns a new field type with irreps only without the gates

        Returns:
            enn.FieldType: the direct sum of trivials and higher freq irreps
        """
        combined = self.trivials
        if self.gated: combined += self.gated
        return enn.FieldType(self.gspace, combined.representations)

    @classmethod
    def build(cls,
            gspace: gspaces.GSpace3D,
            channels: int,
            type: str = 'spherical',
            max_freq: int = 2,
            direct_sum: bool = True,
            channels_hard_limit: bool = False) -> 'GatedFieldType':
        """ Creates a gated field type with the specified number of channels

        Args:
            gspace (gspaces.GSpace3D): group space w.r.t which the features will be equivariant
            channels (int): total number of channels (check also channels_hard_limit)
            type (str, optional): Feature type, can be one of [trivial, spherical, so3]. Defaults to 'spherical'.
            max_freq (int, optional): Maximum frequency for the irreps. Defaults to 2.
            direct_sum (bool, optional): If set to true irreps of frequency greater than 0
                will be combined in a directsum and a single gate will be used for them. Otherwise each irrep
                gets a separate gate. Defaults to True.
            channels_hard_limit (bool, optional): If set to true all the features including the gates
                must not exceed the value specified in channels, otherwise as many features as possible are
                fit within the specified channels and the gates are added as extra channels. Defaults to False.

        Raises:
            ValueError: if the type is not one of the following: [trivial, spherical, so3]
            ValueError: if the dimensionality of the requested type exceeds the number of 
                requested channels

        Returns:
            GatedFieldType: instance of the class that is a wrapper over enn.FieldType of the direct sum
                of trivials, gated and gates with the additional components as attributes and some additional
                functionality
        """

        if type == 'trivial':
            trivials = enn.FieldType(gspace, channels*[gspace.trivial_repr])
            gated, gates = None, None
            return cls(gspace, trivials, gated, gates)

        if type == 'spherical':
            dim = sum([2*l+1 for l in range(max_freq+1)])
        elif type == 'so3':
            dim = sum([(2*l+1)**2 for l in range(max_freq+1)])
        else:
            raise ValueError(f'type must be on of ["trivial", "spherical", "so3"], given: {type}')
        
        if channels_hard_limit:
            dim += 1 if direct_sum else max_freq # 1 gate if using directsum, otherwise gate per freq greater than 0
        
        if dim > channels:
            raise ValueError(f'Cannot build field of type {type}. The field requires at least {dim} channels, {channels} channels specified')

        n_irreps, n_rem = channels // dim, channels % dim
        n_triv = n_irreps + n_rem

        higher_freqs = [gspace.irrep(i) for i in range(1,max_freq+1)]
        higher_freqs = [directsum(higher_freqs)] if direct_sum else higher_freqs

        trivials = enn.FieldType(gspace, n_triv*[gspace.trivial_repr])
        gated = enn.FieldType(gspace, n_irreps*higher_freqs)
        gates = enn.FieldType(gspace, n_irreps*len(higher_freqs)*[gspace.trivial_repr])

        return cls(gspace, trivials, gated, gates)

    def __add__(self, other: 'GatedFieldType') -> 'GatedFieldType':
        assert self.gspace == other.gspace
        if self.gated and other.gated:
            gated = self.gated + other.gated
        elif self.gated:
            gated = self.gated
        elif other.gated:
            gated = other.gated
        else:
            gated = None
        if self.gates and other.gates:
            gates = self.gates + other.gates
        elif self.gates:
            gates = self.gates
        elif other.gates:
            gates = other.gates
        else:
            gates = None
        return GatedFieldType(self.gspace, self.trivials + other.trivials, gated, gates)


class BaseEquiv(BaseSegmentation):
    def __init__(self, in_channels=1, kernel_size=3, padding=0, **kwargs):
        super().__init__(**kwargs)

        self.padding = self.parse_padding(padding, kernel_size)
        self.input_type = enn.FieldType(self.gspace, in_channels*[self.gspace.trivial_repr])

    @property
    @abstractmethod
    def gspace(self): pass

    def init(self):
        for m in self.modules():
            if isinstance(m, enn.R3Conv):
                enn.init.generalized_he_init(m.weights.data, m.basisexpansion, cache=True)

    def pre_forward(self, x):
        if isinstance(x, torch.Tensor):
            x = enn.GeometricTensor(x, self.input_type)
        assert x.type == self.input_type
        return x

    def evaluate_output_shape(self, input_shape):
        return input_shape[..., self.crop:-self.crop, self.crop:-self.crop, self.crop: -self.crop]

