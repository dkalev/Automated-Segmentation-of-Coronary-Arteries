import numpy as np
from torch.utils.data import Sampler


class ASOCASampler(Sampler):
    def __init__(self, shapes, shuffle=False):
        self.shapes = shapes
        self.gen = np.random.default_rng()
        self.shuffle = shuffle

    @property
    def max_patches(self):
        if not hasattr(self, '_max_patches'):
            self._max_patches = max([ v['n_patches'] for v in self.shapes.values() ])
        return self._max_patches
    
    @property
    def total_patches(self):
        if not hasattr(self, '_total_patches'):
            self._total_patches = sum([ v['n_patches'] for v in self.shapes.values() ])
        return self._total_patches 
    
    def get_file_ids(self):
        file_ids = list(self.shapes.keys())
        if self.shuffle:
            return self.gen.permutation(file_ids)
        else:
            return file_ids
    
    def get_patch_idxs(self, file_id):
        shape = self.shapes[file_id]
        if self.shuffle:
            return self.gen.permutation(range(shape['n_patches']))
        else:
            return range(shape['n_patches'])

    def __iter__(self):
        for file_id in self.get_file_ids():
            for index in self.get_patch_idxs(file_id):
                yield file_id * self.max_patches + index
                
    def __len__(self):
        return self.total_patches
