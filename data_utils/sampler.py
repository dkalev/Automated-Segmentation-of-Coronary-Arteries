import torch
import numpy as np
from torch.utils.data import Sampler


class ASOCASampler(Sampler):
    def __init__(self, shapes, shuffle=False, oversample=False, binary_weights=False, alpha=1):
        self.shapes = shapes
        self.gen = np.random.default_rng()
        self.shuffle = shuffle
        self.oversample = oversample
        self.binary_weights = binary_weights
        self.alpha = alpha

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

    def get_sample_weights(self, samples):
        if self.binary_weights:
            samples = torch.round(samples).int()
            p = torch.ones_like(samples) / len(samples)
            ratio = len(samples[samples==0]) / len(samples[samples==1]) * self.alpha
            p[samples==1] *= ratio
            p[samples==0] /= ratio
            return p.tolist()
        else:
            return samples / np.sum(samples)
    
    def get_file_ids(self):
        file_ids = list(self.shapes.keys())
        if self.shuffle:
            return self.gen.permutation(file_ids)
        else:
            return file_ids
    
    def get_patch_idxs(self, file_id):
        meta = self.shapes[file_id]
        n_patches = meta['n_patches']
        if self.oversample:
            weights = self.get_sample_weights(meta['foreground_ratio'])
            return self.gen.choice(range(n_patches), n_patches, p=weights)
        elif self.shuffle:
            return self.gen.permutation(range(n_patches))
        else:
            return range(n_patches)

    def __iter__(self):
        for file_id in self.get_file_ids():
            for index in self.get_patch_idxs(file_id):
                yield file_id * self.max_patches + index
                
    def __len__(self):
        return self.total_patches
