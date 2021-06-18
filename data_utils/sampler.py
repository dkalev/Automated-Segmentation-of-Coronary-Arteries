import torch
import numpy as np
from torch.utils.data import Sampler


class ASOCASampler(Sampler):
    def __init__(self, shapes,
                shuffle=False,
                oversample=False,
                binary_weights=False,
                perc_per_epoch=1,
                oversample_coef=100,
                alpha=1):
        self.shapes = shapes
        self.gen = np.random.default_rng()
        self.shuffle = shuffle
        self.oversample = oversample
        self.binary_weights = binary_weights
        self.perc_per_epoch = perc_per_epoch
        self.alpha = alpha

    @property
    def max_patches(self):
        if not hasattr(self, '_max_patches'):
            self._max_patches = max([ self.shapes[fid]['n_patches'] for fid in self.file_ids ])
        return self._max_patches

    @property
    def total_patches(self):
        if not hasattr(self, '_total_patches'):
            self._total_patches = sum([ self.shapes[fid]['n_patches'] for fid in self.file_ids ])
        return self._total_patches

    def sample_ids(self):
        file_ids = list(self.shapes.keys())
        n_samples = max(1, int(self.perc_per_epoch*len(file_ids)))
        if n_samples < len(file_ids):
            file_ids = np.random.choice(file_ids, n_samples, replace=False)
        file_ids = self.gen.permutation(file_ids) if self.shuffle else file_ids
        return file_ids.tolist()

    @property
    def file_ids(self):
        if not hasattr(self, '_file_ids'):
            self._file_ids = self.sample_ids()
        return self._file_ids

    @file_ids.setter
    def file_ids(self, val):
        self._file_ids = val

    def get_sample_weights(self, samples):
        if self.binary_weights:
            samples = np.round(samples).astype(int)
            p = np.ones_like(samples) / len(samples)
            if not np.any(samples): return p # if all zeros sample uniformly
            ratio = len(samples[samples==0]) / len(samples[samples==1]) * self.alpha
            p[samples==1] *= ratio
            p[samples==0] /= ratio
            return p
        else:
            samples = np.array(samples)
            return samples / np.sum(samples)


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
        for file_id in self.file_ids:
            for index in self.get_patch_idxs(file_id):
                yield file_id * self.max_patches + index

    def __len__(self):
        return self.total_patches

