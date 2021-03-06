import torch
import numpy as np
from torch.utils.data import Sampler
from copy import deepcopy


class ASOCASampler(Sampler):
    def __init__(self, vol_meta,
                shuffle=False,
                oversample=False,
                perc_per_epoch=1,
                weight_update_step=0.01):
        self.gen = np.random.default_rng(0)
        self.shuffle = shuffle
        self.oversample = oversample
        self.perc_per_epoch = perc_per_epoch
        self.w_step = weight_update_step

        self.vol_meta = deepcopy(vol_meta)
        if self.oversample:
            for meta in self.vol_meta.values():
                meta['weights'] = self.get_sample_weights(meta['foreground_ratio'])

    @property
    def max_patches(self):
        if not hasattr(self, '_max_patches'):
            self._max_patches = max([ self.vol_meta[fid]['n_patches'] for fid in self.file_ids ])
        return self._max_patches

    @property
    def total_patches(self):
        if not hasattr(self, '_total_patches'):
            self._total_patches = sum([ self.vol_meta[fid]['n_patches'] for fid in self.file_ids ])
        return self._total_patches

    def sample_ids(self):
        file_ids = list(self.vol_meta.keys())
        n_samples = max(1, int(self.perc_per_epoch*len(file_ids)))
        if n_samples < len(file_ids):
            file_ids = np.random.choice(file_ids, n_samples, replace=False)
        file_ids = self.gen.permutation(file_ids) if self.shuffle else np.array(file_ids)
        return file_ids.tolist()

    @property
    def file_ids(self):
        if not hasattr(self, '_file_ids'):
            self._file_ids = self.sample_ids()
        return self._file_ids

    @file_ids.setter
    def file_ids(self, val):
        self._file_ids = val
        self._max_patches = max([ self.vol_meta[fid]['n_patches'] for fid in val ])
        self._total_patches = sum([ self.vol_meta[fid]['n_patches'] for fid in val ])

    def get_sample_weights(self, samples):
        samples = np.array(samples)
        samples = samples + np.min(samples[samples>0])
        return samples / np.sum(samples)

    def update_patch_weights(self, losses):
        for file_id, patches in losses.items():
            prev_weights = self.vol_meta[file_id]['weights']
            patch_losses = np.zeros_like(prev_weights)
            for patch_id, loss in patches.items(): patch_losses[patch_id] = loss

            patch_losses /= np.max(patch_losses)
            patch_losses *= np.max(prev_weights[patch_losses>0])
            weights_next = prev_weights + self.w_step * patch_losses
            weights_next /= np.sum(weights_next)
            self.vol_meta[file_id]['weights'] = weights_next

    def get_patch_idxs(self, file_id):
        meta = self.vol_meta[file_id]
        n_patches = meta['n_patches']
        if self.oversample:
            return self.gen.choice(range(n_patches), n_patches, p=meta['weights'])
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

