import numpy as np
import torch
import torch.nn.functional as F

def get_padding(input_shape, output_shape):
    input_shape = np.array(input_shape) if not isinstance(input_shape, np.ndarray) else input_shape
    output_shape = np.array(output_shape) if not isinstance(output_shape, np.ndarray) else output_shape
    pad = (output_shape - input_shape).astype(int)
    pad_left = pad // 2
    pad_right = pad - pad_left
    res = list(zip(pad_left[::-1], pad_right[::-1]))
    return tuple(int(x) for y in res for x in y )

def get_patch_padding(vol_shape, patch_size, stride):
    vol_shape = np.array(vol_shape)
    pad_shape = np.ceil((vol_shape - patch_size) / stride) * stride + patch_size
    return get_padding(vol_shape, pad_shape)

def get_n_patches(volume_shape, patch_size, stride):
    # assumes padding is applied to the volume
    volume_shape = np.array(volume_shape) if not isinstance(volume_shape, np.ndarray) else volume_shape
    return int(np.prod(np.ceil((volume_shape - patch_size) / stride) + 1))

def vol2patches(volume, patch_size, stride, padding, pad_value=0):
    if not isinstance(volume, torch.Tensor):
        volume = torch.from_numpy(volume).float()

    padded_vol = F.pad(volume, padding, value=pad_value)
    
    patches = padded_vol.unfold(0, patch_size[0], stride[0]) \
                        .unfold(1, patch_size[1], stride[1]) \
                        .unfold(2, patch_size[2], stride[2])
    patched_shape = tuple(patches.shape)
    patches = patches.reshape(-1, *patch_size)
    return patches, patched_shape

def patches2vol(patches, patch_size, stride, padding=None):
    offset = patch_size - stride
    p = patches.permute(0,3,1,4,2,5)
    p = torch.cat((p[0], p[1:][:,offset[0]:].flatten(end_dim=1)))
    p = p.permute(1,2,3,4,0)
    p = torch.cat((p[0],p[1:][:,offset[1]:].flatten(end_dim=1)))
    p = p.permute(1,2,3,0)
    p = torch.cat((p[0],p[1:][:,offset[2]:].flatten(end_dim=1)))
    
    p = p.permute(1,2,0)
    
    if padding:
        slicer = [slice(pad_left,-pad_right) for pad_left ,pad_right in list(zip(padding[0::2],padding[1::2]))][::-1]
        # otherwise zero padding results in a slice fetching zero items, instead of all
        slicer = [sl if not (sl.start == 0 and sl.stop == 0) else slice(None,None,None) for sl in slicer ]
        p = p[slicer]
        
    return p

def get_volume_pred(patches, vol_meta, stride):
    out_shape = vol_meta['shape_patched'][:3] + patches.shape[1:]
    res = patches2vol(patches.view(out_shape), stride, stride)
    output_pad = get_padding(res.shape, vol_meta['shape_orig'])
    res = F.pad(res, output_pad)
    return res