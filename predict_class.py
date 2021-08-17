from models.classification.cnn import Baseline3DClassification
from data_utils.helpers_classification import get_patch_bbox
import torch
import numpy as np
import nrrd
from tqdm import tqdm
import json


if __name__ == '__main__':
    model = Baseline3DClassification.load_from_checkpoint('/var/scratch/ebekkers/damyan/models/class-cnn-epoch=37-step=5965.ckpt')
    model = torch.nn.DataParallel(model)
    model.eval()
    model.cuda()

    vol, _ = nrrd.read('dataset/raw/ASOCA2020Data/Train/1.nrrd', index_order='C')

    with open('dataset/classification/dataset.json', 'r') as f:
        meta = json.load(f)
    
    stats = meta['stats']
    vol = np.clip(vol, stats['percentile_00_5'], stats['percentile_99_5'])
    vol = (vol - stats['mean']) / stats['std']

    dims = vol.shape
    dims_max = dims - np.array([57,150,150])
    dims_min = np.array([57,150,150])

    # center = np.array([112, 239, 142])
    # dims_min = center - 34
    # dims_max = center + 34

    bs = 500

    voxels = np.stack(np.meshgrid(
        np.arange(dims_min[0], dims_max[0]),
        np.arange(dims_min[1], dims_max[1]),
        np.arange(dims_min[2], dims_max[2]),
        indexing='ij',
    )).T.reshape((-1,3))

    remainder = len(voxels) - bs * (len(voxels)//bs)
    voxels = voxels[:-remainder]

    print(f'N voxels: {len(voxels):n}')

    voxels = voxels.reshape(-1,bs,3)

    preds = np.empty(voxels.shape[:2])
    with torch.no_grad():
        for i in tqdm(range(0, len(voxels))):
            x = np.empty((bs,68,68,68))
            for j in range(bs):
                x[j] = vol[get_patch_bbox(voxels[i][j], 68)]
            x = torch.from_numpy(x).float().unsqueeze(1).cuda()
            y = torch.sigmoid(model(x)).round()
            preds[i] = y.cpu().squeeze(-1).numpy()

            if i % 500 == 0: np.save(f'class_vol_preds_{i}.npy', preds)
    np.save(f'class_vol_preds_{remainder}_{i}.npy', preds)