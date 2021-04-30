import os
import nrrd
import json
import zipfile
import shutil
import numpy as np
from tqdm import tqdm
import nibabel as nib
from pathlib import Path
from collections import OrderedDict


def main(root_dir='dataset'):
    meta = OrderedDict({
        'name': 'ASOCA',
        'tensorImageSize': '4D',
        'modality': { '0': 'CT' },
        'labels': { '0': 'background', '1': 'ca' },
        'numTest': 0,
        'test': [],
    })

    res_data_dir = Path(root_dir, 'nnUNet_raw_data/Task100_ASOCA')
    for dirname in ['imagesTr', 'labelsTr']:
        os.makedirs(Path(res_data_dir, dirname), exist_ok=True)

    with zipfile.ZipFile(Path(root_dir, 'ASOCA2020Data.zip'), 'r') as zip_ref:
        zip_ref.extractall(root_dir)
    raw_data_dir = Path(root_dir, 'ASOCA2020Data')

    for i in tqdm(range(40)):
        vol, _ = nrrd.read(Path(raw_data_dir, 'Train', f'{i}.nrrd'))
        vol = nib.Nifti1Image(vol, affine=np.eye(4))
        nib.save(vol, Path(res_data_dir, 'imagesTr', f'{i:04d}_0000.nii.gz'))

        label, _ = nrrd.read(Path(raw_data_dir, 'Train_Masks', f'{i}.nrrd'))
        label = nib.Nifti1Image(label, affine=np.eye(4))
        nib.save(label, Path(res_data_dir, 'labelsTr', f'{i:04d}.nii.gz'))

    meta['numTraining'] = 40
    meta['training'] = [ {
        'image': f'imagesTr/{i:04d}.nii.gz',
        'label': f'labelsTr/{i:04d}.nii.gz' 
        } for i in range(40)]

    with open(Path(res_data_dir, 'dataset.json'), 'w') as f:
        json.dump(meta, f, indent=4)

    shutil.rmtree(raw_data_dir)

if __name__ == '__main__':
    main()
