from datetime import timedelta
import subprocess
from pathlib import Path
import textwrap
import time
import os

model_specific = {
    "cnn": {
        "default": { "dataset": { "patch_size": [128,128,128], "patch_stride": [120,120,120] }, "train": { "cnn": { "arch": "default" } } },
        "strided": { "dataset": { "patch_size": [128,128,128], "patch_stride": [106,106,106] }, "train": { "cnn": { "arch": "strided" } } },
        "patch64": { "dataset": { "patch_size": [64 ,64 ,64 ], "patch_stride": [120,120,120] }, "train": { "cnn": { "arch": "patch64" } } },
    },
    "unet":    { "dataset": { "patch_size": [128,128,128], "patch_stride": [108,108,108] } },
    "cubereg": { "dataset": { "patch_size": [128,128,128], "patch_stride": [114,114,114] } },
    "icoreg":  { "dataset": { "patch_size": [128,128,128], "patch_stride": [114,114,114] } },
    "eunet":   { "dataset": { "patch_size": [128,128,128], "patch_stride": [108,108,108] } },
    "scnn": {
        "trivial":   { "dataset": { "patch_size": [128,128,128], "patch_stride": [114,114,114] }, "train": { "steerable": { "type": "trivial" } } },
        "spherical": { "dataset": { "patch_size": [128,128,128], "patch_stride": [114,114,114] }, "train": { "steerable": { "type": "spherical" } } },
        "so3":       { "dataset": { "patch_size": [128,128,128], "patch_stride": [114,114,114] }, "train": { "steerable": { "type": "so3" } } },
    },
    "sftcnn": {
        "trivial":   { "dataset": { "patch_size": [128,128,128], "patch_stride": [110,110,110] }, "train": { "steerable": { "type": "trivial" } } },
        "spherical": { "dataset": { "patch_size": [128,128,128], "patch_stride": [110,110,110] }, "train": { "steerable": { "type": "spherical" } } },
        "so3":       { "dataset": { "patch_size": [128,128,128], "patch_stride": [110,110,110] }, "train": { "steerable": { "type": "so3" } } },
    },
}

basic_config = [
    "--dataset.normalize=true",
    "--dataset.data_clip_range=percentile",
    "--dataset.num_workers=4",
    "--dataset.oversample=True",
    "--dataset.crop_empty=False",
    "--dataset.data_dir=dataset/processed",
    "--dataset.sourcepath=dataset/ASOCA2020Data.zip",
    "--train.n_epochs=10",
    "--train.batch_size=1",
    "--train.optim_type=adam",
    "--train.kernel_size=3",
    "--train.loss_type=dice",
]

def parse(dict_input):

    def _parse(dict_input):
        res = []
        for key, val in dict_input.items():
            if not isinstance(val, dict):
                res.append([key, str(val)])
            else:
                res.extend([ [key] + rest for rest in _parse(val) ])
        return res

    return [f'--{".".join(x[:-1])}={x[-1]}' for x in _parse(dict_input)]

def combine_config(basic, model_specific, experiment):
    config = model_specific + basic
    exp_keys = [ x.split('=')[0] for x in experiment ]
    config = [ x for x in config if not any([key in x for key in exp_keys]) ]
    config = config + experiment
    return config

def get_job(run_time, n_gpus, config):
    hparams = ' \\\n\t'.join(config)
    command = f'python -u train.py {hparams}'

    script = f"""
#!/bin/bash
#SBATCH --job-name="ASOCA"
#SBATCH --nodes=1 # Number of nodes
#SBATCH --time={run_time} # expected wall clock time
#SBATCH --partition=gpu_shared # specify partition
#SBATCH --gpus={n_gpus}
#SBATCH --signal=SIGUSR1@90 #enables pl to save a checkpoint if the job is to be terminated
#SBATCH --output=out/%x.%j.out

module load 2020
module load Anaconda3/2020.02
module load CUDA/11.0.2-GCC-9.3.0

source activate asoca

cp -r $HOME/ASOCA_final "$TMPDIR" # copy data to scratch

cd "$TMPDIR"/ASOCA_final

echo "Starting to train"
{command}

cp -r wandb/* $HOME/ASOCA_final/wandb

echo DONE
    """
    script = '\n'.join(script.split('\n')[1:]) # remove first empty line
    return script

if __name__ == '__main__':
    run_time = timedelta(hours=1)

    exps = [
        ('unet', { 'train': { 'gpus': 4, } }),
        (('cnn', 'default'), { 'train': { 'gpus': 4 } }),
        (('cnn', 'strided'), { 'train': { 'gpus': 4 } }),
        (('cnn', 'patch64'), { 'train': { 'gpus': 4 } }),
        ('unet', { 'train': { 'gpus': 4 } }),
        (('scnn', 'trivial'), { 'train': { 'gpus': 1 } }),
        (('scnn', 'spherical'), { 'train': { 'gpus': 1 } }),
        (('scnn', 'so3'), { 'train': { 'gpus': 1 } }),
        (('sftcnn', 'trivial'), { 'train': { 'gpus': 1 } }),
        (('sftcnn', 'spherical'), { 'train': { 'gpus': 1 } }),
        (('sftcnn', 'so3'), { 'train': { 'gpus': 1 } }),
        ('eunet', { 'train': { 'gpus': 4 } }),
    ]

    for model, exp in exps:
        if isinstance(model, str):
            model_spec = parse(model_specific[model])
            model_spec.append(f'--train.model={model}')
        elif isinstance(model, tuple):
            model_spec = parse(model_specific[model[0]][model[1]])
            model_spec.append(f'--train.model={model[0]}')
        else:
            raise ValueError()

        n_gpus = exp['train']['gpus']
        config = combine_config(basic_config, model_spec, parse(exp))
        job_content = get_job(run_time, n_gpus, config)

        try:
            f_name = f'jobs/job-{int(time.time())}.sh'
            with open(f_name, 'w') as f: f.write(job_content)
            subprocess.run(['sbatch',  f_name])
        finally:
            if Path(f_name).is_file(): os.remove(f_name)

