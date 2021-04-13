#!/bin/bash
#SBATCH --job-name="Agrid"
#SBATCH --nodes=1 # Number of nodes
#SBATCH --time=12:00:00 # excepted wall clock time
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --signal=SIGUSR1@90 #enables pl to save a checkpoint if the job is to be terminated
#SBATCH --output=out/%x.%j.out

module load 2020
module load Anaconda3/2020.02
module load CUDA/11.0.2-GCC-9.3.0

export SLURM_JOB_NAME=bash # hack to make pl + ray tune work on slurm

source activate dl

cp -r $HOME/Automated-Segmentation-of-Coronary-Arteries "$TMPDIR" # copy data to scratch

cd "$TMPDIR"/Automated-Segmentation-of-Coronary-Arteries

echo "Grid search"
python -u grid_search.py --n_epochs=15

cp -r ray_tune/* $HOME/Automated-Segmentation-of-Coronary-Arteries/ray_tune

echo DONE