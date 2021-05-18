#!/bin/bash
#SBATCH --job-name="wandb"
#SBATCH --nodes=1 # Number of nodes
#SBATCH --time=4:00:00 # excepted wall clock time
#SBATCH --gpus=1
#SBATCH --partition=gpu_shared
#SBATCH --signal=SIGUSR1@90 #enables pl to save a checkpoint if the job is to be terminated
#SBATCH --output=out/%x.%j.out

module load 2020
module load Anaconda3/2020.02
module load CUDA/11.0.2-GCC-9.3.0

export SLURM_JOB_NAME=bash # hack to make pl + ray tune work on slurm

source activate asoca

cp -r $HOME/ASOCA "$TMPDIR" # copy data to scratch

cd "$TMPDIR"/ASOCA

echo "Grid search with config $1"
wandb agent dkalev/ASOCA/zc3sofjz

echo DONE