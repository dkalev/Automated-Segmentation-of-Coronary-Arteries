#!/bin/bash
#SBATCH --job-name="ASOCA"
#SBATCH --nodes=1 # Number of nodes
#SBATCH --time=04:00:00 # expected wall clock time
#SBATCH --partition=gpu_shared # specify partition
#SBATCH --gpus=1
#SBATCH --signal=SIGUSR1@90 #enables pl to save a checkpoint if the job is to be terminated
#SBATCH --output=out/%x.%j.out

module load 2020
module load Anaconda3/2020.02
module load CUDA/11.0.2-GCC-9.3.0

source activate asoca

cp -r $HOME/ASOCA "$TMPDIR" # copy data to scratch

cd "$TMPDIR"/ASOCA

echo "Training with config file located at $1"
python -u train.py --config_path=$1

cp -r logs/* $HOME/ASOCA/logs

echo DONE