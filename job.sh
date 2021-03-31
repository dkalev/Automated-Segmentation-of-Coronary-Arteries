#!/bin/bash
#SBATCH --job-name="ASOCA"
#SBATCH --nodes=1 # Number of nodes
#SBATCH --time=04:00:00 # excepted wall clock time
#SBATCH --partition=gpu # specify partition
#SBATCH --gpus=4
#SBATCH --signal=SIGUSR1@90 #enables pl to save a checkpoint if the job is to be terminated
#SBATCH --output=out/%x.%j.out

module load 2020
module load Anaconda3/2020.02
module load CUDA/11.0.2-GCC-9.3.0

source activate dl

cp -r $HOME/Automated-Segmentation-of-Coronary-Arteries "$TMPDIR" # copy data to scratch

cd "$TMPDIR"/Automated-Segmentation-of-Coronary-Arteries

echo "Load dataset"
python train.py --model=$1

cp -r logs/* $HOME/Automated-Segmentation-of-Coronary-Arteries/logs

echo DONE