#!/bin/bash
#SBATCH --job-name="nnUNet"
#SBATCH --nodes=1 # Number of nodes
#SBATCH --time=12:00:00 # excepted wall clock time
#SBATCH --partition=gpu_shared # specify partition
#SBATCH --gpus=1
#SBATCH --output=out/%x.%j.out

module load 2020
module load Anaconda3/2020.02
module load CUDA/11.0.2-GCC-9.3.0

source activate dl

cp -r $HOME/ASOCA "$TMPDIR" # copy data to scratch

cd "$TMPDIR"/ASOCA

echo "Training with nnUNet"
nnUNet_train 3d_fullres nnUNetTrainerV2 Task100_ASOCA 4 --npz

cp -r "$TMPDIR"/ASOCA/dataset/nnUNet/nnUNet_trained_models/* $HOME/ASOCA/dataset/nnUNet/nnUNet_trained_models/

echo DONE