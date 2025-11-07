s#!/bin/bash

#SBATCH --job-name=train
#SBATCH --output=trainjobout.txt
#SBATCH --error=trainjoberr.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=5:00:00

module load CUDA
module load cuDNN
module load miniconda
conda activate base
python train_xpm_identity.py --config configs/config.yaml