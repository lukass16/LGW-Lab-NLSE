#!/bin/bash

#SBATCH --job-name=xpm_train_gpu
#SBATCH --output=/home/lrk42/"LGW Lab"/unitary/LGW-Lab-NLSE/"NLSE in PyTorch"/trainjobout_gpu.txt
#SBATCH --error=/home/lrk42/"LGW Lab"/unitary/LGW-Lab-NLSE/"NLSE in PyTorch"/trainjoberr_gpu.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=3:00:00

export WANDB_API_KEY=fa6aa2781732fe60c4a13ca88d919b7f56360fd8

module load CUDA
module load cuDNN
module load miniconda
conda activate torch-gpu
python train_xpm_identity.py --config configs/config.yaml