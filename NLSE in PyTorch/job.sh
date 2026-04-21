#!/bin/bash

#SBATCH --job-name=xpm_train_gpu
#SBATCH --output=/home/lrk42/LGW-Lab-NLSE/"NLSE in PyTorch"/runs/slurm_%j_out.txt
#SBATCH --error=/home/lrk42/LGW-Lab-NLSE/"NLSE in PyTorch"/runs/slurm_%j_err.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=1:00:00

export WANDB_API_KEY=fa6aa2781732fe60c4a13ca88d919b7f56360fd8

cd /home/lrk42/LGW-Lab-NLSE/"NLSE in PyTorch"

module load CUDA
module load cuDNN
module load miniconda
conda activate torch-gpu
python train_xpm_unitary.py --config configs/analysis/trace-spm-off/bs16.yaml
