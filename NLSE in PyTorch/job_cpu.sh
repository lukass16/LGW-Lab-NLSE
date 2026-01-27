#!/bin/bash

#SBATCH --job-name=xpm_train_cpu
#SBATCH --output=/home/lrk42/"LGW Lab"/unitary/LGW-Lab-NLSE/"NLSE in PyTorch"/trainjobout_cpu.txt
#SBATCH --error=/home/lrk42/"LGW Lab"/unitary/LGW-Lab-NLSE/"NLSE in PyTorch"/trainjoberr_cpu.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00

export WANDB_API_KEY=fa6aa2781732fe60c4a13ca88d919b7f56360fd8

module load miniconda
conda activate torch-gpu
python train_xpm_unitary.py --config configs/config.yaml