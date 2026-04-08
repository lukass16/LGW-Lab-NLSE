#!/bin/bash

#SBATCH -p priority_gpu   # (include this line only if you want to request the priority tier)
#SBATCH -A prio_lgw23     # (include this line only if you want to request the priority tier)
#SBATCH --gpus=rtx_5000_ada:1     # specify partition and number of GPUs
#SBATCH -c 1              # number of cpu cores
#SBATCH -t 01:00:00    
#SBATCH --mem=32G
#SBATCH -J lukass_priority_test
#SBATCH -o slurm-%j.out

export WANDB_API_KEY=fa6aa2781732fe60c4a13ca88d919b7f56360fd8

cd /home/lrk42/"LGW Lab"/unitary/LGW-Lab-NLSE/"NLSE in PyTorch"

module load CUDA
module load cuDNN
module load miniconda
conda activate torch-gpu
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
