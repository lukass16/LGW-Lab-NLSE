#!/bin/bash

#SBATCH --job-name=xpm_train_gpu
#SBATCH --output=/home/lrk42/"LGW Lab"/unitary/LGW-Lab-NLSE/"NLSE in PyTorch"/runs/slurm_%j_out.txt
#SBATCH --error=/home/lrk42/"LGW Lab"/unitary/LGW-Lab-NLSE/"NLSE in PyTorch"/runs/slurm_%j_err.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --partition=gpu
#SBATCH --time=0:30:00

export WANDB_API_KEY=fa6aa2781732fe60c4a13ca88d919b7f56360fd8

cd /home/lrk42/"LGW Lab"/unitary/LGW-Lab-NLSE/"NLSE in PyTorch"

module load CUDA
module load cuDNN
module load miniconda
conda activate torch-gpu
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
