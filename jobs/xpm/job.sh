#!/bin/bash

#SBATCH --job-name=xpm_train_gpu
#SBATCH --output=runs/slurm_%j_out.txt
#SBATCH --error=runs/slurm_%j_err.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --partition=gpu_b200
#SBATCH --time=1:00:00

# Submit from the repo root. Set WANDB_API_KEY in the environment (do not commit it).

module load CUDA
module load cuDNN
module load miniconda
conda activate torch-gpu
python -m approaches.xpm_unitary.train --config approaches/xpm_unitary/configs/analysis/trace-spm-off/bs16.yaml
