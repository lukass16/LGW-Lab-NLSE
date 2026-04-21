#!/bin/bash

#SBATCH --job-name=xpm_analysis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --partition=priority_gpu
#SBATCH --account=prio_lgw23
#SBATCH --mem=5G
#SBATCH --time=3:00:00

export WANDB_API_KEY=fa6aa2781732fe60c4a13ca88d919b7f56360fd8

cd "$PROJECT_DIR"

module load CUDA
module load cuDNN
module load miniconda
conda activate torch-gpu
python train_xpm_unitary.py --config "$CONFIG"
