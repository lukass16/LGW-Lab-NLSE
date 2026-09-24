#!/bin/bash

#SBATCH --job-name=xpm_analysis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --partition=priority_gpu
#SBATCH --account=prio_lgw23
#SBATCH --mem=5G
#SBATCH --time=3:00:00

# Expects CONFIG and PROJECT_DIR from the launcher via sbatch --export.
# Set WANDB_API_KEY in the environment (do not commit it).

cd "$PROJECT_DIR"

module load CUDA
module load cuDNN
module load miniconda
conda activate torch-gpu
python -m approaches.xpm_unitary.train --config "$CONFIG"
