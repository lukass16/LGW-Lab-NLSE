#!/bin/bash

#SBATCH --job-name=xpm_train_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --partition=priority_gpu
#SBATCH --account=prio_lgw23
#SBATCH --mem=5G
#SBATCH --time=2:00:00

# Submit from the repo root. Set WANDB_API_KEY in the environment (do not commit it).

module load CUDA
module load cuDNN
module load miniconda
conda activate torch-gpu
python -m approaches.xpm_unitary.train --config approaches/xpm_unitary/configs/analysis/trace-spm-off/bs8.yaml
