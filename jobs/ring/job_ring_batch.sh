#!/bin/bash

#SBATCH --job-name=ring_res_batch
#SBATCH --output=runs/ring_batch_%j_out.txt
#SBATCH --error=runs/ring_batch_%j_err.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=6:00:00

# Submit from the repo root. Set WANDB_API_KEY in the environment (do not commit it).

module load CUDA
module load cuDNN
module load miniconda
conda activate torch-gpu
python -m approaches.ring_resonator.train_scaling --config approaches/ring_resonator/configs/config_ring_scaling_trace_large.yaml
