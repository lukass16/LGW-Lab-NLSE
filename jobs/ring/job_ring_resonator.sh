#!/bin/bash

#SBATCH --job-name=ring_res_scaling
#SBATCH --output=runs/ring_res_%j_out.txt
#SBATCH --error=runs/ring_res_%j_err.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=6:00:00

# Submit from the repo root.

module load CUDA
module load cuDNN
module load miniconda
conda activate torch-gpu
python -m approaches.ring_resonator.train_scaling --N_train 1000 --output_dir results/ring_resonator_scaling
