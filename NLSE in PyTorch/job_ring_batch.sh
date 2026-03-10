#!/bin/bash

#SBATCH --job-name=ring_res_batch
#SBATCH --output=/home/lrk42/"LGW Lab"/unitary/LGW-Lab-NLSE/"NLSE in PyTorch"/ring_batch_out.txt
#SBATCH --error=/home/lrk42/"LGW Lab"/unitary/LGW-Lab-NLSE/"NLSE in PyTorch"/ring_batch_err.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=6:00:00

module load CUDA
module load cuDNN
module load miniconda
conda activate torch-gpu
python ring_resonator_scaling_batch.py --config configs/config_ring_scaling_fidelity.yaml
