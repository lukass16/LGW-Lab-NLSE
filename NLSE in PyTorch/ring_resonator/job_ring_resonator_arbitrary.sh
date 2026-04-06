#!/bin/bash

#SBATCH --job-name=ring_res_arbitrary
#SBATCH --output=/home/lrk42/"LGW Lab"/unitary/LGW-Lab-NLSE/"NLSE in PyTorch"/ring_res_arb_out.txt
#SBATCH --error=/home/lrk42/"LGW Lab"/unitary/LGW-Lab-NLSE/"NLSE in PyTorch"/ring_res_arb_err.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=6:00:00

module load CUDA
module load cuDNN
module load miniconda
conda activate torch-gpu
python ring_resonator_arbitrary.py --N_train 1000 --output_dir results/ring_resonator_arbitrary
