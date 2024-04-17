#!/bin/bash

#SBATCH --job-name=dscal
#SBATCH --gpus=1
#SBATCH --time=00:01:00

#SBATCH --partition=gpu
#SBATCH --qos=gpu-shd

# Check assigned GPU
srun --ntasks=1 rocm-smi

srun --ntasks=1 --cpus-per-task=1 ./a.out

exit 0
