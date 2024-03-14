#!/bin/bash

#SBATCH --job-name=HIP
#SBATCH --gpus=1
#SBATCH --time=00:01:00

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=z19
#SBATCH --partition=gpu
#SBATCH --qos=gpu-shd

srun --ntasks=1 --cpus-per-task=1 ./a.out