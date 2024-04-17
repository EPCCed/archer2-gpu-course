#!/bin/bash

#SBATCH --job-name=HIP
#SBATCH --gpus=2
#SBATCH --time=00:01:00

#SBATCH --partition=gpu
#SBATCH --qos=gpu-shd

srun --ntasks=2 --cpus-per-task=8 ./a.out