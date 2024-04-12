#!/bin/bash

#SBATCH --job-name=HIP
#SBATCH --gpus=1
#SBATCH --time=00:01:00

#SBATCH --partition=gpu
#SBATCH --qos=gpu-shd

srun --ntasks=1 --cpus-per-task=1 ./a.out