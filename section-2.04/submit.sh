#!/bin/bash

#SBATCH --job-name=HIP
#SBATCH --gpus=1
#SBATCH --time=00:01:00

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=[budget code]
#SBATCH --partition=gpu
#SBATCH --qos=gpu-shd

HSA_XNACK=1

srun --ntasks=1 --cpus-per-task=1 rocprof --hsa-trace --sys-trace ./a.out