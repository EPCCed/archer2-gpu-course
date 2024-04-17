#!/bin/bash

#SBATCH --job-name=HIP
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:02:00

#SBATCH --partition=gpu
#SBATCH --qos=gpu-exc

srun --ntasks=1 --cpus-per-task=1 rocprof --hsa-trace --sys-trace ./a.out