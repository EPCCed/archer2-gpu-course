#!/bin/bash

#SBATCH --job-name=HIP
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --time=00:02:00

#SBATCH --partition=gpu
#SBATCH --qos=gpu-shd

srun --ntasks=1 --cpus-per-task=1 rocprof --sys-trace ./a.out