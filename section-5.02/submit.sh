#!/bin/bash

#SBATCH --job-name=HIP
#SBATCH --gpus=2
#SBATCH --time=00:01:00

#SBATCH --partition=gpu
#SBATCH --qos=gpu-shd

module load PrgEnv-amd
module load rocm
module load craype-accel-amd-gfx90a
module load craype-x86-milan
module load cray-libsci_acc

export MPICH_GPU_SUPPORT_ENABLED=1

srun --ntasks=2 --cpus-per-task=8 ./a.out