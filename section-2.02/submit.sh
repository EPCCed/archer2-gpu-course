#!/bin/bash

#SBATCH --time=00:01:00
#SBATCH --partition=gpu
#SBATCH --qos=gpu-shd
#SBATCH --gpus=1

./a.out

