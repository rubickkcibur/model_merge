#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --error job.%j.err
#SBATCH -p gov
#SBATCH --exclusive
#SBATCH -J myFirstGPUJob
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:8

bash eval.sh
