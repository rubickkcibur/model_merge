#!/bin/bash
#SBATCH -o job.%j.out.log
#SBATCH --error job.%j.err.log
#SBATCH -p gov-research
#SBATCH --exclusive
#SBATCH -J myFirstGPUJob
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:8

bash eval.sh
