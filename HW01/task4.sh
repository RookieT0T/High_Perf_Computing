#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -J FirstSlurm
#SBATCH -o FirstSlurm.out -e FirstSlurm.err
#SBATCH -c 2
#SBATCH --time=0-00:01:00


hostname