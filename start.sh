#!/bin/bash

#SBATCH --job-name=baseline
#SBATCH -e output/slurm_%A.err
#SBATCH -o output/slurm_%A.out

#SBATCH --ntasks=5
#SBATCH --time=0-10:00:00
#SBATCH --account=desinformation
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu


module purge
module load miniconda
module load gcc/11.3.0
conda activate fallacy_det

python run.py 