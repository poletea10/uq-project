#!/bin/bash
#   sbatch experiment0_baseline/scripts/00_curate_toy.sh
#SBATCH --job-name=p0_00_curate_toy
#SBATCH --partition=short
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --output=/private/home/ajaras/iLTM-bio/UQ_project/experiment0_baseline/outputs/logs/00_curate_toy_%j.out
#SBATCH --error=/private/home/ajaras/iLTM-bio/UQ_project/experiment0_baseline/outputs/logs/00_curate_toy_%j.err

eval "$(conda shell.bash hook)"
conda activate uq-iltm-bio
export LD_LIBRARY_PATH="/private/groups/ioannidislab/ajaras/conda_envs/uq-iltm-bio/lib:$LD_LIBRARY_PATH"
export PYTHONHASHSEED=0
cd /private/home/ajaras/iLTM-bio/UQ_project
python experiment0_baseline/scripts/00_curate_toy.py "$@"
