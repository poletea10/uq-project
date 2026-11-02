#!/bin/bash
# T1 toy set, CPU methods (hgp).
#   sbatch --array=0-2 experiment0_baseline/scripts/01_toy.sh --method hgp     # one toy rung per array task
#   sbatch experiment0_baseline/scripts/01_toy.sh --figures-only               # after all rungs are done
#   sbatch experiment0_baseline/scripts/01_toy.sh --method hgp                 # all rungs in one job + figures
#SBATCH --job-name=p0_01_toy
#SBATCH --partition=medium
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/private/home/ajaras/iLTM-bio/UQ_project/experiment0_baseline/outputs/logs/01_toy_%A_%a.out
#SBATCH --error=/private/home/ajaras/iLTM-bio/UQ_project/experiment0_baseline/outputs/logs/01_toy_%A_%a.err

eval "$(conda shell.bash hook)"
conda activate uq-iltm-bio
export LD_LIBRARY_PATH="/private/groups/ioannidislab/ajaras/conda_envs/uq-iltm-bio/lib:$LD_LIBRARY_PATH"
export PYTHONHASHSEED=0
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=2"   # HGP: one numpyro chain per host device
export PYTENSOR_FLAGS="compiledir=/tmp/pytensor_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
cd /private/home/ajaras/iLTM-bio/UQ_project
python experiment0_baseline/scripts/01_toy.py "$@"
