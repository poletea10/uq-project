#!/bin/bash
# Step-1 gate (phase1_protocol.md §3, §4.1): torch entropy kernels and replicate targets vs Phase 0's numpy code.
#   sbatch experiment1_uq_heads/scripts/00a_test_entropy.sh
#SBATCH --job-name=p1_00a_test_entropy
#SBATCH --partition=short
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=/private/groups/ioannidislab/UQ_decoupling_exp/slurm_logs/00a_test_entropy_%j.out
#SBATCH --error=/private/groups/ioannidislab/UQ_decoupling_exp/slurm_logs/00a_test_entropy_%j.err

eval "$(conda shell.bash hook)"
conda activate uq-iltm-bio
export LD_LIBRARY_PATH="/private/groups/ioannidislab/ajaras/conda_envs/uq-iltm-bio/lib:$LD_LIBRARY_PATH"
export PYTHONHASHSEED=0
cd /private/home/ajaras/iLTM-bio/UQ_project
python experiment1_uq_heads/scripts/00a_test_entropy.py "$@"
