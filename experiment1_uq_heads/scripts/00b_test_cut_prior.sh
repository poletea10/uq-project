#!/bin/bash
# Step-2 gate (phase1_protocol.md §3, §4.2): observed-cut prior and task filters.
#   sbatch experiment1_uq_heads/scripts/00b_test_cut_prior.sh
#SBATCH --job-name=p1_00b_test_cut_prior
#SBATCH --partition=short
#SBATCH --time=00:45:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/private/groups/ioannidislab/UQ_decoupling_exp/slurm_logs/00b_test_cut_prior_%j.out
#SBATCH --error=/private/groups/ioannidislab/UQ_decoupling_exp/slurm_logs/00b_test_cut_prior_%j.err

eval "$(conda shell.bash hook)"
conda activate uq-iltm-bio
export LD_LIBRARY_PATH="/private/groups/ioannidislab/ajaras/conda_envs/uq-iltm-bio/lib:$LD_LIBRARY_PATH"
export PYTHONHASHSEED=0
export TMPDIR=/tmp            # node-local: multiprocessing / joblib temp dirs cannot be cleaned up on NFS
cd /private/home/ajaras/iLTM-bio/UQ_project
python experiment1_uq_heads/scripts/00b_test_cut_prior.py "$@"
