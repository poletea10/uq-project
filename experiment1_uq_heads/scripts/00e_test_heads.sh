#!/bin/bash
# Step-7 gate (phase1_protocol.md §3, §4.5): heads, loss, metrics and bank validation (CPU, small real tasks; the
# validation checks read the smoke bank's mini-bank, and walk the real bank's task records without a forward).
#   sbatch experiment1_uq_heads/scripts/00e_test_heads.sh
#   sbatch experiment1_uq_heads/scripts/00e_test_heads.sh --skip-real-bank      # without the real-bank walk
#SBATCH --job-name=p1_00e_test_heads
#SBATCH --partition=short
#SBATCH --time=00:45:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --output=/private/groups/ioannidislab/UQ_decoupling_exp/slurm_logs/00e_test_heads_%j.out
#SBATCH --error=/private/groups/ioannidislab/UQ_decoupling_exp/slurm_logs/00e_test_heads_%j.err

eval "$(conda shell.bash hook)"
conda activate uq-iltm-bio
export LD_LIBRARY_PATH="/private/groups/ioannidislab/ajaras/conda_envs/uq-iltm-bio/lib:$LD_LIBRARY_PATH"
export PYTHONHASHSEED=0
export TMPDIR=/tmp            # node-local: multiprocessing / joblib temp dirs cannot be cleaned up on NFS
cd /private/home/ajaras/iLTM-bio/UQ_project
python experiment1_uq_heads/scripts/00e_test_heads.py "$@"
