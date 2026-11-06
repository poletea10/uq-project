#!/bin/bash
# Step-5 gate (phase1_protocol.md §3, §4.4): the frozen backbone wrapper (forward route, hooks, arm B's copy).
#   sbatch experiment1_uq_heads/scripts/00d_test_backbone.sh                       # CPU, small tasks
#   sbatch --partition=gpu --gres=gpu:A5500:1 experiment1_uq_heads/scripts/00d_test_backbone.sh --device cuda
#SBATCH --job-name=p1_00d_test_backbone
#SBATCH --partition=short
#SBATCH --time=00:45:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/private/groups/ioannidislab/UQ_decoupling_exp/slurm_logs/00d_test_backbone_%j.out
#SBATCH --error=/private/groups/ioannidislab/UQ_decoupling_exp/slurm_logs/00d_test_backbone_%j.err

eval "$(conda shell.bash hook)"
conda activate uq-iltm-bio
export LD_LIBRARY_PATH="/private/groups/ioannidislab/ajaras/conda_envs/uq-iltm-bio/lib:$LD_LIBRARY_PATH"
export PYTHONHASHSEED=0
export TMPDIR=/tmp            # node-local: multiprocessing / joblib temp dirs cannot be cleaned up on NFS
cd /private/home/ajaras/iLTM-bio/UQ_project
python experiment1_uq_heads/scripts/00d_test_backbone.py "$@"
