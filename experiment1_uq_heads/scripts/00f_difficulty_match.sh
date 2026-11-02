#!/bin/bash
# Step-2 prior-level gate: observed-cut vs unmodified graph_scm difficulty (TabICLv2 CRPS, ExtraTrees R^2),
# filter rejection rates, y-ancestor sizes, t_A distribution. Tune the target graph on failure, never the filters.
#   sbatch experiment1_uq_heads/scripts/00f_difficulty_match.sh
#   sbatch experiment1_uq_heads/scripts/00f_difficulty_match.sh --hidden-scale 0.5      # a tuning rerun
#SBATCH --job-name=p1_00f_difficulty_match
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --time=2:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=16
#SBATCH --output=/private/groups/ioannidislab/UQ_decoupling_exp/slurm_logs/00f_difficulty_match_%j.out
#SBATCH --error=/private/groups/ioannidislab/UQ_decoupling_exp/slurm_logs/00f_difficulty_match_%j.err

eval "$(conda shell.bash hook)"
conda activate uq-iltm-bio
export LD_LIBRARY_PATH="/private/groups/ioannidislab/ajaras/conda_envs/uq-iltm-bio/lib:$LD_LIBRARY_PATH"
export PYTHONHASHSEED=0
export TMPDIR=/tmp            # node-local: multiprocessing / joblib temp dirs cannot be cleaned up on NFS
cd /private/home/ajaras/iLTM-bio/UQ_project
python experiment1_uq_heads/scripts/00f_difficulty_match.py "$@"
