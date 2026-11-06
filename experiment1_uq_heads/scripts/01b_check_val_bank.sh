#!/bin/bash
# Step-6 gate (phase1_protocol.md §3 step 6): the validation-bank checks, after 01_make_val_bank.sh.
#   sbatch experiment1_uq_heads/scripts/01b_check_val_bank.sh
#   sbatch experiment1_uq_heads/scripts/01b_check_val_bank.sh --max-gs-tasks 100      # quicker PPD comparison
#   sbatch experiment1_uq_heads/scripts/01b_check_val_bank.sh --bank /private/groups/ioannidislab/UQ_decoupling_exp/val_bank_smoke
#SBATCH --job-name=p1_01b_check_val_bank
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=34
#SBATCH --output=/private/groups/ioannidislab/UQ_decoupling_exp/slurm_logs/01b_check_val_bank_%j.out
#SBATCH --error=/private/groups/ioannidislab/UQ_decoupling_exp/slurm_logs/01b_check_val_bank_%j.err

eval "$(conda shell.bash hook)"
conda activate uq-iltm-bio
export LD_LIBRARY_PATH="/private/groups/ioannidislab/ajaras/conda_envs/uq-iltm-bio/lib:$LD_LIBRARY_PATH"
export PYTHONHASHSEED=0
export TMPDIR=/tmp            # node-local: multiprocessing / joblib temp dirs cannot be cleaned up on NFS
cd /private/home/ajaras/iLTM-bio/UQ_project
python experiment1_uq_heads/scripts/01b_check_val_bank.py "$@"
