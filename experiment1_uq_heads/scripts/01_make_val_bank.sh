#!/bin/bash
# Step 6 (phase1_protocol.md §3): build the 2,000-task validation bank, the mini-bank and the reference lines.
#   sbatch experiment1_uq_heads/scripts/01_make_val_bank.sh                         # the bank (resumable: relaunch the
#                                                                                   # same command after a timeout)
#   sbatch experiment1_uq_heads/scripts/01_make_val_bank.sh --n-tasks 8 4 2 \
#       --out /private/groups/ioannidislab/UQ_decoupling_exp/val_bank_smoke         # smoke build first
# CPU workers generate the tasks (R_val = 128 replicate passes each), the GPU computes the forecast targets.
# A5500 = the 24 GB card training will run on: the bank's targets must come from the training GPU's forward.
#SBATCH --job-name=p1_01_make_val_bank
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=34
#SBATCH --output=/private/groups/ioannidislab/UQ_decoupling_exp/slurm_logs/01_make_val_bank_%j.out
#SBATCH --error=/private/groups/ioannidislab/UQ_decoupling_exp/slurm_logs/01_make_val_bank_%j.err

eval "$(conda shell.bash hook)"
conda activate uq-iltm-bio
export LD_LIBRARY_PATH="/private/groups/ioannidislab/ajaras/conda_envs/uq-iltm-bio/lib:$LD_LIBRARY_PATH"
export PYTHONHASHSEED=0
export TMPDIR=/tmp            # node-local: multiprocessing / joblib temp dirs cannot be cleaned up on NFS
cd /private/home/ajaras/iLTM-bio/UQ_project
python experiment1_uq_heads/scripts/01_make_val_bank.py "$@"
