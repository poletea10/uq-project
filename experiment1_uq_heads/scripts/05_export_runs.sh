#!/bin/bash
# Export the Phase 1 runs' relevant metrics as small CSVs + a data dictionary (phase1_protocol.md §9.3), from the wandb
# API and the run folders, to /private/groups/ioannidislab/UQ_decoupling_exp_new/audits/export/. CPU only.
#   sbatch experiment1_uq_heads/scripts/05_export_runs.sh
#SBATCH --job-name=p1_05_export_runs
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=/private/groups/ioannidislab/UQ_decoupling_exp_new/slurm_logs/05_export_runs_%j.out
#SBATCH --error=/private/groups/ioannidislab/UQ_decoupling_exp_new/slurm_logs/05_export_runs_%j.err

eval "$(conda shell.bash hook)"
conda activate uq-iltm-bio
export LD_LIBRARY_PATH="/private/groups/ioannidislab/ajaras/conda_envs/uq-iltm-bio/lib:$LD_LIBRARY_PATH"
export PYTHONHASHSEED=0
export TMPDIR=/tmp
# wandb credentials: this user's own netrc (see scripts/train_A.sh)
WANDB_NETRC=${WANDB_NETRC:-$HOME/.wandb_uq/netrc}
if [ -f "$WANDB_NETRC" ]; then export NETRC="$WANDB_NETRC"; else echo "WARNING: $WANDB_NETRC not found: wandb uses ~/.netrc" >&2; fi
cd /private/home/ajaras/iLTM-bio/UQ_project
python experiment1_uq_heads/scripts/05_export_runs.py "$@"
