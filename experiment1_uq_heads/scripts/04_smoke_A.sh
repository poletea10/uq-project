#!/bin/bash
# Smoke run of the A trainer (phase1_protocol.md §6), ONE GPU job, before launching the A jobs:
#   1. overfit one fixed batch (every head's training MSE must fall to <= 5% of its starting value);
#   2. stage 1 at real size, 200 steps, stopped after 100 and relaunched (the resume test);
#   3. stage 2, 50 steps, from stage 1's final.pt;  4. stage 3, 20 steps at N up to 60K, from stage 2's final.pt;
#   5. scripts/04b_check_smoke.py: final checkpoints never overwritten, stage chaining, resume continuity (LR,
#      batches, wandb id), peak GPU memory, projected wall time of a full A job.
# Stages 1-2 validate on the small smoke bank (fast); stage 3 runs one full validation on the REAL bank, to time it.
# Logs to the wandb project iLTMv2 as run smoke_A_<jobid>, group 'smoke', tag 'smoke' (filter it out of the real runs;
# wandb key: ~/.wandb_uq/netrc, see below).
#   sbatch experiment1_uq_heads/scripts/04_smoke_A.sh
# The job stops at the first failing part (non-zero exit), so `afterok` dependencies can hang on it.
#SBATCH --job-name=p1_04_smoke_A
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=34
#SBATCH --output=/private/groups/ioannidislab/UQ_decoupling_exp/slurm_logs/04_smoke_A_%j.out
#SBATCH --error=/private/groups/ioannidislab/UQ_decoupling_exp/slurm_logs/04_smoke_A_%j.err

eval "$(conda shell.bash hook)"
conda activate uq-iltm-bio
export LD_LIBRARY_PATH="/private/groups/ioannidislab/ajaras/conda_envs/uq-iltm-bio/lib:$LD_LIBRARY_PATH"
export PYTHONHASHSEED=0
export TMPDIR=/tmp            # node-local: multiprocessing / joblib temp dirs cannot be cleaned up on NFS
# wandb credentials: this user's own netrc (wandb reads $NETRC before ~/.netrc, which holds another account's key on
# this shared login). Create it once with:  mkdir -p ~/.wandb_uq && NETRC=~/.wandb_uq/netrc wandb login
WANDB_NETRC=${WANDB_NETRC:-$HOME/.wandb_uq/netrc}
if [ -f "$WANDB_NETRC" ]; then export NETRC="$WANDB_NETRC"; else echo "WARNING: $WANDB_NETRC not found: wandb uses ~/.netrc" >&2; fi
set -eo pipefail
cd /private/home/ajaras/iLTM-bio/UQ_project

ROOT=/private/groups/ioannidislab/UQ_decoupling_exp
RUN=smoke_A
SMOKE_BANK=$ROOT/val_bank_smoke
rm -rf "$ROOT/runs/$RUN"                          # a fresh smoke run every time
TRAIN="python -m iltm2.train.frozen --seed 0 --run_name $RUN --wandb_group smoke --wandb_tags smoke arm_A --wandb_id ${RUN}_${SLURM_JOB_ID:-local}"

echo "=== 1. overfit one fixed batch $(date)"
$TRAIN --overfit_steps 3000 --wandb_log False
echo "=== 2a. stage 1, stopped after 100 of 200 steps $(date)"
$TRAIN --stage 1 --max_steps 200 --stop_after 100 --mini_every 50 --val_bank $SMOKE_BANK
echo "=== 2b. stage 1, relaunched: resumes at step 100 $(date)"
$TRAIN --stage 1 --max_steps 200 --mini_every 50 --val_bank $SMOKE_BANK
echo "=== 3. stage 2, 50 steps $(date)"
$TRAIN --stage 2 --max_steps 50 --mini_every 25 --val_bank $SMOKE_BANK
echo "=== 4. stage 3, 20 steps; full validation on the real bank $(date)"
$TRAIN --stage 3 --max_steps 20 --mini_every 10
echo "=== 5. checks $(date)"
python experiment1_uq_heads/scripts/04b_check_smoke.py --run "$ROOT/runs/$RUN"
