#!/bin/bash
# One A job (design Training §5, protocol §9): the 3 A1 + 3 A2 head pairs of one seed, stages 1 -> 2 -> 3 run in
# sequence inside this one job (one queue wait instead of three). Each stage writes runs/A_seed<S>/stage<s>/final.pt
# once and never overwrites it; the next stage starts from it.
#   sbatch -J A_seed0 experiment1_uq_heads/scripts/train_A.sh 0
#   sbatch -J A_seed0 --dependency=afterok:<smoke job id> --kill-on-invalid-dep=yes \
#          experiment1_uq_heads/scripts/train_A.sh 0                    # start only if the smoke run passed
#   STAGES="2 3" sbatch -J A_seed0 experiment1_uq_heads/scripts/train_A.sh 0      # a subset of the stages
#   sbatch -J A_seed0 experiment1_uq_heads/scripts/train_A.sh 0 --clip_per pair    # extra trainer flags
# After a timeout or a crash, resubmit the same command: finished stages are skipped (their final.pt exists) and an
# unfinished stage resumes from its latest.pt (saved every 500 steps and after every validation).
# wandb: the key is read from ~/.wandb_uq/netrc (see below); never pass the key as an argument.
#SBATCH --job-name=p1_train_A
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --time=4-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=34
#SBATCH --output=/private/groups/ioannidislab/UQ_decoupling_exp/slurm_logs/train_%x_%j.out
#SBATCH --error=/private/groups/ioannidislab/UQ_decoupling_exp/slurm_logs/train_%x_%j.err

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

SEED=${1:?usage: train_A.sh <seed> [trainer flags]}
shift
for STAGE in ${STAGES:-1 2 3}; do
  echo "=== A_seed$SEED stage $STAGE $(date)"
  python -m iltm2.train.frozen --stage "$STAGE" --seed "$SEED" "$@"
done
echo "=== A_seed$SEED done $(date)"
