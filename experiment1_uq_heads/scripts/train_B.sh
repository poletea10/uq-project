#!/bin/bash
# One B job (design Training §4-5, protocol §9): one configuration (head LR x fork LR) of arm B on one seed, stages
# 1 -> 2 -> 3 run in sequence inside this one job. Each stage writes runs/B_h<H>_f<F>_seed<S>/stage<s>/final.pt once and
# never overwrites it; the next stage starts from it. No separate smoke job (user's decision): at a fresh stage 1 the
# trainer first runs one fork forward/backward at the largest stage-3 shape (peak GPU memory) and checks that the
# fork reproduces h_star on the first batch; both print an OK line to the .out, and a failure stops the job.
#   Round 1, the 9 configurations on seed 0 (paired with A_seed0's task stream):
#   for H in 1e-4 3e-4 1e-3; do for F in 3e-6 1e-5 3e-5; do
#     sbatch -J B_h${H}_f${F}_s0 experiment1_uq_heads/scripts/train_B.sh $H $F 0
#   done; done
#   STAGES="2 3" sbatch -J ... experiment1_uq_heads/scripts/train_B.sh 3e-4 1e-5 0           # a subset of the stages
# After a timeout or a crash, resubmit the same command: finished stages are skipped (their final.pt exists) and an
# unfinished stage resumes from its latest.pt.
#SBATCH --job-name=p1_train_B
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
# wandb credentials: this user's own netrc (see scripts/train_A.sh)
WANDB_NETRC=${WANDB_NETRC:-$HOME/.wandb_uq/netrc}
if [ -f "$WANDB_NETRC" ]; then export NETRC="$WANDB_NETRC"; else echo "WARNING: $WANDB_NETRC not found: wandb uses ~/.netrc" >&2; fi
set -eo pipefail
cd /private/home/ajaras/iLTM-bio/UQ_project

HEAD_LR=${1:?usage: train_B.sh <head_lr> <fork_lr> <seed> [trainer flags]}
FORK_LR=${2:?usage: train_B.sh <head_lr> <fork_lr> <seed> [trainer flags]}
SEED=${3:?usage: train_B.sh <head_lr> <fork_lr> <seed> [trainer flags]}
shift 3
for STAGE in ${STAGES:-1 2 3}; do
  echo "=== B head_lr $HEAD_LR fork_lr $FORK_LR seed $SEED: stage $STAGE $(date)"
  python -m iltm2.train.fork --stage "$STAGE" --seed "$SEED" --head_lrs "$HEAD_LR" --fork_lr "$FORK_LR" "$@"
done
echo "=== B head_lr $HEAD_LR fork_lr $FORK_LR seed $SEED: done $(date)"
