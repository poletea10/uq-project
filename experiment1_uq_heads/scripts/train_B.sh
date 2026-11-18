#!/bin/bash
# One B job (transfer guide §4): one configuration (head LR x fork LR x loss exponent p) of arm B on one seed and one
# prior, stages 1 -> 2 -> 3 run in sequence inside this one job. Each stage writes
# runs/B_h<H>_f<F>_p<P>_<prior>_seed<S>/stage<s>/final.pt once and never overwrites it; the next stage starts from it.
# No separate smoke job: at a fresh stage 1 the trainer first runs one fork forward/backward at the largest stage-3
# shape (peak GPU memory) and checks that the fork reproduces h_star on the first batch; both print an OK line to the
# .out, and a failure stops the job.
#   The sweep, 18 configurations on seed 0 (paired with A_carved_seed0's task stream):
#   for H in 3e-4 1e-3 3e-3; do for F in 1e-5 3e-5 1e-4; do for P in 0 1; do
#     sbatch -J B_h${H}_f${F}_p${P}_carved_seed0 experiment1_uq_heads/scripts/train_B.sh $H $F $P carved 0
#   done; done; done
#   STAGES="2 3" sbatch -J ... experiment1_uq_heads/scripts/train_B.sh 1e-3 3e-5 1 carved 0     # a subset of the stages
# After a timeout or a crash, resubmit the same command: finished stages are skipped (their final.pt exists) and an
# unfinished stage resumes from its latest.pt.
#SBATCH --job-name=p1_train_B
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --time=4-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=34
#SBATCH --output=/private/groups/ioannidislab/UQ_decoupling_exp_new/slurm_logs/train_%x_%j.out
#SBATCH --error=/private/groups/ioannidislab/UQ_decoupling_exp_new/slurm_logs/train_%x_%j.err

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

USAGE="usage: train_B.sh <head_lr> <fork_lr> <p> <carved|iid> <seed> [trainer flags]"
HEAD_LR=${1:?$USAGE}
FORK_LR=${2:?$USAGE}
P=${3:?$USAGE}
PRIOR=${4:?$USAGE}
SEED=${5:?$USAGE}
shift 5
NAME="B head_lr $HEAD_LR fork_lr $FORK_LR p $P prior $PRIOR seed $SEED"
for STAGE in ${STAGES:-1 2 3}; do
  echo "=== $NAME: stage $STAGE $(date)"
  python -m iltm2.train.fork --stage "$STAGE" --seed "$SEED" --head_lrs "$HEAD_LR" --fork_lr "$FORK_LR" --p "$P" \
    --prior "$PRIOR" "$@"
done
echo "=== $NAME: done $(date)"
