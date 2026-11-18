#!/bin/bash
# One A job (transfer guide §4): every head pair of arm A on one seed and one prior, head LR {3e-4, 1e-3, 3e-3} x
# loss exponent p {0, 1} = 6 pairs A_lr<LR>_p<P> on one frozen backbone stream. Stages 1 -> 2 -> 3 run in sequence
# inside this one job (one queue wait instead of three); each stage writes runs/A_<prior>_seed<S>/stage<s>/final.pt
# once and never overwrites it, and the next stage starts from it.
#   sbatch -J A_carved_seed0 experiment1_uq_heads/scripts/train_A.sh carved 0          # the sweep's A job
#   sbatch -J A_carved_seed0 --dependency=afterok:<smoke job id> --kill-on-invalid-dep=yes \
#          experiment1_uq_heads/scripts/train_A.sh carved 0                            # start only if the smoke passed
#   sbatch -J A_iid_seed0 experiment1_uq_heads/scripts/train_A.sh iid 0 --head_lrs 1e-3 --p 1   # an ablation job
#   STAGES="2 3" sbatch -J A_carved_seed0 experiment1_uq_heads/scripts/train_A.sh carved 0     # a subset of the stages
# After a timeout or a crash, resubmit the same command: finished stages are skipped (their final.pt exists) and an
# unfinished stage resumes from its latest.pt (saved every 500 steps and after every validation).
# wandb: the key is read from ~/.wandb_uq/netrc (see below); never pass the key as an argument.
#SBATCH --job-name=p1_train_A
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --time=3-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=34
#SBATCH --output=/private/groups/ioannidislab/UQ_decoupling_exp_new/slurm_logs/train_%x_%j.out
#SBATCH --error=/private/groups/ioannidislab/UQ_decoupling_exp_new/slurm_logs/train_%x_%j.err

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

USAGE="usage: train_A.sh <carved|iid> <seed> [trainer flags]"
PRIOR=${1:?$USAGE}
SEED=${2:?$USAGE}
shift 2
for STAGE in ${STAGES:-1 2 3}; do
  echo "=== A_${PRIOR}_seed$SEED stage $STAGE $(date)"
  python -m iltm2.train.frozen --stage "$STAGE" --seed "$SEED" --prior "$PRIOR" "$@"
done
echo "=== A_${PRIOR}_seed$SEED done $(date)"
