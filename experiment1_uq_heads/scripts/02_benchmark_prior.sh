#!/bin/bash
# Step-4 gate (phase1_protocol.md §3): training-stream throughput per stage vs the frozen backbone's forward time.
#   sbatch experiment1_uq_heads/scripts/02_benchmark_prior.sh
#   sbatch experiment1_uq_heads/scripts/02_benchmark_prior.sh --stages 3 --n-steps 6                  # stage-3 rerun
# Gate: prior s/step <= 0.8 x backbone s/step per stage (--margin); out-of-order loader (_dataset.make_dataloader).
# Asks for 34 CPUs: up to 32 DataLoader workers + the main process (a GPU node has 256 CPUs / 8 GPUs).
#SBATCH --job-name=p1_02_benchmark_prior
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=34
#SBATCH --output=/private/groups/ioannidislab/UQ_decoupling_exp/slurm_logs/02_benchmark_prior_%j.out
#SBATCH --error=/private/groups/ioannidislab/UQ_decoupling_exp/slurm_logs/02_benchmark_prior_%j.err

eval "$(conda shell.bash hook)"
conda activate uq-iltm-bio
export LD_LIBRARY_PATH="/private/groups/ioannidislab/ajaras/conda_envs/uq-iltm-bio/lib:$LD_LIBRARY_PATH"
export PYTHONHASHSEED=0
export TMPDIR=/tmp            # node-local: multiprocessing / joblib temp dirs cannot be cleaned up on NFS
cd /private/home/ajaras/iLTM-bio/UQ_project
python experiment1_uq_heads/scripts/02_benchmark_prior.py "$@"
