#!/bin/bash
# Backbone check: TabICLv2 on the toy set. Criterion 1 needs the hgp toy outputs (01_toy.sh --method hgp);
# fits are cached, so rerunning after them only re-grades.
#   sbatch experiment0_baseline/scripts/00a_backbone_check.sh
#SBATCH --job-name=p0_00a_backbone_check
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/private/home/ajaras/iLTM-bio/UQ_project/experiment0_baseline/outputs/logs/00a_backbone_check_%j.out
#SBATCH --error=/private/home/ajaras/iLTM-bio/UQ_project/experiment0_baseline/outputs/logs/00a_backbone_check_%j.err

eval "$(conda shell.bash hook)"
conda activate uq-iltm-bio
export LD_LIBRARY_PATH="/private/groups/ioannidislab/ajaras/conda_envs/uq-iltm-bio/lib:$LD_LIBRARY_PATH"
export PYTHONHASHSEED=0
cd /private/home/ajaras/iLTM-bio/UQ_project
python experiment0_baseline/scripts/00a_backbone_check.py "$@"
