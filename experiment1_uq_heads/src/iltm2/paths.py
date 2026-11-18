"""Where Phase 1 reads and writes on the group disk (phase1_protocol.md §2; iLTMv2_transfer_guide.md §4, §5 Step 1).

    ROOT         the current round's outputs: runs, validation bank, wandb, audits, SLURM logs. Every write goes here.
                 experiment1_uq_heads/outputs is a symlink to it.
    LEGACY_ROOT  the first training round (archived repository UQ_project_v1, commit c1ed503d). Read only: its
                 selected checkpoints are loaded by `iltm2.inference.LEGACY_BEST` (transfer guide §6).
"""
from pathlib import Path

ROOT = Path("/private/groups/ioannidislab/UQ_decoupling_exp_new")
LEGACY_ROOT = Path("/private/groups/ioannidislab/UQ_decoupling_exp")
RUNS_ROOT = ROOT / "runs"
LOG_DIR = ROOT / "slurm_logs"
