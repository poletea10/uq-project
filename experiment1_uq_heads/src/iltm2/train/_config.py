"""Command-line configuration of the Phase 1 trainers (phase1_protocol.md §1.3, §7.2; design Training §2, §6).

Argument names follow tabicl's training parser (`tabicl.train._train_config.build_parser`) wherever the meaning is
the same (`--max_steps`, `--weight_decay`, `--gradient_clipping`, `--scheduler`, `--warmup_proportion`,
`--cosine_num_cycles`, `--cosine_amplitude_decay`, `--cosine_lr_end`, `--muon`, `--wandb_*`), so the flags of
scripts/train_v2_reg_stage*.sh carry over, and the namespace can be handed to tabicl's own `get_scheduler`. tabicl's
parser itself is not reused: most of its arguments configure the full model, DDP and the prior, which the heads'
trainers do not have (the stage shapes of the observed-cut prior live in `iltm2.prior._dataset.STAGES`).

Per-stage defaults (design Training §2): steps 50,000 / 4,000 / 1,000 and LR divisor 1 / 8 / 40, applied to every
swept LR (heads and, in B, the fork). Every value can be overridden (the smoke run shortens the stages).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from iltm2.model._backbone import DEFAULT_BACKBONE
from iltm2.prior._bank import BANK_DIR
from iltm2.prior._dataset import BATCH_SIZE

ROOT = Path("/private/groups/ioannidislab/UQ_decoupling_exp")
RUNS_ROOT = ROOT / "runs"
STAGE_STEPS = {1: 50_000, 2: 4_000, 3: 1_000}
LR_DIVISOR = {1: 1.0, 2: 8.0, 3: 40.0}


def str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {v!r}")


def build_parser(arm: str = "A") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"iLTMv2 Phase 1 trainer, arm {arm}")
    g = p.add_argument_group("job")
    g.add_argument("--stage", type=int, choices=(1, 2, 3), default=1)
    g.add_argument("--seed", type=int, default=0, help="fixes the task stream and the heads' initialization")
    g.add_argument("--run_name", default=None, help="default: A_seed<seed>; the run directory and the wandb id")
    g.add_argument("--runs_root", default=str(RUNS_ROOT))
    g.add_argument("--device", default="cuda")

    g = p.add_argument_group("schedule and optimizer (design Training §2)")
    g.add_argument("--max_steps", type=int, default=None, help="steps of this stage (default 50000 / 4000 / 1000)")
    g.add_argument("--lr_divisor", type=float, default=None, help="max LR = swept LR / divisor (default 1 / 8 / 40)")
    g.add_argument("--head_lrs", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3], help="stage-1 head LRs")
    g.add_argument("--weight_decay", type=float, default=1e-5)
    g.add_argument("--gradient_clipping", type=float, default=5.0)
    g.add_argument("--clip_per", choices=("head", "pair"), default="head",
                   help="clip each head on its own gradient norm (default: the A and E heads stay fully "
                        "independent, design §6) or each head pair on their joint norm")
    g.add_argument("--scheduler", choices=("cosine_with_restarts",), default="cosine_with_restarts")
    g.add_argument("--warmup_proportion", type=float, default=0.01)
    g.add_argument("--warmup_steps", type=int, default=0, help="used only if --warmup_proportion < 0 (tabicl)")
    g.add_argument("--cosine_num_cycles", type=int, default=1)
    g.add_argument("--cosine_amplitude_decay", type=float, default=1.0)
    g.add_argument("--cosine_lr_end", type=float, default=1e-7)
    g.add_argument("--muon", type=str2bool, default=False, help="accepted for tabicl compatibility; must be False")
    if arm == "B":
        g.add_argument("--fork_lr", type=float, required=True, help="stage-1 max LR of the fork (swept)")
        g.add_argument("--k", type=int, default=2, help="forked TFicl blocks")
        g.add_argument("--fork_cosine_lr_end", type=float, default=1e-8,
                       help="end LR of the fork's cosine schedule, every stage (design §2: the fork's stage-3 max "
                            "LR goes down to 3e-6 / 40 = 7.5e-8, below the heads' 1e-7)")

    g = p.add_argument_group("loss (design Architecture §3)")
    g.add_argument("--loss_p", type=float, default=0.0, help="weight exponent p (0 = unweighted, the default)")
    g.add_argument("--loss_c", type=float, default=None, help="weight floor c (needed if p > 0)")
    g.add_argument("--lambda_E", type=float, default=1.0)
    g.add_argument("--init_bias", choices=("warmup", "bank"), default="warmup",
                   help="output bias at stage 1: mean target of the first batch (design §1) or the bank constant")

    g = p.add_argument_group("data (design Training §1-2)")
    g.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    g.add_argument("--R", type=int, default=16, help="replicates per query in training")
    g.add_argument("--num_workers", type=int, default=max(1, len(os.sched_getaffinity(0)) - 2))
    g.add_argument("--prefetch_factor", type=int, default=4)

    g = p.add_argument_group("backbone, validation, checkpoints (protocol §2, §7.4-7.5)")
    g.add_argument("--backbone_path", default=DEFAULT_BACKBONE)
    g.add_argument("--val_bank", default=str(BANK_DIR))
    g.add_argument("--val_every", type=int, default=5000, help="full-bank validation (and always at the stage end)")
    g.add_argument("--mini_every", type=int, default=250, help="mini-bank evaluation")
    g.add_argument("--save_temp_every", type=int, default=500, help="latest.pt (also after every validation)")
    g.add_argument("--log_every", type=int, default=1, help="training metrics to wandb every n steps")

    g = p.add_argument_group("wandb (protocol §7.1)")
    g.add_argument("--wandb_log", type=str2bool, default=True)
    g.add_argument("--wandb_project", default="iLTMv2")
    g.add_argument("--wandb_entity", default=None, help="user or team owning the project (default: your default entity)")
    g.add_argument("--wandb_group", default=None, help="run group (default: the arm, e.g. 'A'; the smoke run: 'smoke')")
    g.add_argument("--wandb_tags", nargs="*", default=None, help="run tags (default: arm, stage-independent)")
    g.add_argument("--wandb_name", default=None, help="default: the run name")
    g.add_argument("--wandb_id", default=None, help="default: the run name (one wandb run per job, all stages)")
    g.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default="online")
    g.add_argument("--wandb_dir", default=str(ROOT / "wandb"))

    g = p.add_argument_group("smoke run (protocol §6)")
    g.add_argument("--stop_after", type=int, default=None,
                   help="stop after this many steps of this launch, saving latest.pt as a killed job would")
    g.add_argument("--overfit_steps", type=int, default=0, help="> 0: run only the overfit-one-batch test")
    g.add_argument("--overfit_lr", type=float, default=1e-3)
    return p


def finalize(args: argparse.Namespace, arm: str = "A") -> argparse.Namespace:
    """Fill the per-stage defaults and check the combination."""
    assert not args.muon, "Muon is not used for the heads (design Training §2)"
    if args.max_steps is None:
        args.max_steps = STAGE_STEPS[args.stage]
    if args.lr_divisor is None:
        args.lr_divisor = LR_DIVISOR[args.stage]
    if args.run_name is None:
        args.run_name = f"{arm}_seed{args.seed}"
    args.wandb_name = args.wandb_name or args.run_name
    args.wandb_id = args.wandb_id or args.run_name
    args.wandb_group = args.wandb_group or arm
    args.wandb_tags = args.wandb_tags if args.wandb_tags is not None else [f"arm_{arm}"]
    if args.loss_p > 0:
        assert args.loss_c is not None and args.loss_c > 0, "--loss_p > 0 needs --loss_c > 0"
    min_lr = min(args.head_lrs) / args.lr_divisor
    assert args.cosine_lr_end < min_lr, f"--cosine_lr_end {args.cosine_lr_end} must be below every max LR ({min_lr:g})"
    if arm == "B":
        fork_lr = args.fork_lr / args.lr_divisor
        assert args.fork_cosine_lr_end < fork_lr, (f"--fork_cosine_lr_end {args.fork_cosine_lr_end} must be below the "
                                                   f"fork's max LR ({fork_lr:g})")
    return args
