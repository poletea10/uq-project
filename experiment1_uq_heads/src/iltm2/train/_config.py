"""Command-line configuration of the Phase 1 trainers (phase1_protocol.md §1.3, §7.2; design Training §2, §6;
transfer guide §3.1-3.6, §3.10, §4).

Argument names follow tabicl's training parser (`tabicl.train._train_config.build_parser`) wherever the meaning is
the same (`--max_steps`, `--weight_decay`, `--gradient_clipping`, `--scheduler`, `--warmup_proportion`,
`--cosine_num_cycles`, `--cosine_amplitude_decay`, `--cosine_lr_end`, `--muon`, `--wandb_*`), so the flags of
scripts/train_v2_reg_stage*.sh carry over, and the namespace can be handed to tabicl's own `get_scheduler`. tabicl's
parser itself is not reused: most of its arguments configure the full model, DDP and the prior, which the heads'
trainers do not have (the stage shapes of the observed-cut prior live in `iltm2.prior._dataset.STAGES`).

Per-stage defaults (transfer guide §3.1-3.2):
    steps          30,000 / 8,000 / 1,200 (40,000 global steps)
    LR divisor     1 / 8 / 40, applied to every swept LR (heads and, in B, the fork)
    full bank      every 5,000 / 4,000 / 600 stage steps and at each stage end: global steps 5,000 ... 30,000, then
                   34,000 and 38,000, then 38,600 and 39,200 (ten validations per run)
Every value can be overridden (the smoke run shortens the stages).

Sweep (transfer guide §3.5, §3.10, §4): head LR {3e-4, 1e-3, 3e-3}; B's fork LR {1e-5, 3e-5, 1e-4}; loss exponent
p in {0, 1}; prior "carved" (the sweep) or "iid" (the winning configurations' ablation). An A job trains every head
LR x p pair on one backbone stream (`--head_lrs`, `--p` lists); a B job trains one (head LR, fork LR, p).
Run names: A_<prior>_seed<S> and B_h<H>_f<F>_p<P>_<prior>_seed<S>.
"""
from __future__ import annotations

import argparse
import os

from iltm2.model._backbone import DEFAULT_BACKBONE
from iltm2.paths import ROOT, RUNS_ROOT
from iltm2.prior._bank import BANK_DIR
from iltm2.prior._dataset import BATCH_SIZE, PRIORS

STAGE_STEPS = {1: 30_000, 2: 8_000, 3: 1_200}
LR_DIVISOR = {1: 1.0, 2: 8.0, 3: 40.0}
VAL_EVERY = {1: 5_000, 2: 4_000, 3: 600}
HEAD_LRS = (3e-4, 1e-3, 3e-3)
FORK_LRS = (1e-5, 3e-5, 1e-4)
LOSS_PS = (0.0, 1.0)


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
    g.add_argument("--prior", choices=PRIORS, default="carved",
                   help="carved (the design, every sweep run) or iid (the carving ablation of the winning configs)")
    g.add_argument("--run_name", default=None,
                   help="default: A_<prior>_seed<seed> / B_h<head_lr>_f<fork_lr>_p<p>_<prior>_seed<seed>; the run "
                        "directory and the wandb id")
    g.add_argument("--runs_root", default=str(RUNS_ROOT))
    g.add_argument("--device", default="cuda")

    g = p.add_argument_group("schedule and optimizer (design Training §2)")
    g.add_argument("--max_steps", type=int, default=None, help="steps of this stage (default 30000 / 8000 / 1200)")
    g.add_argument("--lr_divisor", type=float, default=None, help="max LR = swept LR / divisor (default 1 / 8 / 40)")
    g.add_argument("--head_lrs", type=float, nargs="+", default=list(HEAD_LRS),
                   help="stage-1 head LRs (arm B: exactly one)")
    g.add_argument("--weight_decay", type=float, default=1e-5)
    g.add_argument("--gradient_clipping", type=float, default=5.0)
    if arm == "A":
        g.add_argument("--clip_per", choices=("head", "pair"), default="head",
                       help="clip each head on its own gradient norm (default: the A and E heads stay fully "
                            "independent, design §6) or each head pair on their joint norm")
    else:
        g.add_argument("--clip_per", choices=("joint", "separate"), default="joint",
                       help="clip fork + both heads on their joint gradient norm (default, design §6: in B the heads "
                            "share the fork) or the fork and each head separately")
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
                            "LR goes down to 1e-5 / 40 = 2.5e-7, close to the heads' 1e-7)")
        g.add_argument("--recompute_above", type=int, default=20_000,
                       help="gradient checkpointing in the fork for tasks of more rows (design Training §2)")
        g.add_argument("--preflight", type=str2bool, default=True,
                       help="at a fresh stage 1: check fork output = h_star on the first batch, and run one fork "
                            "forward/backward on the largest stage-3 task shape (N = 60,000) to measure peak memory")

    g = p.add_argument_group("loss (design Architecture §3; transfer guide §3.6)")
    if arm == "A":
        g.add_argument("--p", type=float, nargs="+", default=list(LOSS_PS),
                       help="loss exponents: the job trains every head LR x p pair (default 0 1)")
    else:
        g.add_argument("--p", type=float, required=True, help="loss exponent of this B run (0 or 1)")
    g.add_argument("--loss_c", type=float, default=None,
                   help="weight floor c of w = (e(q) + c)^-p (default: the validation bank's manifest['loss_c'], "
                        "the fixed 0.1; the weighted validation metrics always use the bank's value)")
    g.add_argument("--lambda_E", type=float, default=1.0)
    g.add_argument("--init_bias", choices=("warmup", "bank"), default="warmup",
                   help="output bias at stage 1: mean target of the first batch (design §1) or the bank constant")

    g = p.add_argument_group("data (design Training §1-2; transfer guide §3.8-3.9)")
    g.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    g.add_argument("--R", type=int, default=16, help="replicates per query in training")
    g.add_argument("--subsample", type=str2bool, default=None,
                   help="the subsampled-context pass (default: on in stage 1, off in stages 2-3)")
    g.add_argument("--num_workers", type=int, default=max(1, len(os.sched_getaffinity(0)) - 2))
    g.add_argument("--prefetch_factor", type=int, default=4)

    g = p.add_argument_group("backbone, validation, checkpoints (protocol §2, §7.4-7.5)")
    g.add_argument("--backbone_path", default=DEFAULT_BACKBONE)
    g.add_argument("--val_bank", default=str(BANK_DIR))
    g.add_argument("--val_every", type=int, default=None,
                   help="full-bank validation every n stage steps, and always at the stage end (default 5000 / 4000 "
                        "/ 600)")
    g.add_argument("--mini_every", type=int, default=250, help="mini-bank evaluation")
    g.add_argument("--save_temp_every", type=int, default=500, help="latest.pt (also after every validation)")
    g.add_argument("--log_every", type=int, default=1, help="training metrics to wandb every n steps")

    g = p.add_argument_group("wandb (protocol §7.1)")
    g.add_argument("--wandb_log", type=str2bool, default=True)
    g.add_argument("--wandb_project", default="TabUD", help="wandb project of the current round (the first round logged to iLTMv2)")
    g.add_argument("--wandb_entity", default=None, help="user or team owning the project (default: your default entity)")
    g.add_argument("--wandb_group", default=None, help="run group (default: the arm, e.g. 'A'; the smoke run: 'smoke')")
    g.add_argument("--wandb_tags", nargs="*", default=None, help="run tags (default: arm, prior and, in B, the config)")
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
    from iltm2.model._heads import fmt_lr, fmt_p
    assert not args.muon, "Muon is not used for the heads (design Training §2)"
    if args.max_steps is None:
        args.max_steps = STAGE_STEPS[args.stage]
    if args.lr_divisor is None:
        args.lr_divisor = LR_DIVISOR[args.stage]
    if args.val_every is None:
        args.val_every = VAL_EVERY[args.stage]
    if args.subsample is None:
        args.subsample = args.stage == 1
    ps = args.p if arm == "A" else [args.p]
    assert all(p >= 0 for p in ps), "loss exponents must be >= 0"
    if arm == "B":
        assert len(args.head_lrs) == 1, "arm B trains one head LR per job (--head_lrs <one value>)"
        assert args.overfit_steps == 0, "the overfit smoke test is only implemented for arm A"
    if args.run_name is None:
        args.run_name = (f"A_{args.prior}_seed{args.seed}" if arm == "A" else
                         f"B_h{fmt_lr(args.head_lrs[0])}_f{fmt_lr(args.fork_lr)}_p{fmt_p(args.p)}_{args.prior}"
                         f"_seed{args.seed}")
    args.wandb_name = args.wandb_name or args.run_name
    args.wandb_id = args.wandb_id or args.run_name
    args.wandb_group = args.wandb_group or arm
    if args.wandb_tags is None:
        args.wandb_tags = [f"arm_{arm}", f"prior_{args.prior}"]
        if arm == "B":
            args.wandb_tags += [f"head_lr_{fmt_lr(args.head_lrs[0])}", f"fork_lr_{fmt_lr(args.fork_lr)}",
                                f"p_{fmt_p(args.p)}"]
    if args.loss_c is not None:
        assert args.loss_c > 0, "--loss_c must be > 0"
    min_lr = min(args.head_lrs) / args.lr_divisor
    assert args.cosine_lr_end < min_lr, f"--cosine_lr_end {args.cosine_lr_end} must be below every max LR ({min_lr:g})"
    if arm == "B":
        fork_lr = args.fork_lr / args.lr_divisor
        assert args.fork_cosine_lr_end < fork_lr, (f"--fork_cosine_lr_end {args.fork_cosine_lr_end} must be below the "
                                                   f"fork's max LR ({fork_lr:g})")
    return args
