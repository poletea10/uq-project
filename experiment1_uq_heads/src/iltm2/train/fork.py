"""The B-job trainer (phase1_protocol.md §3 step 9; design Architecture §4 "Arm B", Training §2-5): one head pair on a
trainable fork of the last k TFicl blocks, one configuration (head LR x fork LR) per job.

    python -m iltm2.train.fork --stage {1,2,3} --seed S --head_lrs H --fork_lr F [options of iltm2.train._config]

THE MODEL
    The backbone stays frozen and runs under no_grad; it returns the full residual stream after block 12 - k
    (`fork_input`, ALL rows: the query attends to the context in every block). The `Fork` (trainable copies of
    blocks 13 - k .. 12 and the final LN, initialized with the pretrained weights) maps it to a new h at the query
    rows, which feeds both heads of one `HeadPair` (arm "B"). The originals keep feeding the decoder, so the PPD,
    e(q) and the targets are exactly those of the A jobs. At step 0 the fork's output equals h_star: B starts where
    A1 starts (same trunk initialization for the same seed).

ONE STEP: as in the A trainer (`frozen.py`), per micro-batch group: backbone forward (no grad), targets, fork forward
(gradient checkpointing for tasks above --recompute_above rows), heads, loss x the group's share of the step's query
rows, backward. Then gradient norms (A head, E head, fork), clipping (--clip_per joint: fork + both heads on their
joint norm, design §6; or separate), and two AdamW optimizers with their own schedules: the heads at head_lr / divisor
ending at --cosine_lr_end (1e-7), the fork at fork_lr / divisor ending at --fork_cosine_lr_end (1e-8, design §2).

CHECKS AT THE START OF A FRESH STAGE 1 (instead of a separate B smoke job, user's decision; protocol §6)
    preflight          one fork forward/backward on the largest stage-3 task shape (N = 60,000 rows), so a fork that
                       does not fit on the GPU fails in minutes, not after two stages; peak memory is printed
    check_first_batch  the fork's output equals h_star on the first training batch (|diff| <= 1e-5)
    Each prints an "OK" line to the job's .out; a failure stops the job (non-zero exit).

LOGGING AND CHECKPOINTS: the pair is named "B", so every B run logs the same keys (B/train/..., B/val/..., B/mini/...)
and the 9 configurations are told apart by their config and tags (head_lr_*, fork_lr_*). A best-A or best-E
checkpoint stores the whole model (fork + both heads): the heads share the fork, so a configuration is selected as a
whole (design Training §3), and both heads are logged at every validation, so the trade-off at each best checkpoint
can be read off the curves.
"""
from __future__ import annotations

import copy
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
from tabicl.prior._dataset import Prior

from iltm2.model._backbone import FrozenBackbone
from iltm2.model._heads import HEADS, Fork, HeadPair
from iltm2.prior._cut_graph_scm import CutGraphConfig
from iltm2.prior._dataset import STAGE_KEY_OFFSET, STAGES, UQBatch, sample_uq_task, stage_step_key
from iltm2.prior._filter import RejectionLog
from iltm2.train._base import BaseTrainer, get_scheduler
from iltm2.train._config import build_parser, finalize
from iltm2.train._loss import Targets, batch_targets, group_weight, uq_loss
from iltm2.train._metrics import EMA, train_metrics
from iltm2.train._validate import BankEvaluator, BestTracker
from iltm2.train.frozen import TARGET_KEYS, grad_norm

PAIR = "B"                   # the pair's fixed id: the wandb key prefix of every B run
FORK_TOL = 1e-5              # step-0 check: max |fork output - h_star|


class ForkTrainer(BaseTrainer):
    arm = "B"

    # ----- build

    def build(self):
        a, dev = self.args, self.device
        self.backbone = FrozenBackbone(a.backbone_path, a.device, tap_layers=None, fork_k=a.k)
        self.fork = Fork(self.backbone, k=a.k).to(dev)
        self.pair = HeadPair("B", a.head_lrs[0], a.seed, name=PAIR).to(dev)
        self.opt_heads = self.pair.make_optimizer(lr=self.pair.lr / a.lr_divisor, weight_decay=a.weight_decay)
        self.opt_fork = torch.optim.AdamW(self.fork.parameters(), lr=a.fork_lr / a.lr_divisor,
                                          weight_decay=a.weight_decay, betas=(0.9, 0.999))
        self.sched_heads = get_scheduler(a, self.opt_heads)            # ends at --cosine_lr_end
        fork_cfg = copy.copy(a)
        fork_cfg.cosine_lr_end = a.fork_cosine_lr_end
        self.sched_fork = get_scheduler(fork_cfg, self.opt_fork)        # ends at --fork_cosine_lr_end
        self.ema = {k: EMA(0.99) for k in ("xmse_A", "xmse_E")}
        n_fork = sum(p.numel() for p in self.fork.parameters())
        print(f"{self.backbone!r}\nfork: last {a.k} blocks + LN ({n_fork / 1e6:.1f}M parameters), max LR "
              f"{a.fork_lr / a.lr_divisor:g}; heads: max LR {self.pair.lr / a.lr_divisor:g}; clipping {a.clip_per}",
              flush=True)

    def build_evaluators(self) -> Tuple[BankEvaluator, BankEvaluator]:
        a = self.args
        full = BankEvaluator(a.val_bank, self.backbone)                     # fork input recomputed per task
        mini = BankEvaluator(a.val_bank, self.backbone, mini=True, cache=("fork",))
        return full, mini

    # ----- state

    def heads_state(self) -> dict:
        return dict(pair=self.pair.state_dict(), fork=self.fork.state_dict())

    def load_heads_state(self, d: dict):
        self.pair.load_state_dict(d["pair"])
        self.fork.load_state_dict(d["fork"])

    def optim_state(self) -> dict:
        return dict(opt_heads=self.opt_heads.state_dict(), opt_fork=self.opt_fork.state_dict(),
                    sched_heads=self.sched_heads.state_dict(), sched_fork=self.sched_fork.state_dict())

    def load_optim_state(self, d: dict):
        self.opt_heads.load_state_dict(d["opt_heads"])
        self.opt_fork.load_state_dict(d["opt_fork"])
        self.sched_heads.load_state_dict(d["sched_heads"])
        self.sched_fork.load_state_dict(d["sched_fork"])

    def extra_state(self) -> dict:
        return dict(ema={k: e.state_dict() for k, e in self.ema.items()})

    def load_extra_state(self, d: dict):
        for k, s in d["ema"].items():
            self.ema[k].load_state_dict(s)

    def best_payload(self, head_id: str, h: str) -> dict:
        a = self.args
        return dict(pair=self.pair.state_dict(), fork=self.fork.state_dict(), head_id=head_id, which=h, arm="B",
                    head_lr=a.head_lrs[0], fork_lr=a.fork_lr, k=a.k, seed=a.seed)

    def lrs(self) -> Dict[str, float]:
        return dict(heads=self.opt_heads.param_groups[0]["lr"], fork=self.opt_fork.param_groups[0]["lr"])

    # ----- initialization

    def init_output_bias(self, mean_t_A: float, mean_t_E: float):
        self.pair.init_output_bias(mean_t_A, mean_t_E)
        print(f"output biases set to the mean targets: t_A {mean_t_A:.5g}, t_E {mean_t_E:.5g}", flush=True)

    @torch.no_grad()
    def warmup_means(self, groups: List[UQBatch]) -> Tuple[float, float]:
        sA = sE = n = 0.0
        for g in groups:
            g = g.to(self.device, non_blocking=True)
            T = batch_targets(self.backbone(g.X, g.y_ctx).q, g.y_reps)
            sA, sE, n = sA + float(T.t_A.sum()), sE + float(T.t_E.sum()), n + T.t_A.numel()
        return sA / n, sE / n

    # ----- startup checks (fresh stage 1)

    def preflight(self):
        a, dev = self.args, self.device
        if not a.preflight:
            print("preflight skipped (--preflight False)", flush=True)
            return
        if dev.type != "cuda":
            print("preflight memory probe skipped (no GPU)", flush=True)
            return
        cfg = CutGraphConfig()
        st = STAGES[3]
        N = int(st["max_seq_len"])
        n_ctx = int(st["max_train_size"] * N)                 # the largest context TabICLv2's sampler gives
        d = int(Prior.adjust_max_features(N, cfg.max_features))
        t0 = time.time()
        task, _ = sample_uq_task(a.seed, stage_step_key(3, STAGE_KEY_OFFSET - 1), 0, 3, cfg, R=a.R,
                                 N=N, n_ctx=n_ctx, d=d)        # a key no training step uses
        g = UQBatch.stack([task], stage=3, step=-1, group=0, task_idx=[0], log=RejectionLog(),
                          gen_time=0.0).to(dev)
        t_gen = time.time() - t0
        recompute = g.seq_len > a.recompute_above
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            out = self.backbone(g.X, g.y_ctx)
            T = batch_targets(out.q, g.y_reps)
            A, E = self.pair(self.fork(out.fork_input, out.train_size, recompute=recompute))
            uq_loss(A, E, T.t_A, T.t_E, out.e_q).loss.backward()
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            print(f"FAIL preflight: out of GPU memory on one fork forward/backward at N {N} (n_ctx {n_ctx}, "
                  f"gradient checkpointing {'on' if recompute else 'off'})", flush=True)
            raise
        finally:
            self.opt_heads.zero_grad(set_to_none=True)
            self.opt_fork.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        total = torch.cuda.get_device_properties(dev).total_memory / 2 ** 30
        print(f"OK   preflight: one fork forward/backward on the largest stage-3 task (N {N}, n_ctx {n_ctx}, d "
              f"{int(g.d[0])}, m {g.m}; gradient checkpointing {'on' if recompute else 'off'}) in "
              f"{time.time() - t0:.1f}s (task built in {t_gen:.0f}s): peak GPU memory {peak:.2f} of {total:.1f} GiB",
              flush=True)
        del g, out, T, A, E, task
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()             # the stage summary measures training only

    @torch.no_grad()
    def check_first_batch(self, groups: List[UQBatch]):
        g = groups[0].to(self.device, non_blocking=True)
        out = self.backbone(g.X, g.y_ctx)
        h = self.fork(out.fork_input, out.train_size)
        diff = float((h.double() - out.h_star.double()).abs().max())
        ok = diff <= FORK_TOL
        print(f"{'OK  ' if ok else 'FAIL'} fork output = h_star at step 0 (first batch, group 0: {g.B} tasks x "
              f"{g.m} queries): max |diff| {diff:.1e} (tolerance {FORK_TOL:g})", flush=True)
        if not ok:
            raise RuntimeError("the fork does not reproduce h_star at step 0: B would not start where A1 starts")

    # ----- one step

    def train_step(self, groups: List[UQBatch]) -> Dict[str, float]:
        a, dev = self.args, self.device
        n_step = sum(g.y_reps.shape[0] * g.y_reps.shape[1] for g in groups)
        self.opt_heads.zero_grad(set_to_none=True)
        self.opt_fork.zero_grad(set_to_none=True)
        preds = {h: [] for h in HEADS}
        losses = torch.zeros(3, device=dev)
        cols = {k: [] for k in ("t_A", "t_E", "t_A_a", "t_A_b", "t_E_a", "t_E_b")}
        eqs = []
        for g in groups:
            g = g.to(dev, non_blocking=True)
            out = self.backbone(g.X, g.y_ctx)
            T = batch_targets(out.q, g.y_reps)
            w = group_weight(g.y_reps.shape[0] * g.y_reps.shape[1], n_step)
            h = self.fork(out.fork_input, out.train_size, recompute=g.seq_len > a.recompute_above)
            A, E = self.pair(h)
            lo = uq_loss(A, E, T.t_A, T.t_E, out.e_q, p=a.loss_p, c=a.loss_c, lambda_E=a.lambda_E)
            (lo.loss * w).backward()
            losses += w * torch.stack([lo.loss, lo.loss_A, lo.loss_E]).detach()
            preds["A"].append(A.detach().flatten())
            preds["E"].append(E.detach().flatten())
            for k in cols:
                cols[k].append(getattr(T, k).flatten())
            eqs.append(out.e_q.flatten())
            del out, h, A, E, lo
        T_all = Targets(**{k: torch.cat(v) for k, v in cols.items()})
        m = train_metrics(torch.cat(preds["A"]), torch.cat(preds["E"]), T_all, torch.cat(eqs), prefix=f"{PAIR}/train")
        logs: Dict[str, float] = {f"train/{k}": m.pop(f"{PAIR}/train/{k}") for k in TARGET_KEYS}
        logs.update(m)
        for h in HEADS:
            logs[f"{PAIR}/train/xmse_{h}_ema"] = self.ema[f"xmse_{h}"].update(m[f"{PAIR}/train/xmse_{h}"])
            logs[f"{PAIR}/train/grad_norm_{h}"] = grad_norm(self.pair.head(h).parameters())
        logs[f"{PAIR}/train/grad_norm_fork"] = grad_norm(self.fork.parameters())
        lo = losses.tolist()
        logs.update({f"{PAIR}/train/loss": lo[0], f"{PAIR}/train/loss_A": lo[1], f"{PAIR}/train/loss_E": lo[2],
                     f"{PAIR}/train/lr": self.opt_heads.param_groups[0]["lr"],
                     f"{PAIR}/train/lr_fork": self.opt_fork.param_groups[0]["lr"]})
        if a.clip_per == "joint":                        # design §6: the heads share the fork
            torch.nn.utils.clip_grad_norm_(list(self.pair.parameters()) + list(self.fork.parameters()),
                                           a.gradient_clipping)
        else:
            for mod in (self.pair.A, self.pair.E, self.fork):
                torch.nn.utils.clip_grad_norm_(mod.parameters(), a.gradient_clipping)
        self.opt_heads.step()
        self.opt_fork.step()
        self.sched_heads.step()
        self.sched_fork.step()
        logs[f"{PAIR}/train/fork_dist_from_init"] = self.fork.distance_from_init()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        return logs

    # ----- validation

    def evaluate(self, ev: BankEvaluator, best: Optional[BestTracker], step: int) -> Dict[str, float]:
        return ev.evaluate([self.pair], fork=self.fork, best=best, step=step)


def main(argv=None) -> int:
    args = finalize(build_parser("B").parse_args(argv), arm="B")
    return ForkTrainer(args).run()


if __name__ == "__main__":
    sys.exit(main())
