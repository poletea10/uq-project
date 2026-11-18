"""The A-job trainer (phase1_protocol.md §3 step 8; design Training §5; transfer guide §3.3, §3.6, §3.9, §4): one
frozen backbone pass per micro-batch group feeds every head pair of the job, head LR x loss exponent p.

    python -m iltm2.train.frozen --stage {1,2,3} --seed S [--prior carved|iid] [options of iltm2.train._config]

ONE STEP (BATCH_SIZE tasks, as micro-batch groups of one (N, n_ctx) each; in stage 1 each group has two passes, its
full context and its subsampled context, so a step is 32 tasks and 64 forwards)
    for each pass of each group: backbone forward under no_grad (q, e(q), h_star), the pass's targets from its R
    replicates (`batch_targets`: t_A is the same in both passes, t_E follows the pass's forecast), then for each
    pair: Â, Ê_dir from h_star, the loss with the pair's own p (w = (e(q) + c)^-p, c the bank's floor), and
    backward of loss x (the pass's share of the step's query rows), so each pair's accumulated gradient is that of
    the mean loss over all the step's query rows (both passes), and the pass's activations are freed at once.
    Then, per pair: gradient norms (per head), clipping (--clip_per: each head on its own norm, the default, design
    §6, or the pair's joint norm), its own AdamW step and its own scheduler step. Training metrics are computed on
    the predictions made before the step (protocol §7.3), so each is an evaluation on tasks the heads have not
    trained on; the xMSE is also logged per n bin of the pass's context size (the stage-1 vs mini-bank check of
    guide §5 Step 3).

VALIDATION (`iltm2.train._validate.BankEvaluator`)
    h_star is cached once per launch on the full bank and on the mini-bank, so a validation costs only the head
    MLPs (transfer guide §3.3). Best-per-head checkpoints on the full bank only,
    each pair under its own weighting.

SMOKE TEST --overfit_steps n: a fixed group of stage-1 tasks (its full pass), chosen by its A targets: among the groups
of step 0 (then of the next steps, up to OVERFIT_MAX_STEPS, if none qualifies) the one with the largest share of
queries with t_A > 0, which must be at least OVERFIT_MIN_NONZERO. A task flagged noisy (a private root reaches y) can
still have t_A = 0 at every query (the noise reaches y only in some regions of x); a batch whose t_A is all 0 leaves
the A head nothing to fit (it starts at MIN_TARGET = 1e-6 against 0, MSE 1e-12, and softplus cannot reach 0), so the
choice is made on the targets themselves (from the replicates, before any forward). Its backbone outputs are computed
once, every pair trained n full-batch steps at --overfit_lr with the unweighted
loss (one LR and one loss for all pairs, so pairs that share the initialization give identical results); the training
MSE of each head must fall to at most OVERFIT_RATIO (5%) of its step-0 value, the constant-baseline MSE every head
starts from (protocol §6 run 1: the heads can memorize their targets, so the plumbing from batch to backbone to heads
to loss works). The batch's replicate-noise floor mean(t^2 - t^(a) t^(b)) is reported, not gated: on a batch with
deterministic tasks the t_E floor is far below t_E's spread (a deterministic task's t_E has no replicate noise), so a
floor-based bar asks the E head for a far larger reduction than the A head. Prints OK / FAIL lines.
"""
from __future__ import annotations

import sys
from typing import Dict, List, Optional, Tuple

import torch

from iltm2.model._backbone import FrozenBackbone
from iltm2.model._heads import HEADS, build_a_job_heads
from iltm2.prior._dataset import CutGraphPrior, UQBatch
from iltm2.train._base import BaseTrainer, get_scheduler
from iltm2.train._config import build_parser, finalize
from iltm2.train._loss import Targets, batch_targets, group_weight, uq_loss
from iltm2.train._metrics import EMA, eq_weight, mse, n_bin_index, train_metrics
from iltm2.train._targets import t_aleatoric
from iltm2.train._validate import BankEvaluator, BestTracker

OVERFIT_RATIO = 0.05          # overfit smoke test: final train MSE <= 5% of the step-0 (constant) MSE
OVERFIT_MIN_NONZERO = 0.10    # the overfit batch needs t_A > 0 at >= 10% of its queries (module docstring)
OVERFIT_MAX_STEPS = 4         # steps of the stage-1 stream searched for such a batch
TARGET_KEYS = ("mean_tA", "mean_tE", "mean_eq", "frac_tE_neg")      # train metrics that do not depend on the pair
TARGET_COLS = ("t_A", "t_E", "t_A_a", "t_A_b", "t_E_a", "t_E_b")


def grad_norm(params) -> float:
    grads = [p.grad.detach().flatten() for p in params if p.grad is not None]
    return float(torch.cat(grads).norm()) if grads else 0.0


def step_passes(groups: List[UQBatch]) -> List[UQBatch]:
    """Every forward of a step: each group's full pass and, in stage 1, its subsampled pass."""
    return [gp for g in groups for gp in g.passes]


def n_queries(gp: UQBatch) -> int:
    return gp.y_reps.shape[0] * gp.y_reps.shape[1]


@torch.no_grad()
def warmup_constants(backbone, groups: List[UQBatch], device, c: float) -> Dict[str, float]:
    """The first batch's mean targets over every pass: {"A", "E"} unweighted, {"wA", "wE"} weighted by
    w = (e(q) + c)^-1 (the best constant under each pair's own loss)."""
    s = dict(A=0.0, E=0.0, wA=0.0, wE=0.0)
    n = wsum = 0.0
    for gp in step_passes(groups):
        gp = gp.to(device, non_blocking=True)
        o = backbone(gp.X, gp.y_ctx)
        T = batch_targets(o.q, gp.y_reps)
        w = eq_weight(o.e_q, c)
        s["A"] += float(T.t_A.sum())
        s["E"] += float(T.t_E.sum())
        s["wA"] += float((w * T.t_A).sum())
        s["wE"] += float((w * T.t_E).sum())
        n, wsum = n + T.t_A.numel(), wsum + float(w.sum())
    return dict(A=s["A"] / n, E=s["E"] / n, wA=s["wA"] / wsum, wE=s["wE"] / wsum)


class FrozenTrainer(BaseTrainer):
    arm = "A"

    # ----- build

    def build(self):
        a = self.args
        self.backbone = FrozenBackbone(a.backbone_path, a.device)
        self.pairs = [p.to(self.device) for p in build_a_job_heads(a.seed, a.head_lrs, a.p)]
        self.by_id = {p.head_id: p for p in self.pairs}
        self.opts, self.scheds = {}, {}
        for p in self.pairs:
            opt = p.make_optimizer(lr=p.lr / a.lr_divisor, weight_decay=a.weight_decay)
            self.opts[p.head_id] = opt
            self.scheds[p.head_id] = get_scheduler(a, opt)   # tabicl's cosine_with_restarts, warmup, lr_end
        self.ema = {f"{p.head_id}/{k}": EMA(0.99) for p in self.pairs for k in ("xmse_A", "xmse_E")}
        print(f"{self.backbone!r}\nprior {a.prior}, subsampled pass {a.subsample}; pairs: {', '.join(self.by_id)}; "
              f"max LR = LR / {a.lr_divisor:g}", flush=True)

    def build_evaluators(self) -> Tuple[BankEvaluator, BankEvaluator]:
        a = self.args
        full = BankEvaluator(a.val_bank, self.backbone, cache=("h",))
        mini = BankEvaluator(a.val_bank, self.backbone, mini=True, cache=("h",))
        return full, mini

    # ----- state

    def heads_state(self) -> dict:
        return {hid: p.state_dict() for hid, p in self.by_id.items()}

    def load_heads_state(self, d: dict):
        assert set(d) == set(self.by_id), f"checkpoint pairs {sorted(d)} vs job pairs {sorted(self.by_id)}"
        for hid, sd in d.items():
            self.by_id[hid].load_state_dict(sd)

    def optim_state(self) -> dict:
        return dict(opts={h: o.state_dict() for h, o in self.opts.items()},
                    scheds={h: s.state_dict() for h, s in self.scheds.items()})

    def load_optim_state(self, d: dict):
        for hid in self.by_id:
            self.opts[hid].load_state_dict(d["opts"][hid])
            self.scheds[hid].load_state_dict(d["scheds"][hid])

    def extra_state(self) -> dict:
        return dict(ema={k: e.state_dict() for k, e in self.ema.items()})

    def load_extra_state(self, d: dict):
        for k, s in d["ema"].items():
            self.ema[k].load_state_dict(s)

    def best_payload(self, head_id: str, h: str) -> dict:
        p = self.by_id[head_id]
        return dict(head=p.head(h).state_dict(), head_id=head_id, which=h, arm=p.arm, lr=p.lr, p=p.p, seed=p.seed,
                    prior=self.args.prior)

    def lrs(self) -> Dict[str, float]:
        return {hid: o.param_groups[0]["lr"] for hid, o in self.opts.items()}

    # ----- initialization

    def init_output_bias(self, consts: Dict[str, float]):
        for p in self.pairs:
            pre = "w" if p.weighted else ""
            p.init_output_bias(consts[f"{pre}A"], consts[f"{pre}E"])
        print("output biases set to the mean targets: " + ", ".join(f"{k} {v:.5g}" for k, v in consts.items())
              + " (w: weighted, for p > 0 pairs)", flush=True)

    def warmup_means(self, groups: List[UQBatch]) -> Dict[str, float]:
        return warmup_constants(self.backbone, groups, self.device, self.loss_c)

    # ----- one step

    def train_step(self, groups: List[UQBatch]) -> Dict[str, float]:
        a, dev = self.args, self.device
        passes = step_passes(groups)
        n_step = sum(n_queries(gp) for gp in passes)
        for o in self.opts.values():
            o.zero_grad(set_to_none=True)
        preds = {hid: {h: [] for h in HEADS} for hid in self.by_id}
        losses = {hid: torch.zeros(3, device=dev) for hid in self.by_id}
        cols = {k: [] for k in TARGET_COLS}
        eqs, nbins = [], []
        for gp in passes:
            gp = gp.to(dev, non_blocking=True)
            out = self.backbone(gp.X, gp.y_ctx)
            T = batch_targets(out.q, gp.y_reps)
            w = group_weight(n_queries(gp), n_step)
            for p in self.pairs:
                A, E = p(out.h_star)
                lo = uq_loss(A, E, T.t_A, T.t_E, out.e_q, p=p.p, c=self.loss_c, lambda_E=a.lambda_E)
                (lo.loss * w).backward()
                losses[p.head_id] += w * torch.stack([lo.loss, lo.loss_A, lo.loss_E]).detach()
                preds[p.head_id]["A"].append(A.detach().flatten())
                preds[p.head_id]["E"].append(E.detach().flatten())
            for k in cols:
                cols[k].append(getattr(T, k).flatten())
            eqs.append(out.e_q.flatten())
            nbins.append(torch.full((n_queries(gp),), n_bin_index(gp.train_size), device=dev))
        T_all = Targets(**{k: torch.cat(v) for k, v in cols.items()})
        e_all, nb_all = torch.cat(eqs), torch.cat(nbins)

        logs: Dict[str, float] = {}
        for p in self.pairs:
            hid = p.head_id
            m = train_metrics(torch.cat(preds[hid]["A"]), torch.cat(preds[hid]["E"]), T_all, e_all,
                              prefix=f"{hid}/train", n_bin=nb_all)
            for k in TARGET_KEYS:
                logs[f"train/{k}"] = m.pop(f"{hid}/train/{k}")
            logs.update(m)
            for h in HEADS:
                logs[f"{hid}/train/xmse_{h}_ema"] = self.ema[f"{hid}/xmse_{h}"].update(m[f"{hid}/train/xmse_{h}"])
                logs[f"{hid}/train/grad_norm_{h}"] = grad_norm(p.head(h).parameters())
            lo = losses[hid].tolist()
            logs.update({f"{hid}/train/loss": lo[0], f"{hid}/train/loss_A": lo[1], f"{hid}/train/loss_E": lo[2],
                         f"{hid}/train/lr": self.opts[hid].param_groups[0]["lr"]})
            if a.clip_per == "head":                     # each head's step depends on its own gradient only
                for h in HEADS:
                    torch.nn.utils.clip_grad_norm_(p.head(h).parameters(), a.gradient_clipping)
            else:
                torch.nn.utils.clip_grad_norm_(p.parameters(), a.gradient_clipping)
            self.opts[hid].step()
            self.scheds[hid].step()
        logs["train/n_passes"] = float(len(passes))
        if dev.type == "cuda":
            torch.cuda.synchronize()
        return logs

    # ----- validation

    def evaluate(self, ev: BankEvaluator, best: Optional[BestTracker], step: int) -> Dict[str, float]:
        return ev.evaluate(self.pairs, best=best, step=step)

    # ----- smoke test: overfit one fixed batch

    def overfit(self) -> bool:
        a, dev = self.args, self.device
        prior = CutGraphPrior(a.seed, 1, max_steps=OVERFIT_MAX_STEPS, R=a.R, batch_size=a.batch_size, prior=a.prior,
                              subsample=False)
        grp, share, where = None, 0.0, None
        for step in range(OVERFIT_MAX_STEPS):
            for gi in range(prior.n_groups):
                cand = prior.group(step, gi)
                nz = float((t_aleatoric(cand.y_reps.double()) > 0).double().mean())
                if nz > share:
                    grp, share, where = cand, nz, (step, gi)
            if share >= OVERFIT_MIN_NONZERO:
                break
        if grp is None or share < OVERFIT_MIN_NONZERO:
            print(f"FAIL overfit: no stage-1 group of steps 0-{OVERFIT_MAX_STEPS - 1} has t_A > 0 at "
                  f">= {OVERFIT_MIN_NONZERO:.0%} of its queries (best {share:.1%}): the A head has nothing to fit",
                  flush=True)
            return False
        step, gi = where
        print(f"overfit: group {gi} of step {step} chosen by its A targets: t_A > 0 at {share:.1%} of its queries",
              flush=True)
        grp = grp.to(dev)
        with torch.no_grad():
            out = self.backbone(grp.X, grp.y_ctx)
            T = batch_targets(out.q, grp.y_reps)
        floor = {h: float((t.double() ** 2 - ta.double() * tb.double()).mean())
                 for h, (t, ta, tb) in (("A", (T.t_A, T.t_A_a, T.t_A_b)), ("E", (T.t_E, T.t_E_a, T.t_E_b)))}
        print(f"overfit: group {gi} of step {step} ({grp.B} tasks, {grp.m} queries each, "
              f"{int((~grp.deterministic).sum())} noisy); noise floor of its {a.R}-replicate targets: "
              f"A {floor['A']:.4g}, E {floor['E']:.4g}", flush=True)
        for p in self.pairs:
            p.init_output_bias(float(T.t_A.mean()), float(T.t_E.mean()))
        opts = {p.head_id: torch.optim.AdamW(p.parameters(), lr=a.overfit_lr, weight_decay=a.weight_decay)
                for p in self.pairs}
        first: Dict[str, Dict[str, float]] = {}
        fails = 0
        for it in range(a.overfit_steps + 1):
            for p in self.pairs:
                A, E = p(out.h_star)
                if it == 0 or it == a.overfit_steps:
                    cur = {"A": mse(A.detach(), T.t_A), "E": mse(E.detach(), T.t_E)}
                    if it == 0:
                        first[p.head_id] = cur
                    else:
                        last = cur
                        for h in HEADS:
                            f0 = first[p.head_id][h]
                            ok = last[h] <= OVERFIT_RATIO * f0
                            print(f"{'OK  ' if ok else 'FAIL'} overfit {p.head_id} {h}: train MSE {f0:.4g} -> "
                                  f"{last[h]:.4g} after {a.overfit_steps} steps ({last[h] / f0:.2%} of the start; "
                                  f"pass at <= {OVERFIT_RATIO:.0%}); noise floor {floor[h]:.4g} (reported)",
                                  flush=True)
                            fails += 0 if ok else 1
                if it == a.overfit_steps:
                    continue
                opts[p.head_id].zero_grad(set_to_none=True)
                uq_loss(A, E, T.t_A, T.t_E, out.e_q).loss.backward()
                for h in HEADS:
                    torch.nn.utils.clip_grad_norm_(p.head(h).parameters(), a.gradient_clipping)
                opts[p.head_id].step()
        print(f"overfit: {'ALL OK' if not fails else f'{fails} FAILED'}", flush=True)
        return fails == 0


def main(argv=None) -> int:
    args = finalize(build_parser("A").parse_args(argv), arm="A")
    return FrozenTrainer(args).run()


if __name__ == "__main__":
    sys.exit(main())
