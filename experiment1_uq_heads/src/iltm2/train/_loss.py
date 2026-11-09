"""The heads' loss (phase1_design.md Architecture §3) and the per-batch targets it is computed against.

    L = mean_q[ w (Â - t_A)^2 ] + lambda_E mean_q[ w (Ê_dir - t_E)^2 ],     w = (sg[e(q_theta)] + c)^(-p)

    - mean_q is the average over the query rows passed in. A step made of several micro-batch groups of different
      sizes m weights each group by its share of the step's query rows (`group_weight`), so the step's loss is the
      average over all its query rows, whatever the grouping.
    - p = 0 (the default of every run) gives w = 1 exactly: plain unweighted MSE. p = 2 (inverse-variance weighting,
      with the floor c, e.g. the 5th percentile of e(q) over the prior) is a fallback ablation only. The weight
      depends on (x_*, D) only, through the frozen e(q), so it never moves the minimiser (design §3).
    - The heads share no parameter, so lambda_E only rescales the Ê head's gradient (default 1).
    - Targets are the raw unbiased estimates (`iltm2.train._targets`): never clipped at 0.

`batch_targets` computes, from the frozen forecast q and the R replicates of a micro-batch group, the full targets
(what the loss uses) and the half-split targets (what the xMSE metric uses, protocol §7.3), in float64.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from iltm2.train._targets import half_split_targets, targets


@dataclass
class Targets:
    """Targets of a group's queries, each (B, m) float64: full (loss) and half-split (xMSE)."""
    t_A: Tensor
    t_E: Tensor
    t_A_a: Tensor
    t_A_b: Tensor
    t_E_a: Tensor
    t_E_b: Tensor


@torch.no_grad()
def batch_targets(q: Tensor, y_reps: Tensor) -> Targets:
    """q (B, m, K) the frozen forecast, y_reps (B, m, R) the replicates (R even, >= 4) -> Targets."""
    q, Y = q.double(), y_reps.to(q.device).double()
    t_A, t_E = targets(q, Y)
    (tAa, tEa), (tAb, tEb) = half_split_targets(q, Y)
    return Targets(t_A, t_E, tAa, tAb, tEa, tEb)


def loss_weight(e_q: Tensor, p: float = 0.0, c: Optional[float] = None) -> Tensor:
    """w = (sg[e(q)] + c)^(-p); exactly 1 for p = 0. e(q) is detached: no gradient ever flows into the PPD."""
    if p == 0:
        return torch.ones_like(e_q)
    assert c is not None and c > 0, "p > 0 needs a positive floor c (design §3)"
    return (e_q.detach() + c).pow(-p)


@dataclass
class LossOut:
    loss: Tensor            # scalar: loss_A + lambda_E * loss_E (what is backpropagated)
    loss_A: Tensor          # scalar: weighted MSE of Â (= unweighted MSE for p = 0)
    loss_E: Tensor          # scalar: weighted MSE of Ê_dir
    n: int                  # query rows averaged over


def uq_loss(A_hat: Tensor, E_hat: Tensor, t_A: Tensor, t_E: Tensor, e_q: Tensor, p: float = 0.0,
            c: Optional[float] = None, lambda_E: float = 1.0) -> LossOut:
    """The design §3 loss over the query rows given (any shape, all the same). Targets and e(q) are cast to the
    heads' dtype; they carry no gradient."""
    dt = A_hat.dtype
    w = loss_weight(e_q.to(dt), p, c)
    loss_A = (w * (A_hat - t_A.to(dt).detach()) ** 2).mean()
    loss_E = (w * (E_hat - t_E.to(dt).detach()) ** 2).mean()
    return LossOut(loss_A + lambda_E * loss_E, loss_A, loss_E, A_hat.numel())


def group_weight(n_group: int, n_step: int) -> float:
    """Factor of a group's mean loss in a step of several groups: its share of the step's query rows, so that the
    accumulated gradient is that of the mean over all the step's query rows."""
    return n_group / n_step
