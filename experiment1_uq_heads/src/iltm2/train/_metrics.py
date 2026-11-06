"""Validation metrics, slices and reference lines (phase1_protocol.md §7.3-7.5, §8.1).

ONE DEFINITION FOR THE BANK AND THE TRAINER
    The reference lines are computed once, when the validation bank is built (scripts/01_make_val_bank.py), and
    stored in its manifest; the trainer (step 7-8) logs the heads' metrics during training. Both use the functions
    below and the same key names, so a head curve and its reference lines land on the same wandb panel:
        val/<metric>_<head>/<slice>      a head's value (trainer), e.g. val/xmse_E/hole
        ref/<line>_<head>/<slice>        a reference line (manifest, logged as a constant), e.g. ref/eq_proxy_xmse_E/hole
    <head> is A (aleatoric) or E (epistemic / divergence). Keys are flat strings and values plain floats, the form
    `wandb.log` takes; empty slices are omitted rather than logged as NaN.

SLICES (§7.5): "all" (pooled), the four query types, and the five n bins.
    n is the CONTEXT size n_ctx of the query's task (the bins run from <= 512, which only stage-1-like tasks with
    a 30-90% context fraction reach, up to 16K-60K for the largest stage-3-like tasks).

METRICS
    mse(p, t)          mean (p - t)^2 against the full target (R_val = 128 replicates on the bank)
    xmse(p, ta, tb)    mean (p - ta)(p - tb) with the two half-split targets (replicates 1-64 / 65-128): exact in
                       expectation for the error against the realized task, replicate noise removed (§7.3)
    skill              1 - xMSE_head / xMSE_const (§8.1)
    ratio              mean of max(p, eps) / max(e_q, eps), eps = 1e-6 (§7.5)

REFERENCE LINES (§8.1), per head and slice
    noise_floor   mean (t^2 - ta tb): the replicate-noise part of the raw MSE (raw-MSE panels only)
    const_*       MSE / xMSE of predicting the bank-wide mean target everywhere: where every head starts
    eq_proxy_*    MSE / xMSE / skill of the best predictor of the target from e(q) alone: 50 equal-count e(q) bins
                  predicting their mean target, cross-fitted on two halves of the bank's TASKS (each half scored
                  with the bins fitted on the other half, so every query is scored out-of-fold and no task's own
                  targets enter its prediction)
    plus per slice: count, mean target, mean e(q), mean target ratio t/e(q), share of t_E < 0.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from iltm2.prior._split import QUERY_TYPES

HEADS = ("A", "E")
EPS = 1e-6
# Context-size bins of §7.5, as (name, lower bound exclusive, upper bound inclusive).
N_BINS = (("n_le512", 0, 512), ("n_512-1K", 512, 1024), ("n_1K-4K", 1024, 4096), ("n_4K-16K", 4096, 16384),
          ("n_16K-60K", 16384, 60000))


def n_bin_index(n_ctx: int) -> int:
    for i, (_, lo, hi) in enumerate(N_BINS):
        if lo < n_ctx <= hi:
            return i
    raise ValueError(f"context size {n_ctx} outside the n bins")


def slice_masks(query_type: torch.Tensor, n_bin: torch.Tensor) -> Dict[str, torch.Tensor]:
    """{slice name: boolean mask over queries}: "all", the query types, the n bins (empty slices included)."""
    out = {"all": torch.ones_like(query_type, dtype=torch.bool)}
    for code, name in QUERY_TYPES.items():
        out[name] = query_type == code
    for i, (name, _, _) in enumerate(N_BINS):
        out[name] = n_bin == i
    return out


def mse(p: torch.Tensor, t: torch.Tensor) -> float:
    return float(((p.double() - t.double()) ** 2).mean())


def xmse(p: torch.Tensor, ta: torch.Tensor, tb: torch.Tensor) -> float:
    p = p.double()
    return float(((p - ta.double()) * (p - tb.double())).mean())


def skill(xmse_head: float, xmse_const: float) -> float:
    return 1.0 - xmse_head / xmse_const if xmse_const > 0 else float("nan")


def ratio(p: torch.Tensor, e_q: torch.Tensor) -> float:
    return float((p.double().clamp_min(EPS) / e_q.double().clamp_min(EPS)).mean())


def targets_of(Q, head: str):
    """(t, t^(a), t^(b)) of a head from a query table (`iltm2.prior._bank.BankQueries`)."""
    return (Q.t_A, Q.t_A_a, Q.t_A_b) if head == "A" else (Q.t_E, Q.t_E_a, Q.t_E_b)


def eq_proxy_predictions(e_q: torch.Tensor, t: torch.Tensor, task_id: torch.Tensor, n_bins: int = 50
                         ) -> torch.Tensor:
    """Out-of-fold predictions of t from e(q) alone (module docstring): fold = task_id % 2; each fold's queries are
    predicted with the equal-count bins (edges = e(q) quantiles) and bin means of the other fold."""
    e_q, t = e_q.double(), t.double()
    fold = task_id % 2
    pred = torch.empty_like(t)
    for f in (0, 1):
        fit, score = fold != f, fold == f
        edges = torch.quantile(e_q[fit], torch.linspace(0, 1, n_bins + 1, dtype=torch.float64)[1:-1])
        b_fit, b_score = torch.bucketize(e_q[fit], edges), torch.bucketize(e_q[score], edges)
        sums = torch.zeros(n_bins, dtype=torch.float64).index_add_(0, b_fit, t[fit])
        cnts = torch.zeros(n_bins, dtype=torch.float64).index_add_(0, b_fit, torch.ones_like(t[fit]))
        means = torch.where(cnts > 0, sums / cnts.clamp_min(1), t[fit].mean())   # tied edges can empty a bin
        pred[score] = means[b_score]
    return pred


def reference_lines(Q, const: Optional[Dict[str, float]] = None, eq_proxy: Optional[Dict[str, torch.Tensor]] = None,
                    prefix: str = "ref") -> Dict[str, float]:
    """The §8.1 reference lines and per-slice target summaries of a query table, as a flat {key: float} dict.

    const     {"A": c_A, "E": c_E}: the constant prediction (default: this table's mean targets). The mini-bank
              passes the full bank's constants, which is what a head's initialization outputs.
    eq_proxy  {"A": predictions, "E": predictions} of the e(q) proxy (default: cross-fitted on this table).
    """
    out: Dict[str, float] = {}
    const = const or {h: float(targets_of(Q, h)[0].double().mean()) for h in HEADS}
    if eq_proxy is None:
        eq_proxy = {h: eq_proxy_predictions(Q.e_q, targets_of(Q, h)[0], Q.task_id) for h in HEADS}
    for h in HEADS:
        out[f"{prefix}/const_value_{h}"] = const[h]
    for s, mask in slice_masks(Q.query_type, Q.n_bin).items():
        n = int(mask.sum())
        if n == 0:
            continue
        out[f"{prefix}/count/{s}"] = float(n)
        e_q = Q.e_q[mask]
        out[f"{prefix}/mean_eq/{s}"] = float(e_q.double().mean())
        for h in HEADS:
            t, ta, tb = (x[mask] for x in targets_of(Q, h))
            c = torch.full_like(t, const[h], dtype=torch.float64)
            px = eq_proxy[h][mask]
            c_x = xmse(c, ta, tb)
            out[f"{prefix}/mean_t{h}/{s}"] = float(t.double().mean())
            out[f"{prefix}/noise_floor_{h}/{s}"] = float((t.double() ** 2 - ta.double() * tb.double()).mean())
            out[f"{prefix}/const_mse_{h}/{s}"] = mse(c, t)
            out[f"{prefix}/const_xmse_{h}/{s}"] = c_x
            out[f"{prefix}/eq_proxy_mse_{h}/{s}"] = mse(px, t)
            out[f"{prefix}/eq_proxy_xmse_{h}/{s}"] = xmse(px, ta, tb)
            out[f"{prefix}/eq_proxy_skill_{h}/{s}"] = skill(xmse(px, ta, tb), c_x)
            out[f"{prefix}/ratio_t{h}_eq/{s}"] = ratio(t, e_q)
        out[f"{prefix}/frac_tE_neg/{s}"] = float((Q.t_E[mask] < 0).double().mean())
    return out
