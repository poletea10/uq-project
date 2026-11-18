"""Validation metrics, slices and reference lines (phase1_protocol.md §7.3-7.5, §8.1; transfer guide §3.4).

ONE DEFINITION FOR THE BANK AND THE TRAINER
    The reference lines are computed once, when the validation bank is built (scripts/01_make_val_bank.py), and
    stored in its manifest; the trainer logs the heads' metrics during training. Both use the functions below and
    the same key names, so a head curve and its reference lines land on the same wandb panel:
        val/<metric>_<head>/<slice>      a head's value (trainer), e.g. val/xmse_E/hole
        ref/<line>_<head>/<slice>        a reference line (manifest, logged as a constant), e.g. ref/eq_proxy_xmse_E/hole
    <head> is A (aleatoric) or E (epistemic / divergence). Keys are flat strings and values plain floats, the form
    `wandb.log` takes; empty slices are omitted rather than logged as NaN.

SLICES: "all" (pooled), the six query types (`iltm2.prior._split.QUERY_TYPES`: iid, hole, hole_thinned,
    extrapolation, extrapolation_thinned, duplicate) and the six n bins (n_le256, n_256-512, n_512-1K, n_1K-4K,
    n_4K-16K, n_16K-60K). n is the CONTEXT size n_ctx of the query's task (of the pass, in training).

TWO WEIGHTINGS (transfer guide §3.4, §3.6). Every run reports both, whatever it was trained with:
    unweighted (p = 0)   mse, xmse, skill, and the reference lines below
    weighted   (p = 1)   the same with each query's term multiplied by w = (e(q_theta) + c)^-1, c the bank's
                         floor (manifest["loss_c"], fixed at 0.1): keys wmse, wxmse, wskill and the w-
                         reference lines. Weighted means are count-normalized, mean_q[w x], the form of the loss.
                         The weighted constant baseline is the best constant under that weighting, the weighted
                         mean target sum w t / sum w.

METRICS
    mse(p, t)          mean (p - t)^2 against the full target (R_val = 128 replicates on the bank)
    xmse(p, ta, tb)    mean (p - ta)(p - tb) with the two half-split targets (replicates 1-64 / 65-128): exact in
                       expectation for the error against the realized task, replicate noise removed (§7.3)
    skill              1 - xMSE_head / xMSE_const (§8.1)
    ratio              mean of max(p, eps) / max(e_q, eps), eps = 1e-6 (§7.5)

REFERENCE LINES (§8.1), per head and slice, for both weightings (the weighted ones prefixed with w)
    noise_floor   mean (t^2 - ta tb): the replicate-noise part of the raw MSE (raw-MSE panels only)
    const_*       MSE / xMSE of predicting the bank-wide (weighted) mean target everywhere: where every head starts
    eq_proxy_*    MSE / xMSE / skill of the best predictor of the target from e(q) alone: 50 equal-count e(q) bins
                  predicting their (weighted) mean target, cross-fitted on two halves of the bank's TASKS (each half
                  scored with the bins fitted on the other half, so every query is scored out-of-fold and no task's
                  own targets enter its prediction)
    plus per slice: count, mean target, mean e(q), mean target ratio t/e(q), share of t_E < 0.

HEAD METRICS (what the trainer logs; same slices, same names)
    head_metrics     a head pair's predictions on the bank (val/...) or the mini-bank (mini/...): per head and
                     slice mse, xmse, skill, wmse, wxmse, wskill (skills against the bank's own constant-baseline
                     xMSE of the same weighting), mean prediction; and the ratio pair val/ratio_A_eq/<s> and
                     val/ratio_tA_eq/<s> (protocol §7.5)
    train_metrics    one training step (protocol §7.3): mse and xmse per head against the 16-replicate targets,
                     xmse per n bin (the first-hours check of transfer guide §5 Step 3), means of targets /
                     predictions / e(q), share of t_E < 0
    EMA              smoothing 0.99, bias-corrected, reset at each stage start (protocol §7.3)
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from iltm2.prior._split import QUERY_TYPES

HEADS = ("A", "E")
EPS = 1e-6
# Context-size bins, as (name, lower bound exclusive, upper bound inclusive) (transfer guide §3.4).
N_BINS = (("n_le256", 0, 256), ("n_256-512", 256, 512), ("n_512-1K", 512, 1024), ("n_1K-4K", 1024, 4096),
          ("n_4K-16K", 4096, 16384), ("n_16K-60K", 16384, 60000))
N_BIN_NAMES = tuple(n for n, _, _ in N_BINS)
SLICES = ("all", *QUERY_TYPES.values(), *N_BIN_NAMES)


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
    for i, name in enumerate(N_BIN_NAMES):
        out[name] = n_bin == i
    return out


def eq_weight(e_q: torch.Tensor, c: float) -> torch.Tensor:
    """The p = 1 weight w = (e(q) + c)^-1, float64 (transfer guide §3.4, §3.6)."""
    assert c > 0, "the weighted metrics need the bank's positive floor c"
    return 1.0 / (e_q.double() + c)


def mse(p: torch.Tensor, t: torch.Tensor, w: Optional[torch.Tensor] = None) -> float:
    err = (p.double() - t.double()) ** 2
    return float((err if w is None else w * err).mean())


def xmse(p: torch.Tensor, ta: torch.Tensor, tb: torch.Tensor, w: Optional[torch.Tensor] = None) -> float:
    p = p.double()
    err = (p - ta.double()) * (p - tb.double())
    return float((err if w is None else w * err).mean())


def skill(xmse_head: float, xmse_const: float) -> float:
    return 1.0 - xmse_head / xmse_const if xmse_const > 0 else float("nan")


def ratio(p: torch.Tensor, e_q: torch.Tensor) -> float:
    return float((p.double().clamp_min(EPS) / e_q.double().clamp_min(EPS)).mean())


def targets_of(Q, head: str):
    """(t, t^(a), t^(b)) of a head from a query table (`iltm2.prior._bank.BankQueries`)."""
    return (Q.t_A, Q.t_A_a, Q.t_A_b) if head == "A" else (Q.t_E, Q.t_E_a, Q.t_E_b)


def weighted_mean(t: torch.Tensor, w: Optional[torch.Tensor] = None) -> float:
    t = t.double()
    return float(t.mean()) if w is None else float((w * t).sum() / w.sum())


def eq_proxy_predictions(e_q: torch.Tensor, t: torch.Tensor, task_id: torch.Tensor, n_bins: int = 50,
                         w: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Out-of-fold predictions of t from e(q) alone (module docstring): fold = task_id % 2; each fold's queries are
    predicted with the equal-count bins (edges = e(q) quantiles) and the (w-weighted) bin means of the other fold."""
    e_q, t = e_q.double(), t.double()
    w = torch.ones_like(t) if w is None else w.double()
    fold = task_id % 2
    pred = torch.empty_like(t)
    for f in (0, 1):
        fit, score = fold != f, fold == f
        edges = torch.quantile(e_q[fit], torch.linspace(0, 1, n_bins + 1, dtype=torch.float64)[1:-1])
        b_fit, b_score = torch.bucketize(e_q[fit], edges), torch.bucketize(e_q[score], edges)
        sums = torch.zeros(n_bins, dtype=torch.float64).index_add_(0, b_fit, w[fit] * t[fit])
        wsum = torch.zeros(n_bins, dtype=torch.float64).index_add_(0, b_fit, w[fit])
        fallback = float((w[fit] * t[fit]).sum() / w[fit].sum())
        means = torch.where(wsum > 0, sums / wsum.clamp_min(1e-300), fallback)   # tied edges can empty a bin
        pred[score] = means[b_score]
    return pred


def const_values(Q, c: float) -> Dict[str, float]:
    """The constant baselines of a query table: {"A", "E"} unweighted and {"wA", "wE"} weighted mean targets."""
    w = eq_weight(Q.e_q, c)
    out = {h: weighted_mean(targets_of(Q, h)[0]) for h in HEADS}
    out.update({f"w{h}": weighted_mean(targets_of(Q, h)[0], w) for h in HEADS})
    return out


def proxy_predictions(Q, c: float) -> Dict[str, torch.Tensor]:
    """The e(q) proxy of a query table for both weightings: keys "A", "E", "wA", "wE"."""
    w = eq_weight(Q.e_q, c)
    out = {h: eq_proxy_predictions(Q.e_q, targets_of(Q, h)[0], Q.task_id) for h in HEADS}
    out.update({f"w{h}": eq_proxy_predictions(Q.e_q, targets_of(Q, h)[0], Q.task_id, w=w) for h in HEADS})
    return out


def reference_lines(Q, c: float, const: Optional[Dict[str, float]] = None,
                    eq_proxy: Optional[Dict[str, torch.Tensor]] = None, prefix: str = "ref") -> Dict[str, float]:
    """The §8.1 reference lines and per-slice target summaries of a query table, as a flat {key: float} dict.

    c         the weight floor of the weighted lines (the bank's manifest["loss_c"])
    const     `const_values` (default: this table's own). The mini-bank passes the full bank's constants, which is
              what a head's initialization outputs.
    eq_proxy  `proxy_predictions` (default: cross-fitted on this table).
    """
    out: Dict[str, float] = {f"{prefix}/loss_c": float(c)}
    const = const or const_values(Q, c)
    eq_proxy = eq_proxy if eq_proxy is not None else proxy_predictions(Q, c)
    w_all = eq_weight(Q.e_q, c)
    for h in HEADS:
        out[f"{prefix}/const_value_{h}"] = const[h]
        out[f"{prefix}/wconst_value_{h}"] = const[f"w{h}"]
    for s, mask in slice_masks(Q.query_type, Q.n_bin).items():
        n = int(mask.sum())
        if n == 0:
            continue
        out[f"{prefix}/count/{s}"] = float(n)
        e_q, w = Q.e_q[mask], w_all[mask]
        out[f"{prefix}/mean_eq/{s}"] = float(e_q.double().mean())
        for h in HEADS:
            t, ta, tb = (x[mask] for x in targets_of(Q, h))
            out[f"{prefix}/mean_t{h}/{s}"] = float(t.double().mean())
            out[f"{prefix}/ratio_t{h}_eq/{s}"] = ratio(t, e_q)
            for pre, ww in (("", None), ("w", w)):
                cst = torch.full_like(t, const[f"{pre}{h}"], dtype=torch.float64)
                px = eq_proxy[f"{pre}{h}"][mask]
                c_x = xmse(cst, ta, tb, ww)
                noise = t.double() ** 2 - ta.double() * tb.double()
                out[f"{prefix}/{pre}noise_floor_{h}/{s}"] = float((noise if ww is None else ww * noise).mean())
                out[f"{prefix}/{pre}const_mse_{h}/{s}"] = mse(cst, t, ww)
                out[f"{prefix}/{pre}const_xmse_{h}/{s}"] = c_x
                out[f"{prefix}/{pre}eq_proxy_mse_{h}/{s}"] = mse(px, t, ww)
                out[f"{prefix}/{pre}eq_proxy_xmse_{h}/{s}"] = xmse(px, ta, tb, ww)
                out[f"{prefix}/{pre}eq_proxy_skill_{h}/{s}"] = skill(xmse(px, ta, tb, ww), c_x)
        out[f"{prefix}/frac_tE_neg/{s}"] = float((Q.t_E[mask] < 0).double().mean())
    return out


# ------------------------------------------------------------------------------------------- head metrics


def head_metrics(preds: Dict[str, torch.Tensor], Q, refs: Dict[str, float], c: float, ref_prefix: str = "ref",
                 prefix: str = "val", slices: Optional[tuple] = None) -> Dict[str, float]:
    """A head pair's metrics on a query table, as a flat {key: float} dict (keys without the head id; the caller
    prefixes it, protocol §7.1).

    preds       {"A": Â, "E": Ê_dir}, each (Q,) in the query table's row order
    refs        the table's reference lines (bank manifest["reference_lines"], or manifest["mini"]["reference_lines"]
                with ref_prefix "ref_mini"): skills use refs[f"{ref_prefix}/[w]const_xmse_<h>/<slice>"]
    c           the weight floor (the bank's manifest["loss_c"])
    slices      slice names to report (default: every non-empty slice)
    Keys: {prefix}/{mse,xmse,skill,wmse,wxmse,wskill,mean}_<h>/<s>, {prefix}/ratio_A_eq/<s>, {prefix}/ratio_tA_eq/<s>.
    """
    assert all(len(preds[h]) == len(Q) for h in HEADS), "one prediction per query row"
    out: Dict[str, float] = {}
    w_all = eq_weight(Q.e_q, c)
    for s, mask in slice_masks(Q.query_type, Q.n_bin).items():
        if (slices is not None and s not in slices) or not bool(mask.any()):
            continue
        w = w_all[mask]
        for h in HEADS:
            p = preds[h][mask]
            t, ta, tb = (x[mask] for x in targets_of(Q, h))
            for pre, ww in (("", None), ("w", w)):
                x = xmse(p, ta, tb, ww)
                out[f"{prefix}/{pre}mse_{h}/{s}"] = mse(p, t, ww)
                out[f"{prefix}/{pre}xmse_{h}/{s}"] = x
                c_x = refs.get(f"{ref_prefix}/{pre}const_xmse_{h}/{s}")
                if c_x is not None:
                    out[f"{prefix}/{pre}skill_{h}/{s}"] = skill(x, c_x)
            out[f"{prefix}/mean_{h}/{s}"] = float(preds[h][mask].double().mean())
        e_q = Q.e_q[mask]
        out[f"{prefix}/ratio_A_eq/{s}"] = ratio(preds["A"][mask], e_q)
        out[f"{prefix}/ratio_tA_eq/{s}"] = ratio(Q.t_A[mask], e_q)
    return out


def train_metrics(A_hat: torch.Tensor, E_hat: torch.Tensor, T, e_q: torch.Tensor, prefix: str = "train",
                  n_bin: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Metrics of one training batch (protocol §7.3). T: `iltm2.train._loss.Targets` of the same queries, all
    flattened alike. Computed on the predictions made BEFORE the optimizer step, so each value is an evaluation on
    unseen tasks. With `n_bin` (the n bin index of each query's pass), also xmse per n bin."""
    A_hat, E_hat = A_hat.detach(), E_hat.detach()
    out = {f"{prefix}/mse_A": mse(A_hat, T.t_A), f"{prefix}/mse_E": mse(E_hat, T.t_E),
           f"{prefix}/xmse_A": xmse(A_hat, T.t_A_a, T.t_A_b), f"{prefix}/xmse_E": xmse(E_hat, T.t_E_a, T.t_E_b),
           f"{prefix}/mean_A": float(A_hat.double().mean()), f"{prefix}/mean_E": float(E_hat.double().mean()),
           f"{prefix}/mean_tA": float(T.t_A.double().mean()), f"{prefix}/mean_tE": float(T.t_E.double().mean()),
           f"{prefix}/mean_eq": float(e_q.double().mean()),
           f"{prefix}/frac_tE_neg": float((T.t_E < 0).double().mean())}
    if n_bin is not None:
        for i, name in enumerate(N_BIN_NAMES):
            m = n_bin == i
            if bool(m.any()):
                out[f"{prefix}/xmse_A/{name}"] = xmse(A_hat[m], T.t_A_a[m], T.t_A_b[m])
                out[f"{prefix}/xmse_E/{name}"] = xmse(E_hat[m], T.t_E_a[m], T.t_E_b[m])
    return out


class EMA:
    """Exponential moving average with bias correction: after n updates, sum_i beta^(n-i) (1-beta) x_i / (1-beta^n),
    so a constant input gives that constant from the first update. `reset()` at each stage start (protocol §7.3)."""

    def __init__(self, beta: float = 0.99):
        self.beta, self.value, self.n = beta, 0.0, 0

    def update(self, x: float) -> float:
        self.n += 1
        self.value = self.beta * self.value + (1.0 - self.beta) * float(x)
        return self.get()

    def get(self) -> float:
        return self.value / (1.0 - self.beta ** self.n) if self.n else float("nan")

    def reset(self):
        self.value, self.n = 0.0, 0

    def state_dict(self) -> dict:
        return dict(beta=self.beta, value=self.value, n=self.n)

    def load_state_dict(self, d: dict):
        self.beta, self.value, self.n = d["beta"], d["value"], d["n"]
