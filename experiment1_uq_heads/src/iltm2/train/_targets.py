"""Head targets from R exact replicates of the query target (phase1_design.md §2, phase1_protocol.md §7.3).

Given replicates y^(1..R) of a query from the observed-cut prior and the frozen forecast q_theta:

    t_A = 1/(R(R-1)) Σ_{r<s} |y_r - y_s|                  unbiased for the task's CRPS entropy A
    t_E = (1/R) Σ_r CRPS(sg[q_theta], y_r) - t_A           unbiased for the divergence d_S(q_theta, p_tau)

Raw estimates only: never clip t_E at zero (single targets can be negative; clipping biases the mean).
The half-split targets feed the xMSE metric and the bank's noise floor: the R replicates are split into two
disjoint halves (first R/2, last R/2; the replicates are i.i.d., so contiguous halves are as good as any), and
each half's targets are computed with the same functions as the full target.
"""
import torch

from iltm2.model._entropy import crps


def t_aleatoric(Y):
    """t_A per row. Y: (..., R), R >= 2 -> (...).

    Σ_{r<s} |y_r - y_s| = Σ_k (2k - R + 1) y_(k) over the sorted row (k = 0..R-1): O(R log R) instead of O(R^2).
    """
    R = Y.shape[-1]
    assert R >= 2, "t_A needs at least 2 replicates"
    Ys = torch.sort(Y, dim=-1).values
    coef = 2.0 * torch.arange(R, dtype=Ys.dtype, device=Ys.device) - (R - 1)
    return (Ys @ coef) / (R * (R - 1))


def t_epistemic(q, Y, t_A=None):
    """t_E per row. q: (..., K) forecast quantiles; Y: (..., R) -> (...). Gradients never flow into q."""
    if t_A is None:
        t_A = t_aleatoric(Y)
    return crps(q.detach(), Y).mean(dim=-1) - t_A


def targets(q, Y):
    """(t_A, t_E) from all R replicates."""
    t_A = t_aleatoric(Y)
    return t_A, t_epistemic(q, Y, t_A)


def half_split_targets(q, Y):
    """((t_A^(a), t_E^(a)), (t_A^(b), t_E^(b))) from replicates [:R/2] and [R/2:]. R must be even and >= 4."""
    R = Y.shape[-1]
    assert R % 2 == 0 and R >= 4, f"half-split targets need an even R >= 4, got R = {R}"
    return targets(q, Y[..., : R // 2]), targets(q, Y[..., R // 2 :])
