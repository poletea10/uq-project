"""CRPS entropy and CRPS of TabICLv2's quantile forecast, in torch (phase1_design.md §2, phase1_protocol.md §3 step 1).

One CDF convention, shared with Phase 0's grading (phase0_protocol.md §5.4, experiment0 `QuantileForecast`):
the K = 999 raw quantiles q_k are sorted, and F is piecewise-linear through (q_k, alpha_k) with
alpha_k = (k+1)/(K+1), 0 below q_0 and 1 above q_{K-1}. So F jumps by alpha_0 at q_0 and by 1 - alpha_{K-1} at
q_{K-1}: no tails are added.

    e_CRPS(F)  = ∫ F(1-F) dt                                  (the forecast's entropy, e(q_theta))
    CRPS(F, y) = ∫ (F(t) - 1{t >= y})^2 dt
               = ∫_{-inf}^y F^2 dt + ∫_y^{inf} (1-F)^2 dt     (the score S(q_theta, y))

Both are integrated exactly, segment by segment. On a segment of width h where F goes linearly from a0 to a1,
    ∫ F^2      = h (a0^2 + a0 a1 + a1^2) / 3,
    ∫ F(1-F)   = h [(a0 + a1)/2 - (a0^2 + a0 a1 + a1^2)/3].
CRPS uses prefix sums of the per-segment ∫F^2 and suffix sums of the per-segment ∫(1-F)^2, so each y costs one
searchsorted plus the two partial integrals of the segment that contains it: O(log K) per y, no (y × K) tensor.

All functions take quantiles of shape (..., K) and are differentiable, but the targets never need gradients
through q (they use sg[q_theta]).
"""
import torch


def quantile_levels(K, dtype=torch.float64, device=None):
    """The levels TabICLv2 is trained on: alpha_k = k / (K+1), k = 1..K (tabicl `quantile_dist.py` default)."""
    return torch.linspace(0.0, 1.0, K + 2, dtype=dtype, device=device)[1:-1]


def _sorted_knots(q):
    """Sorted quantiles and their levels, in q's dtype and device."""
    qs = torch.sort(q, dim=-1).values
    return qs, quantile_levels(qs.shape[-1], dtype=qs.dtype, device=qs.device)


def cdf(q, t):
    """F(t) for the convention above. q: (..., K); t: (..., P) with the same leading dims -> (..., P)."""
    qs, a = _sorted_knots(q)
    K = qs.shape[-1]
    qs = qs.expand(*t.shape[:-1], K).contiguous()
    j = torch.searchsorted(qs, t.contiguous(), right=True) - 1          # segment start: q_j <= t < q_{j+1}
    jc = j.clamp(0, K - 2)
    q0, q1 = qs.gather(-1, jc), qs.gather(-1, jc + 1)
    a0, a1 = a[jc], a[jc + 1]
    h = q1 - q0
    u = torch.where(h > 0, (t - q0) / torch.where(h > 0, h, torch.ones_like(h)), torch.zeros_like(h))
    mid = a0 + (a1 - a0) * u.clamp(0.0, 1.0)
    return torch.where(j < 0, torch.zeros_like(t), torch.where(j >= K - 1, torch.ones_like(t), mid))


def e_crps(q):
    """Exact e_CRPS(F) = ∫ F(1-F) dt per forecast. q: (..., K) -> (...)."""
    qs, a = _sorted_knots(q)
    a0, a1 = a[:-1], a[1:]
    seg = 0.5 * (a0 + a1) - (a0 * a0 + a0 * a1 + a1 * a1) / 3.0
    return torch.diff(qs, dim=-1) @ seg


def crps(q, y):
    """Exact CRPS(F, y) per forecast and value. q: (..., K); y: (..., R) with the same leading dims -> (..., R)."""
    qs, a = _sorted_knots(q)
    K = qs.shape[-1]
    qs = qs.expand(*y.shape[:-1], K).contiguous()
    y = y.to(qs.dtype)
    b = 1.0 - a
    a0, a1, b0, b1 = a[:-1], a[1:], b[:-1], b[1:]
    h = torch.diff(qs, dim=-1)                                          # (..., K-1)

    # cum_F2[..., j] = ∫_{q_0}^{q_j} F^2;  suf_G2[..., j] = ∫_{q_j}^{q_{K-1}} (1-F)^2.  Both (..., K).
    seg_F2 = h * (a0 * a0 + a0 * a1 + a1 * a1) / 3.0
    seg_G2 = h * (b0 * b0 + b0 * b1 + b1 * b1) / 3.0
    zero = torch.zeros_like(qs[..., :1])
    cum_F2 = torch.cat([zero, torch.cumsum(seg_F2, dim=-1)], dim=-1)
    suf_G2 = torch.cat([torch.flip(torch.cumsum(torch.flip(seg_G2, [-1]), dim=-1), [-1]), zero], dim=-1)

    # y inside [q_0, q_{K-1}): split the segment q_j <= y < q_{j+1} (so h_j > 0) at y.
    j = torch.searchsorted(qs, y.contiguous(), right=True) - 1
    jc = j.clamp(0, K - 2)
    q_j, h_j = qs.gather(-1, jc), h.gather(-1, jc)
    L = (y - q_j).clamp(min=0.0)
    L = torch.minimum(L, h_j)
    a_y = a0[jc] + (a1[jc] - a0[jc]) * L / torch.where(h_j > 0, h_j, torch.ones_like(h_j))
    b_y = 1.0 - a_y
    left = L * (a0[jc] ** 2 + a0[jc] * a_y + a_y * a_y) / 3.0                    # ∫_{q_j}^y F^2
    right = (h_j - L) * (b_y * b_y + b_y * b1[jc] + b1[jc] ** 2) / 3.0            # ∫_y^{q_{j+1}} (1-F)^2
    mid = cum_F2.gather(-1, jc) + left + right + suf_G2.gather(-1, jc + 1)

    # y outside: F = 0 below q_0 (so (1-F)^2 = 1 on [y, q_0]) and F = 1 above q_{K-1}.
    below = (qs[..., :1] - y) + suf_G2[..., :1]
    above = cum_F2[..., -1:] + (y - qs[..., -1:])
    return torch.where(j < 0, below, torch.where(j >= K - 1, above, mid))
