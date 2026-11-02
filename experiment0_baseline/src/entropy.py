"""CRPS entropy and excess kernels (phase0_protocol.md §4.3), grid and closed forms.

    e_CRPS(F) = ∫ F(1-F) dt = ½ E|X - X'|,        d_S(F, G) = ∫ (F - G)^2 dt.

Grid kernels serve any CDF. Families with exact expressions use them instead, because a grid cannot
resolve a member clipped at sigma_floor:
  - Gaussian mixtures (heteroscedastic GP): folded-normal sums for e, member entropies and the excess
    against a Gaussian or point-mass truth.
  - Piecewise-linear quantile CDFs (TabICLv2): exact segment-wise integral.
scripts/00c_validate_adapters.py checks every closed form against the grid kernel.
"""
import numpy as np
from scipy.special import ndtr

SQRT_PI = np.sqrt(np.pi)
SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)


# ---------------------------------------------------------------- shared t-grid


def make_grid(lo, hi, p):
    """Per-query linear grids: lo, hi of shape (q,) -> (q, p)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    assert np.all(hi > lo), "empty t-grid"
    return lo[:, None] + (hi - lo)[:, None] * np.linspace(0.0, 1.0, p)[None, :]


def check_tails(F, tol, what="cdf"):
    """Both ends of each CDF row must be in the tails: F[:, 0] < tol and F[:, -1] > 1 - tol."""
    lo_bad = F[:, 0] >= tol
    hi_bad = F[:, -1] <= 1.0 - tol
    assert not lo_bad.any() and not hi_bad.any(), (
        f"{what}: t-grid does not reach the tails "
        f"(left max {F[:, 0].max():.2e}, right min {F[:, -1].min():.6f}, tol {tol})")


# ---------------------------------------------------------------- grid kernels


def e_crps(F, t):
    """F, t: (..., p) -> (...)."""
    return np.trapezoid(F * (1.0 - F), t, axis=-1)


def excess(F, G, t):
    d = F - G
    return np.trapezoid(d * d, t, axis=-1)


def project(member_F, weights, t):
    """Member-wise projection on a grid.

    member_F: (q, M, p) member CDFs; weights: (M,) or (q, M), rows sum to 1; t: (q, p).
    Returns (A_hat, e_total, E_hat), each (q,). A_hat is the mean member entropy, never a conversion of
    an aggregated variance split.
    """
    w = np.broadcast_to(np.asarray(weights, float), member_F.shape[:2])
    A = np.sum(w * e_crps(member_F, t[:, None, :]), axis=1)
    bar = np.einsum("qm,qmp->qp", w, member_F)
    e = e_crps(bar, t)
    assert np.all(e - A >= -1e-9 * np.maximum(e, 1.0)), "negative epistemic: projection applied wrongly"
    return A, e, e - A


# ---------------------------------------------------------------- Gaussian closed forms


def folded_mean(m, s):
    """E|Z| for Z ~ N(m, s^2), elementwise; s = 0 gives |m|."""
    m, s = np.broadcast_arrays(np.asarray(m, float), np.asarray(s, float))
    out = np.abs(m).astype(float)
    pos = s > 0
    z = m[pos] / s[pos]
    out[pos] = s[pos] * SQRT_2_OVER_PI * np.exp(-0.5 * z * z) + m[pos] * (2.0 * ndtr(z) - 1.0)
    return out


def _weights(w, shape):
    return np.broadcast_to(np.asarray(w, float), shape)


def gm_member_entropy(sigma):
    """e_CRPS(N(mu, sigma^2)) = sigma / sqrt(pi)."""
    return np.asarray(sigma, float) / SQRT_PI


def gm_entropy(mu, sigma, w, max_block=4_000_000):
    """e_CRPS of a Gaussian mixture, exact: ½ Σ_ij w_i w_j E|N(mu_i - mu_j, sigma_i^2 + sigma_j^2)|.

    mu, sigma: (q, M); w: (M,) or (q, M). Chunked over queries so q_chunk * M^2 <= max_block.
    """
    mu, sigma = np.asarray(mu, float), np.asarray(sigma, float)
    w = _weights(w, mu.shape)
    q, M = mu.shape
    step = max(1, max_block // (M * M))
    out = np.empty(q)
    for i in range(0, q, step):
        sl = slice(i, i + step)
        dm = mu[sl, :, None] - mu[sl, None, :]
        ds = np.sqrt(sigma[sl, :, None] ** 2 + sigma[sl, None, :] ** 2)
        ww = w[sl, :, None] * w[sl, None, :]
        out[sl] = 0.5 * np.sum(ww * folded_mean(dm, ds), axis=(1, 2))
    return out


def gm_expected_abs(mu, sigma, w, loc, sd):
    """E|X - Y|, X ~ mixture(mu, sigma, w), Y ~ N(loc, sd^2) independent. loc, sd: (q,) or (q, R)."""
    mu, sigma = np.asarray(mu, float), np.asarray(sigma, float)
    w = _weights(w, mu.shape)
    loc, sd = np.asarray(loc, float), np.asarray(sd, float)
    if loc.ndim == 1:
        return np.sum(w * folded_mean(mu - loc[:, None], np.sqrt(sigma**2 + sd[:, None] ** 2)), axis=1)
    sd = np.broadcast_to(sd, loc.shape)
    return np.sum(w[:, :, None] * folded_mean(mu[:, :, None] - loc[:, None, :],
                                              np.sqrt(sigma[:, :, None] ** 2 + sd[:, None, :] ** 2)), axis=1)


def gm_excess_vs_normal(mu, sigma, w, loc, sd, e_forecast=None):
    """d_S(mixture, N(loc, sd^2)) = E|X-Y| - ½E|X-X'| - ½E|Y-Y'|, exact. sd = 0 is a point mass."""
    if e_forecast is None:
        e_forecast = gm_entropy(mu, sigma, w)
    return gm_expected_abs(mu, sigma, w, loc, sd) - e_forecast - np.asarray(sd, float) / SQRT_PI


def gm_crps(mu, sigma, w, y, e_forecast=None):
    """CRPS(mixture, y) = E|X - y| - ½E|X - X'|. y: (q,) or (q, R)."""
    if e_forecast is None:
        e_forecast = gm_entropy(mu, sigma, w)
    y = np.asarray(y, float)
    ea = gm_expected_abs(mu, sigma, w, y, np.zeros_like(y))
    return ea - (e_forecast if y.ndim == 1 else e_forecast[:, None])


# ---------------------------------------------------------------- piecewise-linear quantile CDF


def quantile_entropy(qs, alphas):
    """Exact ∫F(1-F) for the CDF that is piecewise-linear through (qs_k, alphas_k), 0 below qs_0, 1 above.

    On a segment where F goes linearly from a0 to a1 over a width dq:
        ∫ F(1-F) = dq * [(a0 + a1)/2 - (a0^2 + a0 a1 + a1^2)/3].
    qs: (q, K) sorted; alphas: (K,).
    """
    dq = np.diff(qs, axis=1)
    a0, a1 = alphas[:-1], alphas[1:]
    seg = 0.5 * (a0 + a1) - (a0 * a0 + a0 * a1 + a1 * a1) / 3.0
    return dq @ seg
