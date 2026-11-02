"""Forecast objects q_m: the predictive CDF each adapter returns in UQOutput.cdf.

Every forecast is callable on a per-query grid, cdf(t) with t of shape (q, p), and knows
  - its entropy e(q_m),
  - its excess against a Gaussian or point-mass truth, d_S(q_m, N(loc, sd^2)) -> the realized E(q_m),
  - its CRPS at observed values (the replicate estimator of §4.2; also used by 00b).
Families with closed forms use them; the rest share one grid construction (§4.3): a uniform core over
the forecast's ±10 s.d. region widened to cover the truth, plus thinner uniform tails out to the
forecast's support. Against a smooth (Gaussian) truth the integrand decays to 0 at both grid ends, so the
trapezoid rule is spectrally accurate there. A point-mass truth at y (toy rung 0; CRPS at an observation)
is split at y instead:
    ∫(F - 1{t >= y})^2 dt = ∫_lo^y F^2 dt + ∫_y^hi (1 - F)^2 dt,
each side on its own grid ending exactly at y. Each side's integrand has a non-zero slope at y, which
leaves the trapezoid rule an O(h^2) endpoint error, so the two sides use Simpson's rule (O(h^4)).
"""
import numpy as np
from scipy.integrate import simpson
from scipy.special import ndtr

import entropy as en


# ---------------------------------------------------------------- shared grid machinery


def composite_grid(core_lo, core_hi, lo, hi, p, p_tail):
    """Row-wise grid: p_tail points on [lo, core_lo), p on [core_lo, core_hi], p_tail on (core_hi, hi]."""
    lo, hi = np.minimum(lo, core_lo), np.maximum(hi, core_hi)
    ut = np.linspace(0.0, 1.0, p_tail + 1)
    left = lo[:, None] + (core_lo - lo)[:, None] * ut[None, :-1]
    core = en.make_grid(core_lo, core_hi, p)
    right = core_hi[:, None] + (hi - core_hi)[:, None] * ut[None, 1:]
    return np.concatenate([left, core, right], axis=1)


def span(lo, hi, p):
    """Row-wise linspace that allows zero-width rows (lo == hi)."""
    return lo[:, None] + (hi - lo)[:, None] * np.linspace(0.0, 1.0, p)[None, :]


class Forecast:
    """Base class: subclasses implement cdf(t, rows), support(), core(), entropy()."""

    n_query: int

    def __call__(self, t, rows=None):
        return self.cdf(t, rows)

    def core(self):
        return self.support()

    def quantiles(self, levels, proto):
        """Quantiles of q_m per query, (q, len(levels)), by inverting the CDF on the shared grid.

        Monotone interpolation: for each level, the first node with F >= level and its predecessor
        bracket it, and t is interpolated linearly between them.
        """
        c_lo, c_hi = self.core()
        s_lo, s_hi = self.support()
        p = proto["t_points"]
        t = composite_grid(c_lo, c_hi, s_lo, s_hi, p, p // 4)
        F = np.maximum.accumulate(np.clip(self.cdf(t), 0.0, 1.0), axis=1)
        out = np.empty((t.shape[0], len(levels)))
        rows = np.arange(t.shape[0])
        for k, lev in enumerate(levels):
            i = np.clip(np.argmax(F >= lev, axis=1), 1, t.shape[1] - 1)
            f0, f1 = F[rows, i - 1], F[rows, i]
            t0, t1 = t[rows, i - 1], t[rows, i]
            frac = np.where(f1 > f0, (lev - f0) / np.where(f1 > f0, f1 - f0, 1.0), 0.0)
            out[:, k] = t0 + np.clip(frac, 0.0, 1.0) * (t1 - t0)
        return out

    # ----- truth side

    def excess_vs_normal(self, loc, sd, proto):
        """Realized E(q_m) = ∫(F_m - Φ_τ)^2 dt against N(loc, sd^2) per query; sd = 0 is a point mass."""
        return self._grid_excess(np.asarray(loc, float), np.asarray(sd, float), proto)

    def crps(self, y, proto):
        """CRPS(q_m, y) per query and column of y: (q,) or (q, R)."""
        y = np.asarray(y, float)
        if y.ndim == 1:
            return self._grid_excess(y, np.zeros_like(y), proto)
        return np.stack([self._grid_excess(y[:, r], np.zeros(len(y)), proto) for r in range(y.shape[1])], 1)

    def _grid_excess(self, loc, sd, proto):
        k, p, tol = proto["t_halfwidth_sd"], proto["t_points"], proto["tail_tol"]
        c_lo, c_hi = self.core()
        s_lo, s_hi = self.support()
        c_lo, c_hi = np.minimum(c_lo, loc - k * sd), np.maximum(c_hi, loc + k * sd)
        t = composite_grid(c_lo, c_hi, s_lo, s_hi, p, p // 4)
        out = np.empty(len(loc))
        pos = sd > 0
        if pos.any():
            r = np.flatnonzero(pos)
            F = self.cdf(t[r], r)
            en.check_tails(F, tol, f"{type(self).__name__} forecast")
            G = ndtr((t[r] - loc[r, None]) / sd[r, None])
            out[r] = en.excess(F, G, t[r])
        if (~pos).any():
            r = np.flatnonzero(~pos)
            y = loc[r]
            lo, hi = np.minimum(t[r, 0], y), np.maximum(t[r, -1], y)
            n_side = 2 * p + 1                                   # odd: Simpson on equal pairs
            tL, tR = span(lo, y, n_side), span(y, hi, n_side)
            FL, FR = self.cdf(tL, r), self.cdf(tR, r)
            en.check_tails(np.concatenate([FL[:, :1], FR[:, -1:]], axis=1), tol,
                           f"{type(self).__name__} forecast")
            out[r] = simpson(FL * FL, x=tL, axis=-1) + simpson((1.0 - FR) ** 2, x=tR, axis=-1)
        return out


# ---------------------------------------------------------------- Gaussian mixture (heteroscedastic GP)


class GaussianMixtureForecast(Forecast):
    def __init__(self, mu, sigma, w, k_sd=10.0):
        self.mu, self.sigma = np.asarray(mu, float), np.asarray(sigma, float)
        self.w = np.broadcast_to(np.asarray(w, float), self.mu.shape)
        self.n_query = self.mu.shape[0]
        self.k_sd = k_sd
        self._e = None

    def cdf(self, t, rows=None, chunk_elems=8_000_000):
        mu, sg, w = (a if rows is None else a[rows] for a in (self.mu, self.sigma, self.w))
        out = np.empty(t.shape)
        step = max(1, chunk_elems // (mu.shape[1] * t.shape[1]))
        for i in range(0, t.shape[0], step):
            sl = slice(i, i + step)
            z = (t[sl, None, :] - mu[sl, :, None]) / sg[sl, :, None]
            out[sl] = np.einsum("qm,qmp->qp", w[sl], ndtr(z))
        return out

    def support(self):
        return ((self.mu - self.k_sd * self.sigma).min(1), (self.mu + self.k_sd * self.sigma).max(1))

    def entropy(self):
        if self._e is None:
            self._e = en.gm_entropy(self.mu, self.sigma, self.w)
        return self._e

    def excess_vs_normal(self, loc, sd, proto=None):
        return en.gm_excess_vs_normal(self.mu, self.sigma, self.w, loc, sd, self.entropy())

    def crps(self, y, proto=None):
        return en.gm_crps(self.mu, self.sigma, self.w, y, self.entropy())


class GridOnly(Forecast):
    """Wraps a forecast and drops its closed forms, forcing the generic grid path (tests only)."""

    def __init__(self, fc):
        self.fc, self.n_query = fc, fc.n_query

    def cdf(self, t, rows=None):
        return self.fc.cdf(t, rows)

    def support(self):
        return self.fc.support()

    def core(self):
        return self.fc.core()


# ---------------------------------------------------------------- quantile head (TabICLv2)


class QuantileForecast(Forecast):
    """CDF piecewise-linear through (q_k, alpha_k), 0 below q_0 and 1 above q_{K-1}."""

    def __init__(self, qs, alphas):
        self.qs = np.sort(np.asarray(qs, float), axis=1)
        self.alphas = np.asarray(alphas, float)
        self.n_query = self.qs.shape[0]

    def cdf(self, t, rows=None):
        idx = np.arange(self.n_query) if rows is None else np.asarray(rows)
        return np.stack([np.interp(t[i], self.qs[j], self.alphas, left=0.0, right=1.0)
                         for i, j in enumerate(idx)])

    def support(self):
        lo, hi = self.qs[:, 0], self.qs[:, -1]
        pad = 1e-6 * np.maximum(hi - lo, 1e-12)
        return lo - pad, hi + pad

    def entropy(self):
        return en.quantile_entropy(self.qs, self.alphas)
