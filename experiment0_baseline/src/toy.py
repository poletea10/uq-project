"""T1 toy set (phase0_protocol.md §2): one curated 1-D function, three noise settings, one nested context.

    y = f(x) + sigma_k(x) * eps,   eps ~ N(0, 1),   f = f_base + f_gap,

    f_base(x) = Σ_j theta_j exp(-(x - c_j)^2 / (2 l^2)),                      theta ~ N(0, s_theta^2 I_J),
    f_gap(x)  = h_[a,b](x) Σ_k eta_k exp(-(x - c~_k)^2 / (2 l_g^2)),          eta ~ N(0, s_eta^2 I_K),
    h_[a,b](x) = logistic((x - a)/w) - logistic((x - b)/w).

s_theta makes the RMS prior s.d. of f_base over x_range equal f_std; s_eta makes the RMS prior s.d. of
f_gap over the sparse strip equal gap_std. Both are computed on a fine grid of the region.

Toy rungs differ only in sigma_k(x): 0 (rung 0), sigma_0 (rung 1), sigma_0 exp(phi_b h_noise(x)) (rung 2).

RNG streams (§9): (theta, eta) from [base, toy_seed, 0]; context x from [base, toy_seed, 1]; eps from
[base, toy_seed, 2]. Context x and eps are drawn once at n_max and shared across toy rungs (common random
numbers); each n uses the first n points (nested contexts). Queries are n_query points evenly spaced on
x_range, shared by every rung and model.
"""
from dataclasses import dataclass

import numpy as np

import config

N_FINE = 2001       # fine grid for the prior scales and the curation statistics


# ---------------------------------------------------------------- building blocks


def logistic(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def strip_indicator(x, a, b, w):
    """h(x) = logistic((x-a)/w) - logistic((x-b)/w): smooth indicator of [a, b]."""
    x = np.asarray(x, float)
    return logistic((x - a) / w) - logistic((x - b) / w)


def _bumps(x, centres, width):
    return np.exp(-((np.asarray(x, float)[:, None] - centres[None, :]) ** 2) / (2.0 * width**2))


def base_centres(tcfg):
    lo, hi = tcfg["x_range"]
    pad = 2.0 * tcfg["bump_width"]
    return np.linspace(lo - pad, hi + pad, tcfg["J"])


def base_basis(x, tcfg):
    """(len(x), J) matrix of the wide bumps b_j(x)."""
    return _bumps(x, base_centres(tcfg), tcfg["bump_width"])


def gap_basis(x, tcfg):
    """(len(x), K) matrix of the windowed narrow bumps h(x) b~_k(x)."""
    a, b = tcfg["sparse_strip"]
    h = strip_indicator(x, a, b, tcfg["gap_edge_width"])
    return h[:, None] * _bumps(x, np.linspace(a, b, tcfg["K"]), tcfg["gap_bump_width"])


def _rms_scale(basis_fn, xs, target, tcfg):
    """s such that sqrt(mean_x Var f(x)) = target, with Var f(x) = s^2 Σ_j basis_j(x)^2."""
    return target / np.sqrt(np.mean(np.sum(basis_fn(xs, tcfg) ** 2, axis=1)))


def theta_scale(tcfg):
    return _rms_scale(base_basis, np.linspace(*tcfg["x_range"], N_FINE), tcfg["f_std"], tcfg)


def eta_scale(tcfg):
    return _rms_scale(gap_basis, np.linspace(*tcfg["sparse_strip"], N_FINE), tcfg["gap_std"], tcfg)


# ---------------------------------------------------------------- the function


@dataclass
class ToyFunction:
    seed: int
    theta: np.ndarray       # (J,)
    eta: np.ndarray         # (K,)
    tcfg: dict

    def base(self, x):
        return base_basis(x, self.tcfg) @ self.theta

    def gap(self, x):
        return gap_basis(x, self.tcfg) @ self.eta

    def __call__(self, x):
        return self.base(x) + self.gap(x)


def draw_function(seed, tcfg, base_seed):
    rng = np.random.default_rng([base_seed, seed, 0])
    theta = rng.normal(0.0, theta_scale(tcfg), tcfg["J"])
    eta = rng.normal(0.0, eta_scale(tcfg), tcfg["K"])
    return ToyFunction(int(seed), theta, eta, tcfg)


def curation_stats(fn):
    """The three pre-registered curation statistics of §2.2, on fine grids."""
    tcfg = fn.tcfg
    xs = np.linspace(*tcfg["x_range"], N_FINE)
    x_gap = np.linspace(*tcfg["sparse_strip"], N_FINE)
    x_noise = np.linspace(*tcfg["noise_strip"], N_FINE)
    f_noise = fn(x_noise)
    return {"max_abs_f": float(np.max(np.abs(fn(xs)))),
            "gap_sd": float(np.std(fn.gap(x_gap))),
            "noise_strip_range": float(f_noise.max() - f_noise.min())}


def passes_curation(stats, tcfg):
    c = tcfg["curation"]
    return (stats["max_abs_f"] <= c["max_abs_f"] and stats["gap_sd"] >= c["min_gap_sd"]
            and stats["noise_strip_range"] >= c["min_noise_strip_range"])


# ---------------------------------------------------------------- noise and context


def sigma_fn(x, rcfg, tcfg):
    x = np.asarray(x, float)
    if rcfg["noise"] == "none":
        return np.zeros_like(x)
    if rcfg["noise"] == "constant":
        return np.full_like(x, np.exp(rcfg["log_sigma0"]))
    if rcfg["noise"] == "bump":
        a, b = tcfg["noise_strip"]
        return np.exp(rcfg["log_sigma0"] + rcfg["phi_b"] * strip_indicator(x, a, b, rcfg["edge_width"]))
    raise ValueError(rcfg["noise"])


def sample_context_x(rng, size, tcfg):
    """Piecewise-uniform on x_range: relative density gap_density on sparse_strip, 1 elsewhere."""
    lo, hi = tcfg["x_range"]
    s0, s1 = tcfg["sparse_strip"]
    edges = np.array([lo, s0, s1, hi])
    dens = np.array([1.0, tcfg["gap_density"], 1.0])
    mass = dens * np.diff(edges)
    cum = np.concatenate([[0.0], np.cumsum(mass)]) / mass.sum()
    u = rng.uniform(0.0, 1.0, size)
    k = np.clip(np.searchsorted(cum, u, side="right") - 1, 0, 2)
    frac = (u - cum[k]) / (cum[k + 1] - cum[k])
    return edges[k] + frac * (edges[k + 1] - edges[k])


def in_strip(x, strip):
    a, b = strip
    return (np.asarray(x) >= a) & (np.asarray(x) <= b)


# ---------------------------------------------------------------- task


@dataclass
class ToyTask:
    rung: int
    fn: ToyFunction
    x_ctx: np.ndarray       # (n_max,) nested: context of size n is the first n points
    eps: np.ndarray         # (n_max,) standard-normal draws, shared across rungs
    x_qry: np.ndarray       # (n_query,) evenly spaced on x_range
    rcfg: dict
    tcfg: dict

    def f(self, x):
        return self.fn(x)

    def sigma(self, x):
        return sigma_fn(x, self.rcfg, self.tcfg)

    @property
    def y_ctx(self):
        return self.f(self.x_ctx) + self.sigma(self.x_ctx) * self.eps

    def context(self, n):
        assert n <= len(self.x_ctx), (n, len(self.x_ctx))
        return self.x_ctx[:n], self.y_ctx[:n]

    @property
    def f_qry(self):
        return self.f(self.x_qry)

    @property
    def sigma_qry(self):
        return self.sigma(self.x_qry)

    def query_regions(self):
        """(in_noise_strip, in_sparse_strip) per query. The strips are disjoint."""
        return in_strip(self.x_qry, self.tcfg["noise_strip"]), in_strip(self.x_qry, self.tcfg["sparse_strip"])


def make_task(rung_k, fn, proto, tcfg):
    base, seed = proto["base_seed"], fn.seed
    n_max = max(proto["n_grid"])
    x_ctx = sample_context_x(np.random.default_rng([base, seed, 1]), n_max, tcfg)
    eps = np.random.default_rng([base, seed, 2]).standard_normal(n_max)
    x_qry = np.linspace(*tcfg["x_range"], tcfg["n_query"])
    return ToyTask(rung_k, fn, x_ctx, eps, x_qry, config.toy_rung(rung_k), tcfg)


def load_task(rung_k, proto=None, tcfg=None):
    """The frozen toy task of one rung. Needs toy_seed recorded by scripts/00_curate_toy.py."""
    proto = proto or config.protocol()
    tcfg = tcfg or config.toy_function()
    assert tcfg["toy_seed"] is not None, "toy_seed not set: run scripts/00_curate_toy.sh first"
    return make_task(rung_k, draw_function(tcfg["toy_seed"], tcfg, proto["base_seed"]), proto, tcfg)
