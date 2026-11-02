"""Sparse heteroskedastic GP, PyMC + numpyro NUTS (phase0_protocol.md §5.2). T1 only (1-D input).

Follows the PyMC example "Heteroskedastic Gaussian Processes", section "Sparse Heteroskedastic GP"
(https://www.pymc.io/projects/examples/en/latest/gaussian_processes/GP-Heteroskedastic.html):
two latent GPs, the mean mu(x) and the log noise s.d. lg_sigma(x), each a DTC sparse latent GP over
inducing points Xu with whitened inducing values,
    u = L v,  v ~ N(0, I),  L = chol(K_uu + jitter I),   f(X) = K_fu K_uu^{-1} u,
    k = eta^2 ExpQuad(ell),  ell ~ InverseGamma(mu=l_mu, sigma=l_sigma),  eta ~ Gamma(2, 1),
    y ~ N(mu(X), exp(m_g + lg_sigma(X))^2),  NUTS with target_accept = 0.95.
mu has a zero mean function; the log-noise latent has a learned constant mean m_g ~ Normal(log_noise_mean_prior)
(standardized-y units), so data-free regions revert to the task's typical noise level instead of to
exp(0) = 1 s.d. of y with the large amplitude a noiseless task forces on a zero-mean g (protocol §5.2).
l_mu, l_sigma come from the pairwise distances of the context inputs (the example's get_l_prior), so
the priors are generic and not matched to the toy function. Inducing points are evenly spaced on x_range
(the example takes every other input; a fixed grid keeps the cost flat in n). y is standardized with
context statistics and the members are mapped back to y units.

Prediction is the example's `conditional`: per posterior draw, mean K_su K_uu^{-1} u plus a draw from
the conditional variance diag(K_ss - Q_ss). Only marginals at each query enter a member CDF, so the
diagonal is exact here. The example reads epistemic uncertainty off the posterior spread of mu and
aleatoric off sigma^2; the CRPS projection of the same members is
    members N(mu_s(x*), sigma_s(x*)^2), w_s = 1/S:  A_hat = mean_s sigma_s/sqrt(pi),
    e_total = e_CRPS(mixture),  E_hat = e_total - A_hat.
Toy rung 0 pushes lg_sigma towards -inf: sigma_s is clipped at sigma_floor and the clip rate reported.
"""
import time

import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial.distance import pdist

from adapters.base import UQOutput, gaussian_members


def lengthscale_prior(x):
    """The example's get_l_prior: InverseGamma(mu, sigma) from the spread of pairwise input distances."""
    d = pdist(np.asarray(x, float)[:, None])
    d = d[d != 0]
    lo = d.min() if d.size else 0.1
    hi = d.max() if d.size else 1.0
    sd = max(0.1, (hi - lo) / 6.0)
    return lo + 3.0 * sd, sd


def inducing_points(x_range, hcfg):
    return np.linspace(*x_range, hcfg["n_inducing"])


def expquad_np(a, b, ell, eta):
    return eta**2 * np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / ell**2)


def conditional_np(x_new, xu, ell, eta, v, jitter, z):
    """Marginal DTC conditional at x_new for one draw (ell, eta, whitened v); z ~ N(0, I) per query."""
    L = np.linalg.cholesky(expquad_np(xu, xu, ell, eta) + jitter * np.eye(len(xu)))
    Kuiu = solve_triangular(L.T, v, lower=False)          # K_uu^{-1} u with u = L v
    Ksu = expquad_np(x_new, xu, ell, eta)
    mean = Ksu @ Kuiu
    tmp = solve_triangular(L, Ksu.T, lower=True)
    var = eta**2 + jitter - np.sum(tmp * tmp, axis=0)     # diag(K_ss + jitter - Q_ss)
    return mean + np.sqrt(np.maximum(var, 0.0)) * z


def predict_members(x_new, xu, draws, jitter, rng):
    """(mu_std, lg_sigma_std), each (q, S), from flattened posterior draws. lg_sigma_std includes m_g."""
    S, q = len(draws["mu_ell"]), len(x_new)
    mu, lg = np.empty((q, S)), np.empty((q, S))
    for s in range(S):
        mu[:, s] = conditional_np(x_new, xu, draws["mu_ell"][s], draws["mu_eta"][s],
                                  draws["u_rotated_mu"][s], jitter, rng.standard_normal(q))
        lg[:, s] = draws["m_g"][s] + conditional_np(x_new, xu, draws["sigma_ell"][s], draws["sigma_eta"][s],
                                                    draws["u_rotated_lg_sigma"][s], jitter, rng.standard_normal(q))
    return mu, lg


def to_output(mu_std, lg_sigma_std, y_mean, y_sd, sigma_floor, fit_seconds=np.nan, extras=None):
    """Conversion only: standardized-scale member draws (q, S) -> UQOutput in y units."""
    mu = y_mean + y_sd * np.asarray(mu_std, float)
    sigma = y_sd * np.exp(np.asarray(lg_sigma_std, float))
    S = mu.shape[1]
    A, e, E, fc, clip = gaussian_members(mu, sigma, np.full(S, 1.0 / S), sigma_floor)
    return UQOutput(A, E, e, fc, fit_seconds, dict(extras or {}), sigma_clip_rate=clip)


def fit(x_ctx, ys, seed, x_range, hcfg):
    import pymc as pm
    import pytensor.tensor as pt

    X = np.asarray(x_ctx, float)[:, None]
    Xu = inducing_points(x_range, hcfg)[:, None]
    l_mu, l_sd = lengthscale_prior(x_ctx)
    jitter = hcfg["jitter"]

    def sparse_latent(name, cov):
        # The example's SparseLatent.prior
        Kuu = cov(Xu)
        L = pt.linalg.cholesky(Kuu + jitter * pt.eye(Xu.shape[0]))
        v = pm.Normal(f"u_rotated_{name}", mu=0.0, sigma=1.0, shape=Xu.shape[0])
        u = pt.dot(L, v)
        Kuiu = pt.linalg.solve_triangular(L.T, pt.linalg.solve_triangular(L, u, lower=True), lower=False)
        return pt.dot(cov(X, Xu), Kuiu)

    with pm.Model():
        mu_ell = pm.InverseGamma("mu_ell", mu=l_mu, sigma=l_sd)
        mu_eta = pm.Gamma("mu_eta", alpha=2.0, beta=1.0)
        sigma_ell = pm.InverseGamma("sigma_ell", mu=l_mu, sigma=l_sd)
        sigma_eta = pm.Gamma("sigma_eta", alpha=2.0, beta=1.0)
        mu_f = sparse_latent("mu", mu_eta**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=mu_ell))
        lg_sigma_f = sparse_latent("lg_sigma", sigma_eta**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=sigma_ell))
        m_g_mu, m_g_sd = hcfg["log_noise_mean_prior"]
        m_g = pm.Normal("m_g", mu=m_g_mu, sigma=m_g_sd)
        pm.Normal("lik", mu=mu_f, sigma=pt.exp(m_g + lg_sigma_f), observed=ys)
        idata = pm.sample(draws=hcfg["draws"], tune=hcfg["tune"], chains=hcfg["chains"],
                          target_accept=hcfg["target_accept"], nuts_sampler="numpyro",
                          random_seed=int(seed), progressbar=False, compute_convergence_checks=False)
    return idata, (l_mu, l_sd)


NAMES = ("mu_ell", "mu_eta", "sigma_ell", "sigma_eta", "m_g", "u_rotated_mu", "u_rotated_lg_sigma")


def _diagnostics(idata):
    import arviz_stats
    post = idata["posterior"]
    rh = [np.nanmax(np.atleast_1d(arviz_stats.rhat(np.asarray(post[k].values), chain_axis=0, draw_axis=1)))
          for k in NAMES]
    div = int(np.asarray(idata["sample_stats"]["diverging"].values).sum())
    return float(np.nanmax(rh)), div


def run(task, n, rng, proto, mcfg):
    x, y = task.context(n)
    y_mean, y_sd = float(np.mean(y)), float(np.std(y))
    assert y_sd > 0, "constant context"
    x_range = task.tcfg["x_range"]
    t0 = time.perf_counter()
    idata, (l_mu, l_sd) = fit(x, (y - y_mean) / y_sd, int(rng.integers(2**31 - 1)), x_range, mcfg)
    post = idata["posterior"]
    draws = {k: np.asarray(post[k].values).reshape(-1, *np.asarray(post[k].values).shape[2:]) for k in NAMES}
    assert len(draws["mu_ell"]) == proto["S"], (len(draws["mu_ell"]), proto["S"])
    mu_std, lg_std = predict_members(task.x_qry, inducing_points(x_range, mcfg), draws, mcfg["jitter"], rng)
    fit_seconds = time.perf_counter() - t0
    rhat, div = _diagnostics(idata)
    extras = {"rhat_max": rhat, "divergences": div, "rhat_flag": bool(rhat > mcfg["rhat_flag"]),
              "l_prior_mu": l_mu, "l_prior_sigma": l_sd, "y_mean": y_mean, "y_sd": y_sd,
              "m_g_mean": float(np.mean(draws["m_g"])),
              "mu_std": mu_std, "lg_sigma_std": lg_std,
              **{f"post_{k}": v for k, v in draws.items()}}
    return to_output(mu_std, lg_std, y_mean, y_sd, proto["sigma_floor"], fit_seconds, extras)
