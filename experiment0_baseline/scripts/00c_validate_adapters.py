"""Kernel unit tests (phase0_protocol.md §4.3) and adapter conversion tests (§10, gotcha 2). No model fits.

Every conversion is fed synthetic native outputs with closed-form answers, on several query rows with
distinct parameters, so an axis mix-up fails. Tolerances: 1e-6 relative on A_hat and e_total; 1e-6 of
e_total absolute on E_hat. Monte Carlo checks pass within 4 standard errors. Non-zero exit on any FAIL.
Adapters covered: heteroscedastic GP, TabICLv2 (total only). Dec-ICL joins when its adapter exists.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from scipy import stats
from scipy.special import ndtr

import config
import entropy as en
from adapters import hgp, tabiclv2
from forecasts import GaussianMixtureForecast, GridOnly, QuantileForecast

FAILS = []
RNG = np.random.default_rng(2026)
PROTO = config.protocol()
REL, MC_Z = 1e-6, 4.0


def check(name, ok, detail=""):
    print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def rel_err(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)) / np.abs(np.asarray(b))))


def gauss_member_grid(mu, sigma, p=8192, k=12.0):
    """Independent reference: member CDFs on a dense grid, projected with entropy.project."""
    t = en.make_grid((mu - k * sigma).min(1), (mu + k * sigma).max(1), p)
    F = ndtr((t[:, None, :] - mu[:, :, None]) / sigma[:, :, None])
    return en.project(F, np.full(mu.shape[1], 1.0 / mu.shape[1]), t)


def check_output(name, out, A, e, E_abs_zero=False):
    check(f"{name}: A_hat", rel_err(out.A_hat, A) < REL, f"(rel {rel_err(out.A_hat, A):.1e})")
    check(f"{name}: e_total", rel_err(out.e_total, e) < REL, f"(rel {rel_err(out.e_total, e):.1e})")
    E_ref = np.zeros_like(e) if E_abs_zero else e - A
    err = np.max(np.abs(out.E_hat - E_ref) / out.e_total)
    check(f"{name}: E_hat", err < REL, f"(max |err|/e_total {err:.1e})")


# ================================================================ entropy kernel (§4.3)


def test_kernel():
    sig = np.array([0.05, 1.0, 7.0])
    t = en.make_grid(-10 * sig, 10 * sig, PROTO["t_points"])
    e = en.e_crps(ndtr(t / sig[:, None]), t)
    check("e_CRPS(N(0, s^2)) = s/sqrt(pi) on the shared grid size", rel_err(e, sig / en.SQRT_PI) < REL,
          f"(rel {rel_err(e, sig / en.SQRT_PI):.1e})")

    # Two-component mixture: grid vs folded-normal closed form vs Monte Carlo ½E|Y - Y'|.
    mu = np.array([[-1.0, 2.0], [0.0, 0.3], [5.0, 5.5]])
    sd = np.array([[0.5, 1.5], [0.2, 0.1], [2.0, 0.7]])
    w = np.array([[0.3, 0.7], [0.5, 0.5], [0.9, 0.1]])
    tg = en.make_grid((mu - 12 * sd).min(1), (mu + 12 * sd).max(1), 8192)
    Fm = ndtr((tg[:, None, :] - mu[:, :, None]) / sd[:, :, None])
    e_grid = en.e_crps(np.einsum("qm,qmp->qp", w, Fm), tg)
    e_closed = en.gm_entropy(mu, sd, w)
    check("mixture e: grid vs folded-normal closed form", rel_err(e_grid, e_closed) < REL,
          f"(rel {rel_err(e_grid, e_closed):.1e})")
    N = 2_000_000
    ok = True
    for i in range(3):
        c1, c2 = RNG.random(N) < w[i, 1], RNG.random(N) < w[i, 1]
        y1 = np.where(c1, RNG.normal(mu[i, 1], sd[i, 1], N), RNG.normal(mu[i, 0], sd[i, 0], N))
        y2 = np.where(c2, RNG.normal(mu[i, 1], sd[i, 1], N), RNG.normal(mu[i, 0], sd[i, 0], N))
        h = 0.5 * np.abs(y1 - y2)
        ok &= abs(h.mean() - e_closed[i]) < MC_Z * h.std() / np.sqrt(N)
    check("mixture e: closed form vs Monte Carlo ½E|Y-Y'|", ok)

    # Excess against a point: the CRPS of a Gaussian forecast, closed form. The excess kernel runs on
    # two grids that meet at y, against the step's two constant sides (0 left of y, 1 right of it).
    m, s, y = np.array([0.0, 1.0, -2.0]), np.array([1.0, 0.3, 2.5]), np.array([0.5, 1.0, 3.0])
    z = (y - m) / s
    crps_closed = s * (z * (2 * ndtr(z) - 1) + 2 * stats.norm.pdf(z) - 1 / en.SQRT_PI)
    tL = en.make_grid(np.minimum(m - 10 * s, y - 1), y, 200_001)
    tR = en.make_grid(y, np.maximum(m + 10 * s, y + 1), 200_001)
    FL, FR = (ndtr((tt - m[:, None]) / s[:, None]) for tt in (tL, tR))
    crps_g = en.excess(FL, np.zeros_like(FL), tL) + en.excess(FR, np.ones_like(FR), tR)
    check("excess kernel vs closed-form Gaussian CRPS at a point (grids split at y)",
          rel_err(crps_g, crps_closed) < 1e-6, f"(rel {rel_err(crps_g, crps_closed):.1e})")
    gm = GaussianMixtureForecast(m[:, None], s[:, None], np.ones(1))
    check("Gaussian-mixture CRPS closed form", rel_err(gm.crps(y), crps_closed) < 1e-12)
    grid_pm = GridOnly(gm).excess_vs_normal(y, np.zeros(3), PROTO)
    check("point-mass truth on the shared grid (split at y, Simpson)", rel_err(grid_pm, crps_closed) < REL,
          f"(rel {rel_err(grid_pm, crps_closed):.1e})")
    loc, sdt = y, np.array([0.4, 0.05, 1.0])
    E_g = GridOnly(gm).excess_vs_normal(loc, sdt, PROTO)
    E_c = gm.excess_vs_normal(loc, sdt)
    check("Gaussian truth on the shared grid vs closed form", rel_err(E_g, E_c) < REL, f"(rel {rel_err(E_g, E_c):.1e})")

    # Quantile inversion (figure bands): Forecast.quantiles of a Gaussian mixture vs scipy.
    levels = (0.05, 0.5, 0.95)
    q_got = gm.quantiles(levels, PROTO)
    q_ref = np.stack([stats.norm.ppf(lv, m, s) for lv in levels], axis=1)
    err = np.max(np.abs(q_got - q_ref) / s[:, None])
    check("Forecast.quantiles inverts a Gaussian CDF", err < 1e-3, f"(max |err|/s {err:.1e})")


# ================================================================ conversions (§10, gotcha 2)


def test_hgp():
    q, S = 4, 50
    y_mean, y_sd = 3.0, 2.0
    mu_std = RNG.normal(0, 1, (q, S)) * np.array([0.01, 0.3, 1.0, 0.0])[:, None]
    lg_std = np.log(np.array([0.2, 0.5, 1.0, 1.5]))[:, None] + RNG.normal(0, 0.2, (q, S))
    out = hgp.to_output(mu_std, lg_std, y_mean, y_sd, PROTO["sigma_floor"])
    mu, sig = y_mean + y_sd * mu_std, y_sd * np.exp(lg_std)
    A_g, e_g, _ = gauss_member_grid(mu, sig)
    check("hgp: A_hat = y_sd * mean(exp(lg_sigma)) / sqrt(pi)", rel_err(out.A_hat, sig.mean(1) / en.SQRT_PI) < 1e-12)
    check_output("hgp mixture vs grid projection", out, A_g, e_g)

    # M identical members N(mu, s^2): A_hat = e_total = s/sqrt(pi), E_hat = 0.
    s_id = y_sd * np.exp(lg_std[:, 0])
    same = hgp.to_output(np.repeat(mu_std[:, :1], S, 1), np.repeat(lg_std[:, :1], S, 1), y_mean, y_sd,
                         PROTO["sigma_floor"])
    check_output("hgp identical members", same, s_id / en.SQRT_PI, s_id / en.SQRT_PI, E_abs_zero=True)

    # sigma -> 0 (toy rung 0): clipped at sigma_floor and reported.
    tiny = hgp.to_output(mu_std, np.full((q, S), -40.0), y_mean, y_sd, PROTO["sigma_floor"])
    check("hgp sigma below the floor clipped and reported", tiny.sigma_clip_rate == 1.0
          and np.allclose(tiny.A_hat, PROTO["sigma_floor"] / en.SQRT_PI))

    # DTC conditional: triangular solves vs a dense reference.
    xu = np.linspace(-3, 3, 40)
    xn = RNG.uniform(-3.5, 3.5, 30)
    ell, eta, jit = 0.6, 1.3, 1e-6
    v = RNG.normal(0, 1, 40)
    got = hgp.conditional_np(xn, xu, ell, eta, v, jit, np.zeros(30))
    Kuu = hgp.expquad_np(xu, xu, ell, eta) + jit * np.eye(40)
    u = np.linalg.cholesky(Kuu) @ v
    Ksu = hgp.expquad_np(xn, xu, ell, eta)
    ref = Ksu @ np.linalg.solve(Kuu, u)
    check("hgp DTC conditional mean vs dense solve", np.max(np.abs(got - ref)) < 1e-6 * np.max(np.abs(ref)))
    var_ref = eta**2 + jit - np.einsum("ij,ji->i", Ksu, np.linalg.solve(Kuu, Ksu.T))
    got_sd = hgp.conditional_np(xn, xu, ell, eta, v, jit, np.ones(30)) - got
    check("hgp DTC conditional variance vs dense solve",
          np.max(np.abs(got_sd**2 - np.maximum(var_ref, 0))) < 1e-6 * eta**2)


def test_tabiclv2():
    K = 999
    alphas = tabiclv2.quantile_levels(K)
    check("tabiclv2 quantile levels are k/(K+1)", np.allclose(alphas, np.arange(1, K + 1) / (K + 1), rtol=0, atol=1e-15))
    # Distinct rows: normals and a skewed t, so an axis mix-up fails.
    qs = np.stack([stats.norm.ppf(alphas, 1.0, 2.0), stats.norm.ppf(alphas, -3.0, 0.1),
                   stats.t.ppf(alphas, 3, -1.0, 0.5)])
    out = tabiclv2.to_output(qs)
    check("tabiclv2 is total only", out.A_hat is None and out.E_hat is None)
    tq = en.make_grid(qs[:, 0] - 1, qs[:, -1] + 1, 400_001)
    e_qg = en.e_crps(out.cdf.cdf(tq), tq)
    check("tabiclv2 e_total: exact segment integral vs grid", rel_err(out.e_total, e_qg) < REL,
          f"(rel {rel_err(out.e_total, e_qg):.1e})")
    check("tabiclv2 forecast CDF hits each level at its quantile",
          np.max(np.abs(out.cdf.cdf(qs) - alphas[None, :])) < 1e-12)
    print(f"     999-quantile N(1, 2^2) entropy {out.e_total[0]:.5f} vs exact {2 / en.SQRT_PI:.5f} (tail truncation)")
    shuffled = tabiclv2.to_output(qs[:, RNG.permutation(K)])
    check("tabiclv2 sorts unsorted quantile rows", np.allclose(shuffled.e_total, out.e_total, rtol=1e-12))

    # Truth side on a quantile forecast: Gaussian and point-mass truths, against an independent dense grid.
    loc, sd = np.array([0.5, -3.05, -0.5]), np.array([0.3, 0.02, 0.0])
    E_got = out.cdf.excess_vs_normal(loc, sd, PROTO)
    tt = en.make_grid(np.minimum(qs[:, 0], loc - 10 * sd) - 1, np.maximum(qs[:, -1], loc + 10 * sd) + 1, 2_000_001)
    G = np.where(sd[:, None] > 0, ndtr((tt - loc[:, None]) / np.where(sd > 0, sd, 1.0)[:, None]),
                 (tt >= loc[:, None]).astype(float))
    E_ref = en.excess(out.cdf.cdf(tt), G, tt)
    # The shared grid (4096 core points) integrates across the 999 kinks of the piecewise-linear CDF.
    check("tabiclv2 E(q_m) on the shared grid vs dense reference", rel_err(E_got, E_ref) < 1e-3,
          f"(rel {rel_err(E_got, E_ref):.1e})")


def main():
    for name, fn in [("kernel", test_kernel), ("hgp", test_hgp), ("tabiclv2", test_tabiclv2)]:
        print(f"--- {name}", flush=True)
        fn()
    print("ALL OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
