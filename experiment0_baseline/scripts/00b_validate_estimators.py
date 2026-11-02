"""Truth-code audit (phase0_protocol.md §10, gotcha 1). Toy rungs 0-2 of the curated function, 50 queries.

A: with 1e5 replicate pairs, ½|y1 - y2| averaged per query must match sigma(x*)/sqrt(pi) (exactly 0 on
   toy rung 0, a flat line on toy rung 1, the noise bump on toy rung 2).
E: for a fixed, deliberately wrong Gaussian forecast, the replicate estimator of §4.2 with R = 100,
   repeated 1000 times, must average to the closed-form ∫(F_m - Φ_tau)^2. Also reports its per-query
   s.d. at R = 100, the Monte Carlo noise T2 and T3 will carry. The closed form is cross-checked against
   the grid kernel.
"Within 3 Monte Carlo standard errors" is applied family-wise: each check passes if every query's
|z| is below the Bonferroni threshold for 50 queries at family-wise level 0.0027 (the 3-SE level of
one test); the share of queries inside a plain 3 SE is printed alongside.
Writes outputs/figures/00b_estimators.png. Non-zero exit on any FAIL.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import config
import toy
import truth
from forecasts import GaussianMixtureForecast, GridOnly

FAILS = []
N_Q, PAIRS, R, REPS = 50, 100_000, 100, 1000
Z_FW = norm.ppf(1.0 - 0.0027 / (2 * N_Q))


def check(name, ok, detail=""):
    print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def main():
    proto = config.protocol()
    rng = np.random.default_rng([proto["base_seed"], 9])
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
    for c, k in enumerate(config.TOY_RUNGS):
        task = toy.load_task(k, proto)
        idx = np.linspace(0, len(task.x_qry) - 1, N_Q).round().astype(int)
        x, f, s = task.x_qry[idx], task.f_qry[idx], task.sigma_qry[idx]
        A_true = truth.aleatoric(s)

        # ---------------- A
        y1 = f[:, None] + s[:, None] * rng.standard_normal((N_Q, PAIRS))
        y2 = f[:, None] + s[:, None] * rng.standard_normal((N_Q, PAIRS))
        h = 0.5 * np.abs(y1 - y2)
        est, se = h.mean(1), h.std(1, ddof=1) / np.sqrt(PAIRS)
        if k == 0:
            check("toy rung0 A estimator exactly 0", np.all(est == 0.0) and np.all(A_true == 0.0))
        else:
            z = (est - A_true) / se
            check(f"toy rung{k} A estimator vs sigma/sqrt(pi)", np.all(np.abs(z) < Z_FW),
                  f"(max|z| {np.abs(z).max():.2f} < {Z_FW:.2f}; {np.mean(np.abs(z) < 3):.0%} within 3 SE)")
        ax = axes[0, c]
        ax.plot(x, A_true, "b-", lw=1.2, label="sigma/sqrt(pi)")
        ax.errorbar(x, est, yerr=3 * se, fmt="r.", ms=3, lw=0.8, label="½|y1-y2|, 1e5 pairs ±3SE")
        ax.set_title(f"toy rung {k}: A"); ax.legend(fontsize=7)

        # ---------------- E (deliberately wrong Gaussian forecast)
        mu_m = f + 0.4 + 0.3 * np.sin(x)
        sd_m = 0.6 * s + 0.15
        gm = GaussianMixtureForecast(mu_m[:, None], sd_m[:, None], np.ones(1))
        E_closed = truth.epistemic(gm, f, s, proto)
        E_grid = truth.epistemic(GridOnly(gm), f, s, proto)
        rel = np.max(np.abs(E_grid - E_closed) / E_closed)
        check(f"toy rung{k} E closed form vs grid kernel", rel < 1e-5, f"(max rel diff {rel:.1e})")

        est_reps = np.empty((N_Q, REPS))
        for r in range(REPS):
            Y = f[:, None] + s[:, None] * rng.standard_normal((N_Q, R))
            est_reps[:, r] = truth.replicate_epistemic(gm, Y, proto)
        m, sd = est_reps.mean(1), est_reps.std(1, ddof=1)
        if k > 0:
            z = (m - E_closed) / (sd / np.sqrt(REPS))
            check(f"toy rung{k} replicate E estimator unbiased (R={R}, {REPS} reps)", np.all(np.abs(z) < Z_FW),
                  f"(max|z| {np.abs(z).max():.2f}; {np.mean(np.abs(z) < 3):.0%} within 3 SE)")
        print(f"     toy rung{k}: per-query s.d. of E_hat at R={R}: median {np.median(sd):.3g} "
              f"(median E {np.median(E_closed):.3g}, median rel. s.d. {np.median(sd / E_closed):.2f})")
        if k == 0:
            # Point-mass truth: replicates are all equal to f, so A_hat = 0 and E_hat = CRPS(q, f) exactly.
            check("toy rung0 replicate E == closed form exactly", np.allclose(est_reps[:, 0], E_closed, rtol=1e-12))
        ax = axes[1, c]
        ax.plot(x, E_closed, "b-", lw=1.2, label="closed form ∫(F-Φ)²")
        ax.errorbar(x, m, yerr=3 * sd / np.sqrt(REPS), fmt="r.", ms=3, lw=0.8, label=f"replicate est. R={R}")
        ax.fill_between(x, m - sd, m + sd, color="r", alpha=0.1, lw=0, label="±1 s.d. single estimate")
        ax.set_title(f"toy rung {k}: E of a wrong forecast"); ax.set_xlabel("x"); ax.legend(fontsize=7)

    fig.tight_layout()
    out = config.FIG_DIR / "00b_estimators.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"figure: {out}")
    print("ALL OK" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
