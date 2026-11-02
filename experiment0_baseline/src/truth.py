"""Realized truth (phase0_protocol.md §4.1-4.2).

    A = e(p_tau(.|x*)),    E(q_m) = E_{Y~p_tau}[S(q_m, Y)] - A = d_S(q_m, p_tau).

T1 (closed form): p_tau = N(f(x*), sigma(x*)^2), so A = sigma / sqrt(pi) and E(q_m) is the Cramér distance
between the model's forecast and that Gaussian (a step at f on toy rung 0), via the forecast's own closed
form or the shared grid.

T2, T3 (and the estimator audit of 00b): unbiased estimators from R exact replicates y^(1..R),
    A_hat = 1/(R(R-1)) Σ_{r<s} |y_r - y_s|,   E_hat = (1/R) Σ_r CRPS(q_m, y_r) - A_hat.
"""
import numpy as np

import entropy as en


def aleatoric(sigma_q):
    return np.asarray(sigma_q, float) / en.SQRT_PI


def epistemic(forecast, f_q, sigma_q, proto):
    return forecast.excess_vs_normal(f_q, sigma_q, proto)


# ---------------------------------------------------------------- replicate estimators


def replicate_aleatoric(Y):
    """Unbiased CRPS-entropy estimate per row of Y (q, R), R >= 2.

    Σ_{r<s} |y_r - y_s| = Σ_k (2k - R + 1) y_(k) over the sorted row (k = 0..R-1).
    """
    Y = np.sort(np.asarray(Y, float), axis=1)
    R = Y.shape[1]
    assert R >= 2
    coef = 2.0 * np.arange(R) - R + 1.0
    return (Y @ coef) / (R * (R - 1))


def replicate_epistemic(forecast, Y, proto, A_hat=None):
    """Unbiased excess-risk estimate per row: mean CRPS of q_m over the replicates minus A_hat."""
    if A_hat is None:
        A_hat = replicate_aleatoric(Y)
    return forecast.crps(Y, proto).mean(axis=1) - A_hat
