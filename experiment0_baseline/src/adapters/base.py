"""UQOutput and the member-wise projections shared by the adapters (phase0_protocol.md §4.3, §5).

Every adapter returns quantities in y units, plus the forecast q_m itself (UQOutput.cdf), which the
truth E(q_m) needs. A_hat is always the mean member entropy, never a conversion of a method's
pre-aggregated "data uncertainty" scalar.
"""
from dataclasses import dataclass, field

import numpy as np

import entropy as en
from forecasts import GaussianMixtureForecast


@dataclass
class UQOutput:
    A_hat: np.ndarray | None          # (n_query,), y units; None for total-only methods
    E_hat: np.ndarray | None
    e_total: np.ndarray
    cdf: object                       # forecasts.Forecast: q_m, needed for the truth E(q_m)
    fit_seconds: float
    extras: dict = field(default_factory=dict)   # native outputs, unprojected, for audit
    sigma_clip_rate: float = np.nan              # fraction of predicted sigmas raised to sigma_floor
    n_forward: float = np.nan                    # model forward passes (in-context methods)


def clip_sigma(sigma, floor):
    sigma = np.asarray(sigma, float)
    return np.maximum(sigma, floor), float(np.mean(sigma < floor))


def gaussian_members(mu, sigma, w, sigma_floor):
    """Project a Gaussian-member family {N(mu_m, sigma_m^2), w_m} member-wise, in closed form.

    mu, sigma: (q, M) in y units; w: (M,) or (q, M). Returns (A_hat, e_total, E_hat, forecast, clip_rate).
    """
    sigma, clip_rate = clip_sigma(sigma, sigma_floor)
    w = np.broadcast_to(np.asarray(w, float), np.shape(mu))
    assert np.allclose(w.sum(1), 1.0)
    fc = GaussianMixtureForecast(mu, sigma, w)
    A = np.sum(w * en.gm_member_entropy(sigma), axis=1)
    e = fc.entropy()
    E = e - A
    assert np.all(E >= -1e-9 * np.maximum(e, 1e-12)), "negative epistemic: projection applied wrongly"
    return A, e, np.maximum(E, 0.0), fc, clip_rate
