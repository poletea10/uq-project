"""TabICLv2 (Qu et al., 2026): total only (phase0_protocol.md §5.4).

q_m is the CDF piecewise-linear through the K = 999 raw quantiles at levels alpha_k = k/(K+1) (the levels
TabICLv2 is trained on; averaged over the ensemble by TabICLRegressor.predict), 0 below the first and 1
above the last. e_total is its exact ∫F(1-F). A_hat and E_hat are None; the evaluator skips them.
Runs with TabICLRegressor's default preprocessing and ensemble.
"""
import time

import numpy as np

from adapters.base import UQOutput
from forecasts import QuantileForecast


def quantile_levels(K):
    return np.linspace(0.0, 1.0, K + 2)[1:-1]


def to_output(qs, fit_seconds=np.nan, extras=None):
    """Conversion only: raw quantiles (q, K) -> UQOutput (total only)."""
    qs = np.asarray(qs, float)
    fc = QuantileForecast(qs, quantile_levels(qs.shape[1]))
    return UQOutput(None, None, fc.entropy(), fc, fit_seconds, dict(extras or {}), n_forward=1)


def run(task, n, rng, proto, mcfg):
    import torch
    from tabicl import TabICLRegressor

    x, y = task.context(n)
    t0 = time.perf_counter()
    reg = TabICLRegressor(model_path=mcfg["checkpoint"], allow_auto_download=False,
                          device="cuda" if torch.cuda.is_available() else "cpu",
                          random_state=int(rng.integers(2**31 - 1)))
    reg.fit(x.reshape(-1, 1), y)
    qs = np.asarray(reg.predict(task.x_qry.reshape(-1, 1), output_type="raw_quantiles"), float)
    fit_seconds = time.perf_counter() - t0
    n_unsorted = int(np.sum(np.any(np.diff(qs, axis=1) < 0, axis=1)))
    return to_output(qs, fit_seconds, {"quantiles": qs.astype(np.float32), "n_unsorted_rows": n_unsorted})
