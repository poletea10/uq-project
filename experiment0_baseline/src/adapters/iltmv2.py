"""iLTMv2: the Phase 1 heads on the frozen TabICLv2 regressor (phase0_protocol.md §5.5; experiment1 Phase 1).

A_hat is the aleatoric head's Â and E_hat the epistemic head's Ê_dir, used directly (proposal §4.4): the heads
predict them, so there is no member family to project. Both come out of `iltm2.inference.ILTMv2Regressor` already in
y units (the heads live in TabICLv2's z-normalized units; CRPS entropy and divergence are 1-homogeneous, so they are
multiplied by the context's y scale there). q_m is TabICLv2's own forecast on the raw training-mode route (design
Architecture §1, the route the heads were trained on), as a CDF piecewise-linear through its 999 raw quantiles at
levels k/1000, 0 below the first and 1 above the last: the same convention as the TabICLv2 adapter (§5.4).
e_total is that CDF's exact entropy. extras: the quantiles, Â, Ê_dir, iltm2's own e(q_theta) and the amortization
gap e(q_theta) - Â (proposal §4.7), the y scale and the checkpoints used.

Which model (configs/methods.yaml): `arm` takes that arm's checkpoints selected on the validation bank
(`iltm2.inference.BEST`, phase1_protocol.md §9); `checkpoint_A` (+ `checkpoint_E`, `pair`) points at any other
checkpoint, of any arm (A1, A2, B).
"""
import time

import numpy as np

from adapters.base import UQOutput
from adapters.tabiclv2 import quantile_levels
from forecasts import QuantileForecast

_REGRESSORS = {}       # one model per configuration and process (the backbone takes seconds to load)


def to_output(qs, A, E, e_q=None, fit_seconds=np.nan, extras=None):
    """Conversion only: raw quantiles (q, K) and the heads' Â, Ê_dir (q,), y units -> UQOutput."""
    qs = np.asarray(qs, float)
    A, E = np.asarray(A, float), np.asarray(E, float)
    assert A.shape == E.shape == (qs.shape[0],), "one Â and one Ê_dir per query"
    fc = QuantileForecast(qs, quantile_levels(qs.shape[1]))
    ex = dict(extras or {})
    ex.update(quantiles=qs.astype(np.float32), A_head=A, E_head=E)
    if e_q is not None:
        e_q = np.asarray(e_q, float)
        ex.update(e_q=e_q, amortization_gap=e_q - A)
    return UQOutput(A, E, fc.entropy(), fc, fit_seconds, ex, n_forward=1)


def regressor(mcfg):
    import torch
    from iltm2.inference import ILTMv2Regressor
    from iltm2.model._backbone import DEFAULT_BACKBONE

    key = tuple(sorted((k, str(v)) for k, v in mcfg.items()))
    if key not in _REGRESSORS:
        kw = dict(backbone_path=mcfg.get("backbone", DEFAULT_BACKBONE),
                  device=mcfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
        if mcfg.get("checkpoint_A"):
            reg = ILTMv2Regressor(mcfg["checkpoint_A"], mcfg.get("checkpoint_E"), pair=mcfg.get("pair"), **kw)
        else:
            reg = ILTMv2Regressor.from_best(mcfg["arm"], **kw)
        print(f"iLTMv2: {reg!r}", flush=True)
        _REGRESSORS[key] = reg
    return _REGRESSORS[key]


def run(task, n, rng, proto, mcfg):
    x, y = task.context(n)
    reg = regressor(mcfg)
    t0 = time.perf_counter()
    reg.fit(x.reshape(-1, 1), y)
    u = reg.predict_uncertainty(task.x_qry.reshape(-1, 1))
    fit_seconds = time.perf_counter() - t0
    extras = {"y_scale": reg.y_std, "arm_A": reg.arm["A"], "arm_E": reg.arm["E"],
              "checkpoint_A": reg.specs["A"].path, "checkpoint_E": reg.specs["E"].path}
    return to_output(u["quantiles"], u["A"], u["E"], u["e_q"], fit_seconds, extras)
