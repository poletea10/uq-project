"""Method registry and per-(rung, n) evaluation on the T1 toy set (phase0_protocol.md §2, §9).

run_toy fits one method on one (toy rung, n), computes the realized truth A and E(q_m) of its own
forecast, and writes one parquet (one row per query) to outputs/raw/toy/{method}/, plus the adapter's
native extras next to it. Each row also carries the 5/50/95 % quantiles of q_m (by CDF inversion), so the
figures never need to rebuild the forecast. Existing outputs are reused unless overwrite.
"""
import importlib
import os

import numpy as np
import pandas as pd

import config
import toy
import truth

METHODS = ("hgp", "tabiclv2")
TOY_METHODS = ("hgp",)            # T1 rows; Dec-ICL and iLTMv2 join when their adapters exist
TOTAL_ONLY = ("tabiclv2",)        # backbone check only
QUANTILE_LEVELS = (0.05, 0.5, 0.95)


def adapter(method):
    assert method in METHODS, f"unknown method {method}; choose from {METHODS}"
    return importlib.import_module(f"adapters.{method}")


def fit_rng(proto, tcfg, rung_k, n):
    """Model-fit stream [base, toy_seed, rung, 3, n] (§9)."""
    return np.random.default_rng([proto["base_seed"], tcfg["toy_seed"], rung_k, 3, n])


def _col(v, m):
    return np.full(m, np.nan) if v is None else np.asarray(v, float)


def _atomic_write(df, extras, raw_file, extras_file):
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_x = extras_file.with_name(extras_file.stem + f".tmp{os.getpid()}.npz")
    np.savez_compressed(tmp_x, **{k: np.asarray(v) for k, v in extras.items()})
    os.replace(tmp_x, extras_file)
    tmp = raw_file.with_suffix(f".tmp{os.getpid()}")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, raw_file)


def run_toy(method, rung_k, n, overwrite=False, verbose=True):
    raw_file = config.toy_raw_path(method, rung_k, n)
    if raw_file.exists() and not overwrite:
        return pd.read_parquet(raw_file)
    proto, tcfg, mcfg = config.protocol(), config.toy_function(), config.methods()[method]
    task = toy.load_task(rung_k, proto, tcfg)
    out = adapter(method).run(task, n, fit_rng(proto, tcfg, rung_k, n), proto, mcfg)

    m = len(task.x_qry)
    A_true = truth.aleatoric(task.sigma_qry)
    E_true = truth.epistemic(out.cdf, task.f_qry, task.sigma_qry, proto)
    qs = out.cdf.quantiles(QUANTILE_LEVELS, proto)
    in_noise, in_sparse = task.query_regions()
    df = pd.DataFrame({
        "method": method, "rung": rung_k, "n": n, "toy_seed": tcfg["toy_seed"],
        "q_idx": np.arange(m), "x": task.x_qry, "f_true": task.f_qry, "sigma_true": task.sigma_qry,
        "A_true": A_true, "E_true": E_true,
        "A_hat": _col(out.A_hat, m), "E_hat": _col(out.E_hat, m), "e_total": _col(out.e_total, m),
        "q05": qs[:, 0], "q50": qs[:, 1], "q95": qs[:, 2],
        "in_noise_strip": in_noise, "in_sparse_strip": in_sparse,
        "fit_seconds": out.fit_seconds, "n_forward": out.n_forward, "sigma_clip_rate": out.sigma_clip_rate})
    if verbose:
        a = "" if out.A_hat is None else f" A_hat~{np.median(out.A_hat):.3g} (A~{np.median(A_true):.3g})"
        e = "" if out.E_hat is None else f" E_hat~{np.median(out.E_hat):.3g}"
        print(f"[{method} toy rung{rung_k} n={n}]{a}{e} e_total~{np.median(out.e_total):.3g} "
              f"E_true~{np.median(E_true):.3g} clip {out.sigma_clip_rate:.2g} fit {out.fit_seconds:.1f}s",
              flush=True)
    _atomic_write(df, out.extras, raw_file, config.toy_extras_path(method, rung_k, n))
    return df


def load_toy(method, rungs=config.TOY_RUNGS):
    """All saved toy outputs of a method (empty DataFrame if none)."""
    files = [f for k in rungs for f in sorted((config.RAW_DIR / "toy" / method).glob(f"rung{k}_n*.parquet"))]
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
