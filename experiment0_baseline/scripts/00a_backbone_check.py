"""Backbone check (phase0_protocol.md §2.7): TabICLv2 on toy rungs 0-2 at every n, and the pass criteria.

    python 00a_backbone_check.py [--overwrite]

1. Fits TabICLv2 on every (toy rung, n): parquet per (rung, n) in outputs/raw/toy/tabiclv2/, reused on
   re-runs unless --overwrite.
2. Draws one 3x4 figure per toy rung to outputs/figures/toy/backbone_rung{k}_tabiclv2.png: row 1 as in
   §2.6; row 2 realized A, realized expected score A + E(q) and TabICLv2's total e(q); row 3 realized E(q).
   This doubles as the motivating figure.
3. Evaluates the pre-registered criteria (protocol.yaml `backbone_check`) at every n in n_eval, with
   "outside" = queries outside both strips:
   1. Fit: on every toy rung, mean CRPS of TabICLv2 outside <= fit_ratio_max x that of the heteroscedastic
      GP. CRPS here is the exact expected score A + E(q_m) under the known truth (no sampled y). Needs the
      GP's toy outputs (01_toy.sh --method hgp); without them criterion 1 is PENDING.
   2. Locality: toy rung 0, mean e(q) on the sparse strip >= locality_ratio_min x mean e(q) outside.
   3. Noise: toy rung 2, n >= noise_n_min, mean e(q) on the noise strip >= noise_ratio_min x mean e(q) outside.
   The check fails if criterion 1 fails at any n, or if criterion 2 or 3 fails at more than max_failing_n n.
   Writes outputs/tables/00a_backbone_check.csv. Exit 0 on PASS, 1 on FAIL or PENDING.
"""
import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

import config
import evaluate
import figures
import toy

METHOD, REFERENCE = "tabiclv2", "hgp"


def outside(d):
    return ~d.in_noise_strip & ~d.in_sparse_strip


def criteria(tab, ref, bc):
    """One row per (criterion, rung, n): value, threshold, ok (None if the reference is missing)."""
    rows = []
    for n in bc["n_eval"]:
        for k in config.TOY_RUNGS:
            d = tab[(tab.rung == k) & (tab.n == n)]
            r = ref[(ref.rung == k) & (ref.n == n)] if not ref.empty else ref
            if r.empty:
                rows.append({"criterion": 1, "rung": k, "n": n, "value": np.nan,
                             "threshold": bc["fit_ratio_max"], "ok": None})
                continue
            crps_tab = (d.A_true + d.E_true)[outside(d)].mean()
            crps_ref = (r.A_true + r.E_true)[outside(r)].mean()
            ratio = crps_tab / crps_ref
            rows.append({"criterion": 1, "rung": k, "n": n, "value": ratio, "threshold": bc["fit_ratio_max"],
                         "ok": bool(ratio <= bc["fit_ratio_max"]),
                         "detail": f"CRPS outside: TabICLv2 {crps_tab:.4g}, GP {crps_ref:.4g}"})
        d = tab[(tab.rung == 0) & (tab.n == n)]
        ratio = d.e_total[d.in_sparse_strip].mean() / d.e_total[outside(d)].mean()
        rows.append({"criterion": 2, "rung": 0, "n": n, "value": ratio, "threshold": bc["locality_ratio_min"],
                     "ok": bool(ratio >= bc["locality_ratio_min"])})
        if n >= bc["noise_n_min"]:
            d = tab[(tab.rung == 2) & (tab.n == n)]
            ratio = d.e_total[d.in_noise_strip].mean() / d.e_total[outside(d)].mean()
            rows.append({"criterion": 3, "rung": 2, "n": n, "value": ratio, "threshold": bc["noise_ratio_min"],
                         "ok": bool(ratio >= bc["noise_ratio_min"])})
    return pd.DataFrame(rows)


def verdict(table, bc):
    c1 = table[table.criterion == 1]
    if c1.ok.isna().any():
        return "PENDING", "criterion 1 needs the heteroscedastic GP's toy outputs: run 01_toy.sh --method hgp"
    failing = table.ok.map(lambda v: not pd.isna(v) and not v)
    fail_n = {c: sorted(int(n) for n in set(table[(table.criterion == c) & failing].n)) for c in (1, 2, 3)}
    reasons = []
    if fail_n[1]:
        reasons.append(f"criterion 1 fails at n={fail_n[1]}")
    for c in (2, 3):
        if len(fail_n[c]) > bc["max_failing_n"]:
            reasons.append(f"criterion {c} fails at n={fail_n[c]} (> {bc['max_failing_n']})")
    return ("FAIL" if reasons else "PASS"), "; ".join(reasons) or "all criteria met"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--overwrite", action="store_true", help="refit even if outputs exist")
    args = p.parse_args()
    proto = config.protocol()
    bc = proto["backbone_check"]

    failed = []
    for k in config.TOY_RUNGS:
        for n in proto["n_grid"]:
            try:
                evaluate.run_toy(METHOD, k, n, overwrite=args.overwrite)
            except Exception:
                print(f"FAIL {METHOD} toy rung{k} n={n}\n{traceback.format_exc()}", flush=True)
                failed.append((k, n))
    if failed:
        print(f"{len(failed)} FAILED fits (rung, n): {failed}")
        sys.exit(1)

    tab = evaluate.load_toy(METHOD)
    for k in config.TOY_RUNGS:
        d = tab[tab.rung == k]
        task = toy.load_task(k, proto)
        lims = figures.row_limits(task, [d], "backbone", proto)
        out = config.FIG_DIR / "toy" / f"backbone_rung{k}_{METHOD}.png"
        figures.context(task, d, f"toy rung {k}: {METHOD} (backbone check, total only)", out, proto, lims,
                        mode="backbone")
        print(f"figure: {out}")

    table = criteria(tab, evaluate.load_toy(REFERENCE), bc)
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out = config.TABLE_DIR / "00a_backbone_check.csv"
    table.to_csv(out, index=False)
    for _, r in table.iterrows():
        status = "PENDING" if pd.isna(r.ok) else ("OK  " if r.ok else "FAIL")
        op = "<=" if r.criterion == 1 else ">="
        detail = r.get("detail")
        print(f"{status} criterion {r.criterion} toy rung {r.rung} n={r.n}: ratio {r.value:.3g} {op} "
              f"{r.threshold}  {detail if isinstance(detail, str) else ''}")
    print(f"table: {out}")
    v, why = verdict(table, bc)
    print(f"BACKBONE CHECK: {v} ({why})")
    sys.exit(0 if v == "PASS" else 1)


if __name__ == "__main__":
    main()
