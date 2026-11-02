"""T1 toy set (phase0_protocol.md §2): fit a model on toy rungs 0-2 at every n, then draw the figures.

    python 01_toy.py --method hgp [--rungs 0 1 2] [--n ...] [--overwrite] [--no-figures]
    python 01_toy.py --figures-only
    python 01_toy.py --paper-figure [--paper-rung 2] [--paper-n 1024] [--allow-missing] [--figures-only]

1. Fits every selected (toy rung, n): one parquet per (method, rung, n) in outputs/raw/toy/{method}/, reused
   on re-runs unless --overwrite. In a SLURM array job with no --rungs, the array index is the toy rung.
2. Writes outputs/tables/toy_{method}_diagnostics.csv (fit time, sigma clip rate, and the method's
   convergence diagnostics where it has them) and prints it.
3. Draws one 3x4 context figure per (T1 model, toy rung) to outputs/figures/toy/rung{k}_{method}.png, for
   every T1 model that has outputs, so that the row limits are shared across models within a rung.
   Array jobs skip this step; run a final --figures-only once every rung is done.
4. --paper-figure: the 3x3 paper figure (§2.6) at (--paper-rung, --paper-n), from saved outputs only (never
   fits), to outputs/figures/toy/paper_rung{k}_n{n}.png and .pdf. Needs outputs of hgp, decicl and iltmv2 at
   that (rung, n); exits non-zero listing the missing ones, unless --allow-missing, which draws each missing
   column empty and titled "<name> (pending)". Alone (no --method, no --figures-only) it draws only this
   figure; with --figures-only it also redraws the per-(model, rung) figures.
Non-zero exit on any failed fit.
"""
import argparse
import os
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

DIAG_KEYS = ("rhat_max", "divergences", "rhat_flag", "m_g_mean")


def fit_all(method, rungs, ns, overwrite):
    failed = []
    for k in rungs:
        for n in ns:
            try:
                evaluate.run_toy(method, k, n, overwrite=overwrite)
            except Exception:
                print(f"FAIL {method} toy rung{k} n={n}\n{traceback.format_exc()}", flush=True)
                failed.append((k, n))
    return failed


def diagnostics(method):
    raw = evaluate.load_toy(method)
    if raw.empty:
        return
    rows = []
    for (k, n), d in raw.groupby(["rung", "n"]):
        row = {"rung": int(k), "n": int(n), "fit_seconds": d.fit_seconds.iloc[0],
               "sigma_clip_rate": d.sigma_clip_rate.iloc[0]}
        with np.load(config.toy_extras_path(method, int(k), int(n))) as z:
            row.update({key: z[key].item() for key in DIAG_KEYS if key in z.files})
        rows.append(row)
    table = pd.DataFrame(rows)
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out = config.TABLE_DIR / f"toy_{method}_diagnostics.csv"
    table.to_csv(out, index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"table: {out}")


def draw(proto, rungs):
    for k in rungs:
        frames = {m: evaluate.load_toy(m, [k]) for m in evaluate.TOY_METHODS}
        frames = {m: d for m, d in frames.items() if not d.empty}
        if not frames:
            continue
        task = toy.load_task(k, proto)
        lims = figures.row_limits(task, list(frames.values()), "t1", proto)
        for m, d in frames.items():
            out = config.FIG_DIR / "toy" / f"rung{k}_{m}.png"
            figures.context(task, d, f"toy rung {k}: {m}", out, proto, lims, mode="t1")
            print(f"figure: {out}")


def draw_paper(proto, k, n, allow_missing):
    """Returns the list of missing models (empty when the figure was drawn complete)."""
    frames = {m: evaluate.load_toy(m, [k]) for m, _ in figures.PAPER_COLUMNS}
    missing = [m for m, d in frames.items() if d.empty or d[d["n"] == n].empty]
    if missing and not allow_missing:
        print(f"paper figure: no saved outputs at toy rung {k}, n={n} for {missing} "
              f"(pass --allow-missing to draw them as pending)")
        return missing
    out = config.FIG_DIR / "toy" / f"paper_rung{k}_n{n}.png"
    figures.paper(toy.load_task(k, proto), frames, n, out, proto)
    print(f"figure: {out} (+ .pdf){'; pending: ' + str(missing) if missing else ''}")
    return []


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=evaluate.TOY_METHODS)
    p.add_argument("--rungs", type=int, nargs="+", default=None)
    p.add_argument("--n", type=int, nargs="+", default=None, help="context sizes (default: protocol n_grid)")
    p.add_argument("--overwrite", action="store_true", help="refit even if outputs exist")
    p.add_argument("--figures-only", action="store_true")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--paper-figure", action="store_true", help="draw the 3x3 paper figure from saved outputs")
    p.add_argument("--paper-rung", type=int, default=2)
    p.add_argument("--paper-n", type=int, default=1024)
    p.add_argument("--allow-missing", action="store_true", help="paper figure: draw missing models as pending")
    args = p.parse_args()
    proto = config.protocol()
    is_array = "SLURM_ARRAY_TASK_ID" in os.environ
    if args.rungs is not None:
        rungs = args.rungs
    elif is_array and not args.figures_only:
        rungs = [int(os.environ["SLURM_ARRAY_TASK_ID"])]
    else:
        rungs = list(config.TOY_RUNGS)
    ns = args.n or proto["n_grid"]

    failed, missing = [], []
    paper_only = args.paper_figure and not args.method and not args.figures_only
    if not (args.figures_only or paper_only):
        assert args.method, "--method is required unless --figures-only or --paper-figure"
        failed = fit_all(args.method, rungs, ns, args.overwrite)
        diagnostics(args.method)
    if args.figures_only or not (paper_only or args.no_figures or is_array):
        draw(proto, rungs)
    if args.paper_figure:
        missing = draw_paper(proto, args.paper_rung, args.paper_n, args.allow_missing)
    if failed:
        print(f"{len(failed)} FAILED fits (rung, n): {failed}")
    if failed or missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
