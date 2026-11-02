"""T1 figures (phase0_protocol.md §2.6, §2.7).

Context figure: 3 rows x one column per n, one figure per (model, toy rung).
  row 1: context points; true mean f with true 90 % band f ± 1.645 sigma; forecast median and 90 % band
         (5 %-95 % quantiles of q_m).
  row 2 (mode "t1"):       realized A (blue), A_hat (red) and total e(q_m) (black).             linear
        (mode "backbone"): realized A, realized expected score A + E(q), total e(q).             linear
  row 3 (mode "t1"):       realized E(q_m) (blue) and E_hat (red).          log, floor sigma_floor/sqrt(pi)
        (mode "backbone"): realized E(q).
Noise strip (orange) and sparse strip (purple) shaded in every panel; rug of the context x on rows 2-3;
sigma clip rate in a corner of row 2. Row 1 uses the fixed fig_y_lim of toy_function.yaml (wide forecast
bands are clipped); rows 2-3 limits come from row_limits() so that, within a toy rung, the
figures of all T1 models share them.

Paper figure (§2.6): 3 rows x 3 columns (heteroscedastic GP, Dec-ICL, iLTMv2) at one (toy rung, n), with
the "t1" rows and row limits shared across the columns.

Toy-function figure (00_curate_toy): f with f_base and f_gap, and each rung's n_max context.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import entropy as en

Z90 = 1.6448536269514722
TRUTH_C, EST_C, TOTAL_C, SCORE_C = "tab:blue", "tab:red", "k", "tab:green"
NOISE_C, SPARSE_C = "tab:orange", "tab:purple"
PAPER_COLUMNS = (("hgp", "Heteroscedastic GP"), ("decicl", "Dec-ICL"), ("iltmv2", "iLTMv2"))


def _save(fig, path, extra_suffixes=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    for suffix in extra_suffixes:
        fig.savefig(path.with_suffix(suffix))
    plt.close(fig)


def _shade_strips(ax, tcfg):
    ax.axvspan(*tcfg["noise_strip"], color=NOISE_C, alpha=0.08, lw=0)
    ax.axvspan(*tcfg["sparse_strip"], color=SPARSE_C, alpha=0.08, lw=0)


def _rug(ax, x):
    ax.plot(x, np.zeros_like(x), "|", color="k", ms=6, alpha=0.5, transform=ax.get_xaxis_transform())


def _floor(proto):
    return proto["sigma_floor"] / en.SQRT_PI


def _row_columns(mode):
    """Columns drawn in rows 2 and 3."""
    if mode == "t1":
        return ["A_true", "A_hat", "e_total"], ["E_true", "E_hat"]
    return ["A_true", "score_true", "e_total"], ["E_true"]


def _with_score(d):
    return d.assign(score_true=d.A_true + d.E_true)


def row_limits(task, frames, mode, proto):
    """Shared y-limits {'y', 'A', 'E'} over the given frames (one per model) of one toy rung."""
    floor = _floor(proto)
    frames = [_with_score(d) for d in frames]
    a_cols, e_cols = _row_columns(mode)
    a_all = np.concatenate([d[a_cols].to_numpy().ravel() for d in frames])
    e_all = np.maximum(np.concatenate([d[e_cols].to_numpy().ravel() for d in frames]), floor)
    a_max = np.nanmax(a_all)
    return {"y": tuple(task.tcfg["fig_y_lim"]),
            "A": (0.0, 1.05 * a_max if a_max > 0 else 1.0),
            "E": (np.nanmin(e_all) / 1.5, np.nanmax(e_all) * 1.5)}


def _column(axes_col, task, dn, n, proto, lims, mode):
    """Draw one column (rows 1-3) for the raw rows dn of one (model, toy rung, n); dn None draws it empty."""
    floor = _floor(proto)
    top, mid, bot = axes_col
    if dn is not None:
        dn = _with_score(dn).sort_values("x")
        xs = np.linspace(*task.tcfg["x_range"], 400)
        f_xs, s_xs = task.f(xs), task.sigma(xs)
        x, y = task.context(n)
        top.fill_between(xs, f_xs - Z90 * s_xs, f_xs + Z90 * s_xs, color="0.85", lw=0, label="truth 90%")
        top.plot(xs, f_xs, color="0.4", lw=1.0, label="truth mean")
        top.fill_between(dn.x, dn.q05, dn.q95, color=EST_C, alpha=0.18, lw=0, label="forecast 90%")
        top.plot(dn.x, dn.q50, color=EST_C, lw=1.0, label="forecast median")
        top.scatter(x, y, s=4, c="k", alpha=0.5, lw=0, zorder=3, label="context")

        if mode == "t1":
            mid.plot(dn.x, dn.A_true, color=TRUTH_C, lw=1.2, label="realized A")
            mid.plot(dn.x, dn.A_hat, color=EST_C, lw=1.2, label="Â")
            mid.plot(dn.x, dn.e_total, color=TOTAL_C, lw=1.4, ls="--", label="total e(q)")
            bot.plot(dn.x, np.maximum(dn.E_true, floor), color=TRUTH_C, lw=1.2, label="realized E(q_m)")
            bot.plot(dn.x, np.maximum(dn.E_hat, floor), color=EST_C, lw=1.2, label="Ê")
        else:
            mid.plot(dn.x, dn.A_true, color=TRUTH_C, lw=1.2, label="realized A")
            mid.plot(dn.x, dn.score_true, color=SCORE_C, lw=1.2, label="realized A + E(q)")
            mid.plot(dn.x, dn.e_total, color=TOTAL_C, lw=1.4, ls="--", label="total e(q)")
            bot.plot(dn.x, np.maximum(dn.E_true, floor), color=TRUTH_C, lw=1.2, label="realized E(q)")
        clip = float(dn.sigma_clip_rate.iloc[0])
        if np.isfinite(clip):
            mid.text(0.98, 0.97, f"σ clip {clip:.1%}", transform=mid.transAxes, ha="right", va="top", fontsize=7)
        for ax in (mid, bot):
            _rug(ax, x)
    top.set_ylim(lims["y"])
    mid.set_ylim(lims["A"])
    bot.set_yscale("log")
    bot.set_ylim(lims["E"])
    for ax in (top, mid, bot):
        _shade_strips(ax, task.tcfg)
    bot.set_xlabel("x")


def _row_labels_and_legends(axes, mode):
    axes[0, 0].set_ylabel("y")
    axes[1, 0].set_ylabel("A" if mode == "t1" else "CRPS entropy / score")
    axes[2, 0].set_ylabel("E")
    for r, loc in enumerate(("best", "upper left", "lower left")):
        for ax in axes[r]:
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=6, loc=loc)
                break


def context(task, d, title, path, proto, lims, mode="t1"):
    """One context figure. d: raw rows of one (method, toy rung), all n."""
    ns = sorted(int(n) for n in d.n.unique())
    fig, axes = plt.subplots(3, len(ns), figsize=(3.4 * len(ns), 8.0), sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": [1.3, 1, 1]})
    for c, n in enumerate(ns):
        _column(axes[:, c], task, d[d.n == n], n, proto, lims, mode)
        axes[0, c].set_title(f"n={n}", fontsize=9)
    _row_labels_and_legends(axes, mode)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    _save(fig, path)


def paper(task, frames, n, path, proto):
    """3x3 paper figure at one (toy rung, n). frames: {method: raw rows of that method and toy rung, or
    None/empty if pending}; columns follow PAPER_COLUMNS, and a pending column is drawn empty and titled
    "<name> (pending)". Row limits are shared across the columns that have outputs. Saved as path (PNG)
    and path with a .pdf suffix."""
    at_n = {m: d[d.n == n] for m, d in frames.items()
            if d is not None and not d.empty and not d[d.n == n].empty}
    if at_n:
        lims = row_limits(task, list(at_n.values()), "t1", proto)
    else:
        floor = _floor(proto)
        lims = {"y": tuple(task.tcfg["fig_y_lim"]), "A": (0.0, 1.0), "E": (floor, 1.0)}
    fig, axes = plt.subplots(3, len(PAPER_COLUMNS), figsize=(3.8 * len(PAPER_COLUMNS), 8.0), sharex=True,
                             squeeze=False, gridspec_kw={"height_ratios": [1.3, 1, 1]})
    for c, (m, name) in enumerate(PAPER_COLUMNS):
        _column(axes[:, c], task, at_n.get(m), n, proto, lims, "t1")
        axes[0, c].set_title(name if m in at_n else f"{name} (pending)", fontsize=10)
    _row_labels_and_legends(axes, "t1")
    fig.suptitle(f"toy rung {task.rung}, n={n}", fontsize=10)
    fig.tight_layout()
    _save(fig, path, extra_suffixes=(".pdf",))


def toy_function(tasks, stats, n_grid, path):
    """tasks: {rung: ToyTask} (same f, same context x); stats: curation statistics of f."""
    t0 = next(iter(tasks.values()))
    tcfg = t0.tcfg
    a, b = tcfg["sparse_strip"]
    in_gap = (t0.x_ctx >= a) & (t0.x_ctx <= b)
    gap_counts = ", ".join(f"{int(in_gap[:n].sum())} at n={n}" for n in n_grid)
    xs = np.linspace(*tcfg["x_range"], 2001)
    fig, axes = plt.subplots(1 + len(tasks), 1, figsize=(9, 2.6 * (1 + len(tasks))), sharex=True, sharey=True)
    ax = axes[0]
    ax.plot(xs, t0.fn(xs), "k", lw=1.4, label="f")
    ax.plot(xs, t0.fn.base(xs), color="0.5", lw=1.0, ls="--", label="f_base")
    ax.plot(xs, t0.fn.gap(xs), color=SPARSE_C, lw=1.0, label="f_gap")
    ax.set_title(f"toy_seed {t0.fn.seed}: max|f| {stats['max_abs_f']:.2f}, sd(f_gap) on sparse strip "
                 f"{stats['gap_sd']:.2f}, range of f on noise strip {stats['noise_strip_range']:.2f}", fontsize=9)
    ax.legend(fontsize=7)
    for ax, (k, task) in zip(axes[1:], sorted(tasks.items())):
        x, y = task.context(len(task.x_ctx))
        s = task.sigma(xs)
        ax.fill_between(xs, task.fn(xs) - Z90 * s, task.fn(xs) + Z90 * s, color="0.85", lw=0)
        ax.plot(xs, task.fn(xs), color="0.4", lw=1.0)
        ax.scatter(x, y, s=3, c="k", alpha=0.5, lw=0)
        ax.set_title(f"toy rung {k}: n={len(x)} context; context points in the sparse strip: {gap_counts}",
                     fontsize=9)
    for ax in axes:
        _shade_strips(ax, tcfg)
    axes[-1].set_xlabel("x")
    fig.tight_layout()
    _save(fig, path)
