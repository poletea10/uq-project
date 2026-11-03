"""Split figures (phase1_protocol.md §5, after step 3): ten observed-cut tasks, drawn to check by eye that the tasks
are heteroscedastic where they are noisy and that the split does what it should.

TASKS
    Ten observed-cut tasks with exactly 2 features and the stage-1 row count (N = 1,024), from a fixed figure seed
    disjoint from every other seed. A task is kept only if both features have at least 10 unique values, as
    TabICLv2 does for its prior plots (Figure F.1 caption); otherwise categorical columns stack points on top of
    each other. There is NO selection on noise: about 60% of tasks are exactly deterministic (no private root
    reaches y, t_A = 0 everywhere), and the figures show that mix; each title says which kind the task is. The full
    pipeline runs (pool, g, filter, split, duplicates). For the figure only, every pool row gets R_fig = 128
    replicates, so the aleatoric spread can be shown everywhere, not just at queries.

FIXED CONTEXT FRACTION (figures only)
    Stage 1 draws the context fraction uniformly in [0.3, 0.9]; for readability the figures fix it at 80%
    (n_ctx = 819 of N = 1,024), so the m = 205 queries do not crowd the panels. Everything else, including the
    tail size T and the hole size k, is drawn by the stage-1 samplers as in training.

ONE FIGURE PER TASK, FOUR PANELS, SAME AXES (x1, x2: the task's two standardized features)
    1 Target            all P pool rows, coloured by g(y^(0)) (sequential colormap, colourbar)
    2 Aleatoric spread  all P pool rows, coloured by t_A from the 128 replicates, min-max scaled per task to [0, 1];
                        a deterministic task has t_A = 0 everywhere and shows a flat panel saying so
    3 Context           context rows (blue, drawn on top) and every other pool row (grey, behind): the discarded
                        tail / hole rows and all query rows (i.i.d., hole, extrapolation)
    4 Queries           red circles i.i.d., red squares hole, red triangles extrapolation, purple circles duplicates
    Panels 3 and 4 both show the two carved regions: the hole as a solid circle centred on the hole's centre row
    with radius = the distance to its farthest row (Euclidean in standardized X), and the tail boundary as a dashed
    line (perpendicular to the random direction u at the smallest tail score; the tail is the side away from the
    pool). With d = 2 both are exact, not projections. Axes do not have equal scales, so the circle can look like
    an ellipse; it is the true circle in data space.

WHAT TO LOOK FOR (protocol §5)
    - Panel 2 is not uniform on noisy tasks (heteroscedastic noise), and does not track the split.
    - Panel 3: no blue point inside the circle or beyond the dashed line (the context never sees the regions); the
      grey points (all P - n_ctx non-context rows) fill those regions and are scattered among the blue
      elsewhere (the i.i.d. queries).
    - Panel 4: squares inside the circle, triangles beyond the dashed line, i.i.d. circles spread over the rest,
      purple duplicates sitting where context points are.

OUTPUT
    outputs/audits/split_figures/task{01..10}.png (group disk), and one summary line per task on stdout.

    sbatch experiment1_uq_heads/scripts/03_plot_split_tasks.sh
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from iltm2.prior._cut_graph_scm import CutGraphConfig, CutGraphSCM
from iltm2.prior._dataset import STAGES, STREAM_REPLICATES, STREAM_SPLIT, _round, sample_task_shape, task_rng
from iltm2.prior._split import DUP, EXT, HOLE, IID, make_split
from iltm2.train._targets import t_aleatoric

OUT = Path(__file__).resolve().parents[1] / "outputs" / "audits" / "split_figures"
FIG_SEED = 8_888            # disjoint from the training (0), test-gate (1, 2, 3) and validation-bank seeds
N_TASKS = 10
R_FIG = 128
MIN_UNIQUE = 10
CTX_FRACTION = 0.8          # figures only (stage 1 draws it in [0.3, 0.9])
N_STAGE1 = STAGES[1]["max_seq_len"]   # 1,024
CFG = CutGraphConfig()


def eligible_tasks():
    """Yield (task_idx, task) for observed-cut tasks with exactly 2 features of >= 10 unique values each."""
    idx = 0
    while True:
        with task_rng(FIG_SEED, 0, idx):
            shape = sample_task_shape(1, CFG, n_ctx=_round(CTX_FRACTION * N_STAGE1), d=2)
            task, _ = CutGraphSCM(CFG).sample(shape)
        if task.X.shape[1] == 2 and all(len(torch.unique(task.X[:, j])) >= MIN_UNIQUE for j in range(2)):
            yield idx, task
        idx += 1


def draw_regions(ax, X, sp):
    """The hole (solid circle around its centre, radius = distance to its farthest row) and the tail boundary
    (dashed line perpendicular to u at the smallest tail score), drawn without changing the axis limits."""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    if sp.hole_centre >= 0:
        c = X[sp.hole_centre]
        r = float((X[sp.hole_rows] - c).norm(dim=1).max())
        ax.add_patch(Circle(c.numpy(), r, fill=False, lw=1.8, ec="black", zorder=6))
    if len(sp.tail_rows) > 0:
        u = sp.u.numpy()
        t = np.linspace(-1e3, 1e3, 2)
        line = sp.tail_threshold * u[None, :] + t[:, None] * np.array([-u[1], u[0]])[None, :]
        ax.plot(line[:, 0], line[:, 1], ls="--", lw=1.8, c="black", zorder=6)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


REGION_HANDLES = [Line2D([], [], c="black", lw=1.8, label="hole"),
                  Line2D([], [], c="black", ls="--", lw=1.8, label="tail boundary")]


def draw_context_panel(ax, X, sp):
    """Panel 3: context rows (blue, on top) and every other pool row (grey, behind): the discarded tail / hole rows
    and all the query rows (before the duplicate overwrite), with the two regions."""
    x = X.numpy()
    not_ctx = torch.ones(len(x), dtype=torch.bool)
    not_ctx[sp.ctx_rows] = False
    ax.scatter(*x[not_ctx.numpy()].T, s=9, c="darkgrey", zorder=1)
    ax.scatter(*x[sp.ctx_rows.numpy()].T, s=9, c="tab:blue", zorder=2)
    draw_regions(ax, X, sp)
    n_not = int(not_ctx.sum())
    handles = [Line2D([], [], ls="", marker="o", ms=5, c="tab:blue", label=f"context ({len(sp.ctx_rows)})"),
               Line2D([], [], ls="", marker="o", ms=5, c="darkgrey",
                      label=f"not context ({n_not}: {len(sp.discarded)} discarded, {n_not - len(sp.discarded)} "
                            f"queries)")]
    ax.legend(handles=handles + REGION_HANDLES, fontsize=8, loc="best", framealpha=0.85)


def draw_query_panel(ax, X, sp):
    """Panel 4: queries by type, with the two regions."""
    x, qr, qt = X.numpy(), sp.query_rows.numpy(), sp.query_type.numpy()
    styles = [(IID, "o", "tab:red", "i.i.d."), (HOLE, "s", "tab:red", "hole"),
              (EXT, "^", "tab:red", "extrapolation"), (DUP, "o", "tab:purple", "duplicate")]
    handles = []
    for code, marker, color, name in styles:
        sel = qt == code
        ax.scatter(*x[qr[sel]].T, s=24, c=color, marker=marker, edgecolors="none", zorder=3)
        handles.append(Line2D([], [], ls="", marker=marker, ms=6, c=color, label=f"{name} ({int(sel.sum())})"))
    draw_regions(ax, X, sp)
    ax.legend(handles=handles + REGION_HANDLES, fontsize=8, loc="best", framealpha=0.85)


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    n_noisy = 0
    for j, (idx, task) in enumerate(eligible_tasks(), start=1):
        shape = task.shape
        with task_rng(FIG_SEED, 0, idx, STREAM_SPLIT):
            sp = make_split(task.X, shape, CFG.dup_fraction)
        with task_rng(FIG_SEED, 0, idx, STREAM_REPLICATES):
            Y = task.replicates(torch.arange(shape.P), R_FIG)       # (P, 128), g-transformed
        tA = t_aleatoric(Y.double()).numpy()
        noisy = task.target.n_private_roots_used > 0
        n_noisy += noisy
        X, x = task.X, task.X.numpy()

        fig, axes = plt.subplots(1, 4, figsize=(25, 6), sharex=True, sharey=True)
        kind = ("noisy: a private root reaches y" if noisy
                else "deterministic: no private root reaches y, t_A = 0 everywhere")
        fig.suptitle(f"task {j:02d} (task_idx {idx}) - {kind}\nN={shape.N}, n_ctx={shape.n_ctx}, "
                     f"m_ext={shape.m_ext}, m_hole={shape.m_hole}, m_iid={shape.m_iid}, "
                     f"duplicates={int((sp.query_type == DUP).sum())}, tail T={shape.T}, hole k={shape.k}, "
                     f"P={shape.P}", fontsize=11)

        sc = axes[0].scatter(*x.T, s=7, c=task.y0.numpy(), cmap="viridis")
        fig.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.02)
        axes[0].set_title("1  target g(y^(0))")

        span = tA.max() - tA.min()
        if span > 0:
            sc = axes[1].scatter(*x.T, s=7, c=(tA - tA.min()) / span, cmap="magma", vmin=0, vmax=1)
            fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.02)
            axes[1].set_title(f"2  aleatoric spread t_A (R={R_FIG}), min-max scaled\n"
                              f"t_A in [{tA.min():.3g}, {tA.max():.3g}], share of rows with t_A > 0: "
                              f"{np.mean(tA > 0):.0%}", fontsize=10)
        else:
            axes[1].scatter(*x.T, s=7, c="lightgrey")
            axes[1].set_title(f"2  aleatoric spread t_A (R={R_FIG})")
            axes[1].text(0.5, 0.5, f"t_A = {tA.max():.3g} everywhere\n(deterministic task)",
                         transform=axes[1].transAxes, ha="center", va="center", fontsize=12,
                         bbox=dict(facecolor="white", alpha=0.85))

        draw_context_panel(axes[2], X, sp)
        axes[2].set_title("3  context")
        draw_query_panel(axes[3], X, sp)
        axes[3].set_title("4  queries")
        for ax in axes:
            ax.set_xlabel("x1")
        axes[0].set_ylabel("x2")
        fig.tight_layout()
        path = OUT / f"task{j:02d}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"task {j:02d} (task_idx {idx}): {'noisy' if noisy else 'deterministic'}, t_A max {tA.max():.3g}, "
              f"rows with t_A > 0 {np.mean(tA > 0):.0%}, T={shape.T}, k={shape.k}; {path}", flush=True)
        if j == N_TASKS:
            break
    print(f"\n{N_TASKS} figures in {OUT}: {n_noisy} noisy, {N_TASKS - n_noisy} deterministic", flush=True)


if __name__ == "__main__":
    main()
