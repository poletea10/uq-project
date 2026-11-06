"""Prior-level gate of step 2: are observed-cut tasks like unmodified graph_scm tasks? (proposal §4.3)

WHY
    The cut changes the prior the frozen backbone was pretrained on. X keeps graph_scm's mechanism but not exactly
    its distribution (no y node in the X-graph, a different acceptance event, ensure_iid), and y comes from a new
    target graph, without latent confounding or anti-causal features. If the cut made tasks much easier or much
    harder, or moved them out of the backbone's domain, the heads would be trained on a regime the backbone's
    forecast q_theta was never shaped for. This gate compares the two priors on matched tasks before any training.
    If it fails, tune the TARGET GRAPH's private-root noise (--private-root-scale, --private-root-dims), never the
    filters (the filters define the backbone's prior), rerun, and write the final config and the tolerance into the
    Phase 1 docs. The defaults are the calibrated prior (design Training §1, "The observed-cut prior").

WHAT
    Per stage (default 1 and 2), --n-tasks pairs of tasks with matched (N, n_ctx, d), drawn from TabICLv2's stage
    samplers:
      cut       `CutGraphSCM` with query_mix (1, 0, 0): a plain i.i.d. split, no holes or extrapolation, for this
                test only (so P = N);
      graph_scm `GraphSCM` with the pretraining config (`ensure_iid=False`, as v2 was trained) and the same filter
                loop as tabicl's `GraphPrior.generate_dataset`.
    Both get the same random context/query split. On each task:
      (a) frozen TabICLv2 on the queries: one raw training-mode forward (the forward route of design Architecture
          §1; no TabICLRegressor ensemble). Scored by its mean CRPS (iltm2's exact CRPS) and by the R^2 of its
          median quantile as a point prediction;
      (b) ExtraTrees held-out R^2 on the same split: the task's difficulty, measured by a model that owes nothing to
          either prior.
    Also logged: the filter rejection rate per stage and reason for both priors (graph_scm's NaN guard, which turns a
    NaN dataset into a constant one, is logged as "non_finite", not as "constant_columns"), the y-ancestor subgraph
    size (cut vs graph_scm), and the distribution of t_A from R replicates per query, with its zero-mass fraction
    (protocol step 2). Stages in --rejection-only-stages (default 3) only generate tasks and log rejection rates
    (60K-row forwards and ExtraTrees fits are too slow for a gate).

PASS RULE (per stage; recorded in the docs once the gate passes)
    1. Rejection rate: cut rejection rate in [5%, 30%] (protocol step 2).
    2. Difficulty profile: ExtraTrees R^2 quantiles q10, q25, q50, q75, q90 of the two priors differ by at most
       --r2-quantile-tol (0.10) at each. Quantiles compare the whole shape of the difficulty distribution (a median
       alone misses, e.g., a missing near-deterministic mass), and R^2 is bounded, so differences are readable.
    3. In-domain check: tasks are split into --n-bins (4) equal-count bins of ExtraTrees R^2, with edges from the
       pooled tasks of both priors. Per bin and prior, the median of R^2(TabICLv2) - R^2(ExtraTrees) gets a bootstrap
       95% CI. A bin passes if the two medians differ by at most --gap-tol (0.05) or the CIs overlap. This compares
       the backbone's advantage over a prior-free model on tasks of the SAME difficulty: a harder slice of the
       prior keeps the same advantage, a slice outside the backbone's domain loses it. The top bin (both models
       near R^2 = 1, saturated) is reported but not gated; bins with fewer than --min-bin (10) tasks of either prior
       are reported as "too few" and not gated; if no gated bin remains, the check fails (not assessable).
    Reported, not gated: log10-CRPS quantiles, KS D and KS / Mann-Whitney p-values (CRPS and R^2; with hundreds of
    tasks p-values flag even negligible shifts), the t_A zero-mass fraction, y-ancestor sizes, and (cut only) the
    R^2 ceiling 1 - E[Var(y|x)]/Var(y) from the replicates, with the gaps ceiling - R^2 of ExtraTrees and TabICLv2:
    a large gap means cut tasks are hard because y is a complex function of x, a low ceiling that y is mostly
    noise. graph_scm has no replicates, so it has no ceiling.

    sbatch experiment1_uq_heads/scripts/00f_difficulty_match.sh [--stages 1 2] [--n-tasks 256] [--private-root-scale 0.3]
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy.stats import ks_2samp, mannwhitneyu
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score

from tabicl._model.tabicl import TabICL
from tabicl.prior._dataset import Prior, should_filter
from tabicl.prior._graph_scm import GraphSCM
from tabicl.prior.graph_lib._base import Context, DatasetProperties
from tabicl.prior.graph_lib._dataset import check_x_y_ancestors_overlap
from tabicl.prior.graph_lib._graph import RandomDAG
from tabicl.prior.graph_lib._properties import sample_categorical_sizes

from iltm2.model._entropy import crps
from iltm2.prior._cut_graph_scm import CutGraphConfig, CutGraphSCM, NonFiniteReplicate, v2_prior_config
from iltm2.prior._dataset import STREAM_REPLICATES, STREAM_SPLIT, STREAM_TASK, sample_task_shape, task_rng
from iltm2.prior._filter import RejectionLog, ancestors
from iltm2.prior._graph_scm_ref import PRETRAIN, graph_scm_task
from iltm2.train._targets import t_aleatoric

OUT = Path(__file__).resolve().parents[1] / "outputs" / "audits" / "difficulty_match"
DEFAULT_CKPT = "/private/groups/ioannidislab/ajaras/tabicl-dev/tabicl-regressor-v2-20260212.ckpt"
# ------------------------------------------------------------------------------------------- unmodified graph_scm
# `graph_scm_task` (tabicl's generate_dataset loop with rejections counted) lives in iltm2.prior._graph_scm_ref,
# shared with scripts/01b_check_val_bank.py.


def graph_scm_y_ancestor_size(d):
    """Size of y's ancestor subgraph in one graph_scm draw: the graph part of tabicl's `RandomDataset.sample`,
    copied (node count, DAG, feature placement per group, graph check), without evaluating any function."""
    ctx = Context(config=PRETRAIN)
    cfg, sampler = ctx.config, ctx.sampler
    props = DatasetProperties(n_train=1, n_test=0,
                              cat_sizes={"x": sample_categorical_sizes(d, ctx, max_cat_size=200), "y": [0]})
    while True:
        n_nodes = sampler.randint("n_nodes", cfg.min_n_nodes, cfg.max_n_nodes + 1, use_log=True)
        graph = RandomDAG(ctx).sample(n_nodes)
        specs = [dict() for _ in range(n_nodes)]
        for group in list(set(s.group for s in props.feature_specs.values())):   # as tabicl (PYTHONHASHSEED=0)
            fs = {k: v for k, v in props.feature_specs.items() if v.group == group}
            if cfg.subsample_feature_nodes:
                n_fn = sampler.randint("n_feature_nodes", 1, n_nodes + 1)
                nodes = np.random.permutation(n_nodes)[:n_fn]
            else:
                nodes = np.arange(n_nodes)
            idx = np.random.choice(nodes, replace=True, size=len(fs))
            for i, (name, spec) in enumerate(fs.items()):
                specs[idx[i]][name] = spec
        if check_x_y_ancestors_overlap(graph, specs):
            break
    y_node = next(i for i, s in enumerate(specs) if "y_0" in s)
    return len(ancestors(graph, y_node))


# ------------------------------------------------------------------------------------------- one task pair (CPU)


def et_r2(X, y, n_ctx):
    et = ExtraTreesRegressor(n_estimators=100, n_jobs=1, random_state=0).fit(X[:n_ctx].numpy(), y[:n_ctx].numpy())
    return float(r2_score(y[n_ctx:].numpy(), et.predict(X[n_ctx:].numpy())))


def make_pair(seed, stage, i, cfg, R, rejection_only, max_redraws=10):
    """Generate one matched pair (cut, graph_scm) and everything that does not need the GPU.

    A non-finite replicate rejects the cut task entirely (design §1.4): it is revoked in the log (counted as
    "non_finite_replicate") and a new cut task of the same shape is drawn, on a separate deterministic RNG stream.
    """
    torch.set_num_threads(1)
    cut_log, gs_log = RejectionLog(), RejectionLog()
    with task_rng(seed, stage + (100 if rejection_only else 0), i):
        shape = sample_task_shape(stage, cfg)
        task, _ = CutGraphSCM(cfg).sample(shape, cut_log)
        Xg, yg = graph_scm_task(shape, gs_log)
        gs_anc = graph_scm_y_ancestor_size(shape.d)
    if not rejection_only:
        with task_rng(seed, stage, i, STREAM_SPLIT):
            perm = torch.randperm(shape.N)                  # same split for both priors: context first
        for k in range(max_redraws + 1):
            try:
                with task_rng(seed, stage, i, STREAM_REPLICATES + 100 * k):
                    Y = task.replicates(perm[shape.n_ctx:], R).double()   # (m, R) g-transformed replicates
                break
            except NonFiniteReplicate:
                cut_log.revoke("non_finite_replicate", noisy=task.target.n_private_roots_used > 0)
                if k == max_redraws:
                    raise
                with task_rng(seed, stage, i, STREAM_TASK + 100 * (k + 1)):
                    task, _ = CutGraphSCM(cfg).sample(shape, cut_log)
    out = dict(stage=stage, task=i, N=shape.N, n_ctx=shape.n_ctx, d=shape.d, cut_log=cut_log, gs_log=gs_log,
               cut_y_ancestors=task.diagnostics["y_ancestor_size"], gs_y_ancestors=gs_anc,
               **{f"cut_{k}": v for k, v in task.diagnostics.items() if k != "y_ancestor_size"})
    if rejection_only:
        return out
    Xc, yc = task.X[perm], task.y0[perm]
    Xg, yg = Xg[perm], yg[perm]
    tA = t_aleatoric(Y)
    # R^2 ceiling (cut only): the best R^2 any predictor of y from x can reach on this task's queries,
    # 1 - E[Var(y | x)] / Var(y), with Var(y | x) estimated per query from its R replicates (unbiased) and Var(y)
    # from the queries' y^(0). ExtraTrees R^2 far below the ceiling means y is a hard function of x (complexity);
    # a low ceiling means y is mostly noise.
    noise_var = float(Y.var(dim=1, unbiased=True).mean())
    total_var = float(yc[shape.n_ctx:].double().var(unbiased=True))
    ceiling = 1.0 - noise_var / total_var if total_var > 0 else float("nan")
    out.update(cut_X=Xc, cut_y=yc, gs_X=Xg, gs_y=yg, cut_r2=et_r2(Xc, yc, shape.n_ctx),
               gs_r2=et_r2(Xg, yg, shape.n_ctx), tA=tA.numpy(), cut_ceiling=ceiling,
               cut_det_tA_nonzero=bool(task.target.n_private_roots_used == 0 and bool((tA > 0).any())))
    return out


# ------------------------------------------------------------------------------------------- frozen TabICLv2 (GPU)


def load_backbone(path, device):
    """TabICLv2 exactly as the sklearn wrapper builds it (`TabICL(**ckpt["config"])`), in training mode: the raw
    forward of pretraining (`train/_run.py: run_micro_batch`). Dropout is 0, so training mode is deterministic.
    Step 5's `_backbone.py` replaces this; its unit test pins both to tabicl's own forward."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = TabICL(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).train()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def backbone_scores(model, X, y, n_ctx, device):
    """(mean CRPS, R^2 of the median quantile) of TabICLv2's 999-quantile forecast over the query rows (n_ctx:).

    The median quantile is index K // 2 = 499, at level 500/1000 = 0.5 (levels k/(K+1), k = 1..K).
    """
    q = model(X[None].to(device), y[None, :n_ctx].to(device))[0].double()   # (m, 999); d=None as in v2 training
    yq = y[n_ctx:].to(device).double()
    score = float(crps(q, yq[:, None]).mean())
    r2 = float(r2_score(yq.cpu().numpy(), q[:, q.shape[1] // 2].cpu().numpy()))
    return score, r2


# ------------------------------------------------------------------------------------------- report

QS = (0.1, 0.25, 0.5, 0.75, 0.9)


def quantiles(a):
    a = np.asarray(a, float)
    return " ".join(f"q{int(p * 100)}={np.quantile(a, p):.4g}" for p in QS)


def summarize_logs(rows, key):
    log = RejectionLog()
    for r in rows:
        log += r[key]
    return log


def bootstrap_median_ci(a, seed, n_boot=2000):
    """(median, 2.5% and 97.5% quantiles of the bootstrap distribution of the median)."""
    a = np.asarray(a, float)
    rng = np.random.default_rng(seed)
    meds = np.median(a[rng.integers(0, len(a), size=(n_boot, len(a)))], axis=1)
    return float(np.median(a)), float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))


def in_domain_bins(et_cut, gap_cut, et_gs, gap_gs, n_bins, min_bin, tol):
    """Pass-rule item 3. Equal-count bins of ExtraTrees R^2 (edges from both priors pooled); per bin and prior the
    median gap R^2(TabICLv2) - R^2(ET) with a bootstrap 95% CI. Returns (rows, passed)."""
    edges = np.quantile(np.concatenate([et_cut, et_gs]), np.linspace(0.0, 1.0, n_bins + 1))
    rows = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        last = b == n_bins - 1

        def inside(e):
            return (e >= lo) & ((e <= hi) if last else (e < hi))

        mc, mg = inside(et_cut), inside(et_gs)
        row = dict(bin=b, lo=float(lo), hi=float(hi), n_cut=int(mc.sum()), n_gs=int(mg.sum()), gated=not last)
        if row["n_cut"] >= min_bin and row["n_gs"] >= min_bin:
            c, g = bootstrap_median_ci(gap_cut[mc], 2 * b), bootstrap_median_ci(gap_gs[mg], 2 * b + 1)
            row.update(cut_med=c[0], cut_lo=c[1], cut_hi=c[2], gs_med=g[0], gs_lo=g[1], gs_hi=g[2],
                       ok=bool(abs(c[0] - g[0]) <= tol or (c[1] <= g[2] and g[1] <= c[2])))
        else:
            row["ok"] = None                                 # too few tasks of one prior: not assessable
        rows.append(row)
    gated = [r for r in rows if r["gated"] and r["ok"] is not None]
    return rows, bool(gated) and all(r["ok"] for r in gated)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", type=int, nargs="*", default=[1, 2])
    ap.add_argument("--rejection-only-stages", type=int, nargs="*", default=[3])
    ap.add_argument("--n-tasks", type=int, default=256)
    ap.add_argument("--n-tasks-rejection", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    default = CutGraphConfig()                              # the prior's calibrated defaults
    ap.add_argument("--private-root-scale", type=float, default=default.private_root_scale,
                    help="private roots' output scale (graph_scm: 1.0)")
    ap.add_argument("--private-root-dims", type=int, nargs=2, default=list(default.private_root_dims_range),
                    help="private roots' number of dims, randint(lo, hi) on a log scale (graph_scm: 1 32)")
    ap.add_argument("--rejection-range", type=float, nargs=2, default=[0.05, 0.30])
    ap.add_argument("--r2-quantile-tol", type=float, default=0.10)
    ap.add_argument("--gap-tol", type=float, default=0.05)
    ap.add_argument("--n-bins", type=int, default=4)
    ap.add_argument("--min-bin", type=int, default=10)
    args = ap.parse_args()

    cfg = CutGraphConfig(query_mix=(1.0, 0.0, 0.0), private_root_scale=args.private_root_scale,
                         private_root_dims_range=tuple(args.private_root_dims))
    tag = f"prs{args.private_root_scale:g}_prd{args.private_root_dims[0]}-{args.private_root_dims[1]}"
    out_dir = OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sha = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    print(f"target-graph config: {tag}; checkpoint {args.checkpoint} (sha256 {sha[:16]}...); device {device}; "
          f"n_jobs {args.n_jobs}", flush=True)
    model = load_backbone(args.checkpoint, device) if args.stages else None

    def verdict(ok, text):
        print(f"  {'OK  ' if ok else 'FAIL'} {text}", flush=True)
        return ok

    fails, summary = [], dict(config=tag, checkpoint_sha256=sha, args=vars(args), stages={})
    for stage in sorted(set(args.stages) | set(args.rejection_only_stages)):
        rej_only = stage not in args.stages
        n = args.n_tasks_rejection if rej_only else args.n_tasks
        t0 = time.time()
        rows = Parallel(n_jobs=args.n_jobs)(delayed(make_pair)(args.seed, stage, i, cfg, args.R, rej_only)
                                            for i in range(n))
        cut_log, gs_log = summarize_logs(rows, "cut_log"), summarize_logs(rows, "gs_log")
        print(f"\n# stage {stage} ({n} pairs, {'rejection rates only' if rej_only else 'full'}; "
              f"generation {time.time() - t0:.0f}s)", flush=True)
        print(f"  cut        {cut_log.summary()}")
        det, noisy = cut_log.split("deterministic"), cut_log.split("noisy")
        acc_det = det["accepted"] / max(1, det["accepted"] + noisy["accepted"])
        print(f"             deterministic target graphs (no private-root ancestor of y): "
              f"{cut_log.deterministic_fraction:.1%} of attempts, {acc_det:.1%} of accepted tasks; rejection rate "
              f"noisy {noisy['rate']:.1%} ({noisy['attempts']} attempts), deterministic {det['rate']:.1%} "
              f"({det['attempts']} attempts)")
        print(f"  graph_scm  {gs_log.summary()}")
        cut_anc, gs_anc = [r["cut_y_ancestors"] for r in rows], [r["gs_y_ancestors"] for r in rows]
        print(f"  y-ancestor subgraph size  cut: {quantiles(cut_anc)} | graph_scm: {quantiles(gs_anc)}")
        st = dict(n=n, cut_rejection=cut_log.rate, gs_rejection=gs_log.rate,
                  cut_rejected=dict(cut_log.rejected), gs_rejected=dict(gs_log.rejected),
                  cut_graph_resamples=cut_log.graph_resamples, cut_deterministic_fraction=cut_log.deterministic_fraction,
                  cut_noisy_rejection=noisy["rate"], cut_deterministic_rejection=det["rate"],
                  cut_accepted_deterministic_share=acc_det,
                  cut_y_ancestors_median=float(np.median(cut_anc)), gs_y_ancestors_median=float(np.median(gs_anc)))

        # 1. rejection rate
        lo, hi = args.rejection_range
        if not verdict(lo <= cut_log.rate <= hi,
                       f"[1] cut rejection rate {cut_log.rate:.1%} in [{lo:.0%}, {hi:.0%}] (graph_scm {gs_log.rate:.1%})"):
            fails.append(f"stage {stage} rejection rate")

        if not rej_only:
            for r in rows:
                r["cut_crps"], r["cut_tab_r2"] = backbone_scores(model, r["cut_X"], r["cut_y"], r["n_ctx"], device)
                r["gs_crps"], r["gs_tab_r2"] = backbone_scores(model, r["gs_X"], r["gs_y"], r["n_ctx"], device)
            col = lambda k: np.array([r[k] for r in rows], float)   # noqa: E731
            cc, gc, cr, gr = col("cut_crps"), col("gs_crps"), col("cut_r2"), col("gs_r2")
            ct, gt = col("cut_tab_r2"), col("gs_tab_r2")
            tA = np.concatenate([r["tA"] for r in rows])

            # 2. difficulty profile
            dq = {p: float(np.quantile(cr, p) - np.quantile(gr, p)) for p in QS}
            print(f"  ExtraTrees R^2   cut: {quantiles(cr)}\n                   graph_scm: {quantiles(gr)}")
            if not verdict(all(abs(v) <= args.r2_quantile_tol for v in dq.values()),
                           "[2] ExtraTrees R^2 quantiles, |cut - graph_scm| <= "
                           f"{args.r2_quantile_tol}: " + " ".join(f"q{int(p * 100)} {v:+.3f}" for p, v in dq.items())):
                fails.append(f"stage {stage} R^2 profile")

            # 3. in-domain check
            bins, ok_bins = in_domain_bins(cr, ct - cr, gr, gt - gr, args.n_bins, args.min_bin, args.gap_tol)
            print("  in-domain: median R^2(TabICLv2) - R^2(ET) [95% CI] per ExtraTrees-R^2 bin")
            for b in bins:
                head = f"    bin {b['bin']} [{b['lo']:.3f}, {b['hi']:.3f}] n={b['n_cut']}/{b['n_gs']}"
                if b["ok"] is None:
                    print(f"{head}: too few tasks, not gated")
                    continue
                tagb = ("OK" if b["ok"] else "FAIL") if b["gated"] else "saturated, not gated"
                print(f"{head}: cut {b['cut_med']:+.3f} [{b['cut_lo']:+.3f}, {b['cut_hi']:+.3f}]  "
                      f"graph_scm {b['gs_med']:+.3f} [{b['gs_lo']:+.3f}, {b['gs_hi']:+.3f}]  {tagb}")
            if not verdict(ok_bins, f"[3] in-domain: every gated bin has |delta median| <= {args.gap_tol} or "
                                    "overlapping CIs (top bin excluded; at least one gated bin assessable)"):
                fails.append(f"stage {stage} in-domain")

            # reported, not gated
            print(f"  reported: log10 CRPS  cut: {quantiles(np.log10(np.maximum(cc, 1e-12)))}\n"
                  f"                        graph_scm: {quantiles(np.log10(np.maximum(gc, 1e-12)))}")
            ks_c, ks_r = ks_2samp(cc, gc), ks_2samp(cr, gr)
            print(f"            CRPS KS D={ks_c.statistic:.3f} p={ks_c.pvalue:.2g}, "
                  f"Mann-Whitney p={mannwhitneyu(cc, gc).pvalue:.2g}; "
                  f"ET R^2 KS D={ks_r.statistic:.3f} p={ks_r.pvalue:.2g}, Mann-Whitney p={mannwhitneyu(cr, gr).pvalue:.2g}")
            print(f"            TabICLv2 R^2  cut: {quantiles(ct)} | graph_scm: {quantiles(gt)}")
            ceil = col("cut_ceiling")
            print(f"            cut R^2 ceiling 1 - E[Var(y|x)]/Var(y) (from replicates): {quantiles(ceil)}\n"
                  f"            cut ceiling - ExtraTrees R^2 (complexity gap): {quantiles(ceil - cr)}\n"
                  f"            cut ceiling - TabICLv2 R^2: {quantiles(ceil - ct)}; "
                  f"tasks with ceiling >= 0.9: {np.mean(ceil >= 0.9):.1%}, with ET R^2 >= 0.9: {np.mean(cr >= 0.9):.1%}")
            n_det = sum(r["cut_n_private_roots_used"] == 0 for r in rows)
            n_det_bad = sum(r["cut_det_tA_nonzero"] for r in rows)
            print(f"            deterministic cut tasks with any t_A > 0 (must be 0: replicates differ only through U2): "
                  f"{n_det_bad}/{n_det}")
            print(f"            t_A (R={args.R}, {len(tA)} queries): zero-mass fraction {np.mean(tA == 0):.2%}; "
                  f"{quantiles(tA)}; all-zero tasks {np.mean([np.all(r['tA'] == 0) for r in rows]):.1%}")
            st.update(r2_quantile_diff={str(p): v for p, v in dq.items()}, in_domain_bins=bins,
                      crps_ks_d=float(ks_c.statistic), r2_ks_d=float(ks_r.statistic),
                      tA_zero_mass=float(np.mean(tA == 0)), cut_ceiling_median=float(np.median(ceil)),
                      cut_complexity_gap_median=float(np.median(ceil - cr)), cut_crps_median=float(np.median(cc)),
                      gs_crps_median=float(np.median(gc)), cut_r2_median=float(np.median(cr)),
                      gs_r2_median=float(np.median(gr)))
        summary["stages"][stage] = st
        keep = [k for k in rows[0] if not k.endswith(("_X", "_y", "_log")) and k != "tA"]
        pd.DataFrame([{k: r[k] for k in keep} for r in rows]).to_parquet(out_dir / f"stage{stage}_tasks.parquet")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {out_dir}\n{'ALL OK' if not fails else f'{len(fails)} FAILED: ' + '; '.join(fails)}", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
