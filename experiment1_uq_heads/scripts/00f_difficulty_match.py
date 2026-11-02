"""Prior-level gate of step 2: are observed-cut tasks as hard as unmodified graph_scm tasks? (proposal §4.3)

WHY
    The cut changes the prior the frozen backbone was pretrained on. X keeps graph_scm's mechanism but not exactly
    its distribution (no y node in the X-graph, a different acceptance event, ensure_iid), and y comes from a new
    target graph, without latent confounding or anti-causal features. If the cut made tasks much easier or much
    harder, the heads would be trained on a regime the backbone's forecast q_theta was never shaped for. This gate
    compares the two priors on matched tasks before any training. If it fails, tune the TARGET GRAPH
    (--hidden-dims, --hidden-scale, --private-root-scale), never the filters (the filters define the backbone's
    prior), rerun, and write the final config and the tolerance into the Phase 1 docs.

WHAT
    Per stage (default 1 and 2), --n-tasks pairs of tasks with matched (N, n_ctx, d), drawn from TabICLv2's stage
    samplers:
      cut       `CutGraphSCM` with query_mix (1, 0, 0): a plain i.i.d. split, no holes or extrapolation, for this
                test only (so P = N);
      graph_scm `GraphSCM` with the pretraining config (`ensure_iid=False`, as v2 was trained) and the same filter
                loop as tabicl's `GraphPrior.generate_dataset`.
    Both get the same random context/query split. On each task:
      (a) frozen TabICLv2 CRPS on the queries: one raw training-mode forward (the forward route of design
          Architecture §1; no TabICLRegressor ensemble), scored with iltm2's exact CRPS;
      (b) ExtraTrees held-out R^2 on the same split.
    Also logged: the filter rejection rate per stage and reason for both priors (protocol step-2 gate: 5%-30% for
    the cut), the y-ancestor subgraph size (cut vs graph_scm), and the distribution of t_A from R replicates per
    query, with its zero-mass fraction (protocol step 2). Stages in --rejection-only-stages (default 3) only
    generate tasks and log rejection rates (60K-row CRPS and ExtraTrees fits are too slow for a gate).

PASS RULE (proposed default; recorded in the docs once fixed)
    Per stage: median CRPS ratio cut / graph_scm in [0.8, 1.25], |median R^2 cut - median R^2 graph_scm| <= 0.05,
    and cut rejection rate in [5%, 30%]. KS and Mann-Whitney p-values are reported, not gated: with hundreds of
    tasks they flag even negligible shifts.

    sbatch experiment1_uq_heads/scripts/00f_difficulty_match.sh [--stages 1 2] [--n-tasks 256] [--hidden-scale 0.5]
"""
import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
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
from iltm2.prior._cut_graph_scm import CutGraphConfig, CutGraphSCM, v2_prior_config
from iltm2.prior._dataset import STREAM_REPLICATES, STREAM_SPLIT, sample_task_shape, task_rng
from iltm2.prior._filter import RejectionLog, ancestors
from iltm2.train._targets import t_aleatoric

OUT = Path(__file__).resolve().parents[1] / "outputs" / "audits" / "difficulty_match"
DEFAULT_CKPT = "/private/groups/ioannidislab/ajaras/tabicl-dev/tabicl-regressor-v2-20260212.ckpt"
PRETRAIN = v2_prior_config(ensure_iid=False)   # unmodified graph_scm, exactly as TabICLv2 was trained


# ------------------------------------------------------------------------------------------- unmodified graph_scm


def graph_scm_task(shape, log):
    """tabicl's `GraphPrior.generate_dataset` loop for one regression dataset, with rejections counted.

    Same steps: GraphSCM -> delete_unique_features -> should_filter, retry on failure. tabicl pads X to
    max_features; here max_features = d (no padding), as for the cut tasks, so both see the same columns.
    GraphSCM's NaN guard turns a NaN dataset into a constant one, which the constant-column check rejects.
    """
    params = dict(regression=True, seq_len=shape.N, train_size=shape.n_ctx, num_features=shape.d,
                  max_features=shape.d, config=PRETRAIN, device="cpu")
    while True:
        X, y = GraphSCM(**params)()
        X, d = Prior.delete_unique_features(X[None], torch.tensor([shape.d]))
        if not (d > 0).all():
            log.reject("constant_columns")
            continue
        if should_filter(X[0], y, PRETRAIN, is_classif=False):
            log.reject("unpredictable")
            continue
        log.accepted += 1
        return X[0, :, :int(d[0])], y


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


def make_pair(seed, stage, i, cfg, R, rejection_only):
    """Generate one matched pair (cut, graph_scm) and everything that does not need the GPU."""
    torch.set_num_threads(1)
    cut_log, gs_log = RejectionLog(), RejectionLog()
    with task_rng(seed, stage + (100 if rejection_only else 0), i):
        shape = sample_task_shape(stage, cfg)
        task, _ = CutGraphSCM(cfg).sample(shape, cut_log)
        Xg, yg = graph_scm_task(shape, gs_log)
        gs_anc = graph_scm_y_ancestor_size(shape.d)
    out = dict(stage=stage, task=i, N=shape.N, n_ctx=shape.n_ctx, d=shape.d, cut_log=cut_log, gs_log=gs_log,
               cut_y_ancestors=task.diagnostics["y_ancestor_size"], gs_y_ancestors=gs_anc,
               **{f"cut_{k}": v for k, v in task.diagnostics.items() if k != "y_ancestor_size"})
    if rejection_only:
        return out
    with task_rng(seed, stage, i, STREAM_SPLIT):
        perm = torch.randperm(shape.N)                      # same split for both priors: context first
    Xc, yc = task.X[perm], task.y0[perm]
    Xg, yg = Xg[perm], yg[perm]
    with task_rng(seed, stage, i, STREAM_REPLICATES):
        tA = t_aleatoric(task.replicates(perm[shape.n_ctx:], R).double())
    out.update(cut_X=Xc, cut_y=yc, gs_X=Xg, gs_y=yg, cut_r2=et_r2(Xc, yc, shape.n_ctx),
               gs_r2=et_r2(Xg, yg, shape.n_ctx), tA=tA.numpy())
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
def backbone_crps(model, X, y, n_ctx, device):
    """Mean CRPS of TabICLv2's 999-quantile forecast over the query rows (rows n_ctx: of X)."""
    q = model(X[None].to(device), y[None, :n_ctx].to(device))        # (1, m, 999); d=None as in v2 training
    return float(crps(q[0].double(), y[n_ctx:, None].to(device).double()).mean())


# ------------------------------------------------------------------------------------------- report


def quantiles(a):
    a = np.asarray(a, float)
    return " ".join(f"q{int(p * 100)}={np.quantile(a, p):.4g}" for p in (0.1, 0.25, 0.5, 0.75, 0.9))


def summarize_logs(rows, key):
    log = RejectionLog()
    for r in rows:
        log += r[key]
    return log


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
    ap.add_argument("--hidden-dims", type=int, nargs=2, default=[1, 32])
    ap.add_argument("--hidden-scale", type=float, default=1.0)
    ap.add_argument("--private-root-scale", type=float, default=1.0)
    ap.add_argument("--crps-ratio", type=float, nargs=2, default=[0.8, 1.25])
    ap.add_argument("--r2-tol", type=float, default=0.05)
    ap.add_argument("--rejection-range", type=float, nargs=2, default=[0.05, 0.30])
    args = ap.parse_args()

    cfg = CutGraphConfig(query_mix=(1.0, 0.0, 0.0), hidden_dims_range=tuple(args.hidden_dims),
                         hidden_scale=args.hidden_scale, private_root_scale=args.private_root_scale)
    tag = f"h{args.hidden_dims[0]}-{args.hidden_dims[1]}_hs{args.hidden_scale:g}_prs{args.private_root_scale:g}"
    out_dir = OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sha = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    print(f"target-graph config: {tag}; checkpoint {args.checkpoint} (sha256 {sha[:16]}...); device {device}; "
          f"n_jobs {args.n_jobs}", flush=True)
    model = load_backbone(args.checkpoint, device) if args.stages else None

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
        print(f"  graph_scm  {gs_log.summary()}")
        cut_anc, gs_anc = [r["cut_y_ancestors"] for r in rows], [r["gs_y_ancestors"] for r in rows]
        print(f"  y-ancestor subgraph size  cut: {quantiles(cut_anc)} | graph_scm: {quantiles(gs_anc)}")
        st = dict(n=n, cut_rejection=cut_log.rate, gs_rejection=gs_log.rate,
                  cut_rejected=dict(cut_log.rejected), gs_rejected=dict(gs_log.rejected),
                  cut_graph_resamples=cut_log.graph_resamples,
                  cut_y_ancestors_median=float(np.median(cut_anc)), gs_y_ancestors_median=float(np.median(gs_anc)))
        lo, hi = args.rejection_range
        ok_rej = lo <= cut_log.rate <= hi
        print(f"  {'OK  ' if ok_rej else 'FAIL'} cut rejection rate {cut_log.rate:.1%} in [{lo:.0%}, {hi:.0%}] "
              f"(graph_scm {gs_log.rate:.1%})")
        if not ok_rej:
            fails.append(f"stage {stage} rejection rate")

        if not rej_only:
            for r in rows:
                r["cut_crps"] = backbone_crps(model, r["cut_X"], r["cut_y"], r["n_ctx"], device)
                r["gs_crps"] = backbone_crps(model, r["gs_X"], r["gs_y"], r["n_ctx"], device)
            cc, gc = np.array([r["cut_crps"] for r in rows]), np.array([r["gs_crps"] for r in rows])
            cr, gr = np.array([r["cut_r2"] for r in rows]), np.array([r["gs_r2"] for r in rows])
            tA = np.concatenate([r["tA"] for r in rows])
            ratio, dr2 = np.median(cc) / np.median(gc), np.median(cr) - np.median(gr)
            print(f"  TabICLv2 CRPS  cut: {quantiles(cc)}\n                 graph_scm: {quantiles(gc)}\n"
                  f"                 KS p={ks_2samp(cc, gc).pvalue:.2g}, Mann-Whitney p={mannwhitneyu(cc, gc).pvalue:.2g}")
            print(f"  ExtraTrees R^2 cut: {quantiles(cr)}\n                 graph_scm: {quantiles(gr)}\n"
                  f"                 KS p={ks_2samp(cr, gr).pvalue:.2g}, Mann-Whitney p={mannwhitneyu(cr, gr).pvalue:.2g}")
            print(f"  t_A (R={args.R}, {len(tA)} queries): zero-mass fraction {np.mean(tA == 0):.2%}; "
                  f"{quantiles(tA)}; per-task zero-mass tasks {np.mean([np.all(r['tA'] == 0) for r in rows]):.1%}")
            ok_c = args.crps_ratio[0] <= ratio <= args.crps_ratio[1]
            ok_r = abs(dr2) <= args.r2_tol
            print(f"  {'OK  ' if ok_c else 'FAIL'} median CRPS ratio cut/graph_scm {ratio:.3f} in "
                  f"[{args.crps_ratio[0]}, {args.crps_ratio[1]}]")
            print(f"  {'OK  ' if ok_r else 'FAIL'} |median R^2 difference| {abs(dr2):.3f} <= {args.r2_tol}")
            fails += [f"stage {stage} CRPS ratio"] * (not ok_c) + [f"stage {stage} R^2"] * (not ok_r)
            st.update(crps_ratio=float(ratio), r2_diff=float(dr2), tA_zero_mass=float(np.mean(tA == 0)),
                      cut_crps_median=float(np.median(cc)), gs_crps_median=float(np.median(gc)),
                      cut_r2_median=float(np.median(cr)), gs_r2_median=float(np.median(gr)))
        summary["stages"][stage] = st
        keep = [k for k in rows[0] if not k.endswith(("_X", "_y", "_log")) and k != "tA"]
        pd.DataFrame([{k: r[k] for k in keep} for r in rows]).to_parquet(out_dir / f"stage{stage}_tasks.parquet")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {out_dir}\n{'ALL OK' if not fails else f'{len(fails)} FAILED: ' + '; '.join(fails)}", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
