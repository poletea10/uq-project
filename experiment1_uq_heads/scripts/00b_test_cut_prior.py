"""Gate for step 2 of the order of work (phase1_protocol.md §3, unit tests of §4.2): the observed-cut prior and filter.

WHAT IS TESTED
    iltm2.prior._cut_graph_scm   the observed-cut prior: X from graph_scm's X-graph, y = Psi(X_obs, U2) from a
                                 graph_scm-style target graph, exact replicates by redrawing U2 only. Every
                                 training task, the validation bank and the T2 test bank come from it, so the
                                 targets t_A, t_E are only exact if this code is.
    iltm2.prior._cached_graph    the cached graph_lib pieces both graphs are built from (only U1 / U2 redrawn)
    iltm2.prior._postprocess     fit/apply versions of tabicl's outlier clamp and standard scaling, and g
    iltm2.prior._dataset         task RNG, task shapes (pipeline steps 1-3) and `sample_task`
    iltm2.prior._filter          tabicl's task filters (graph check on the target graph, constant columns,
                                 predictability test, non-finite values), which keep our tasks inside the
                                 frozen backbone's (filtered) prior.

HOW
    Small tasks (stage-1 shapes cut down to N = 256, d = 6) so the gate runs in minutes on CPU. The checks follow
    protocol §4.2 and add the ones the new target-graph design needs:
      1. caching: everything but the noise is cached (replay test);
      2. X is bit-identical across replicates, and one tiled pass equals R separate passes;
      3. exactness: the target graph sees only observed values;
      4. structure of the target graph;
      5. g is fitted once and applied unchanged;
      6. determinism per (seed, step, task_idx), in-process and across processes;
      7. filters and task-shape bookkeeping.
    The "replay" mechanism (`RootPointLaw` tape): with the tape in "record" mode every draw of base points at a
    root (the noise) is stored; in "replay" mode the stored draws are used instead of fresh ones. If a replayed pass
    under a different global RNG state reproduces the recorded pass bit for bit, nothing but those base points is
    redrawn between calls, which is the caching requirement of proposal §4.3.

OUTPUT
    One "OK  <check>" or "FAIL <check>" line per check. Exit code 1 if any check failed.

    sbatch experiment1_uq_heads/scripts/00b_test_cut_prior.sh
"""
import multiprocessing as mp
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from tabicl.prior._reg2cls import outlier_removing, standard_scaling
from tabicl.prior.graph_lib._base import Context
from tabicl.prior.graph_lib._converter import CategoricalConverter, NumericalConverter

from iltm2.prior._cached_graph import set_tape
from iltm2.prior._cut_graph_scm import (CutGraphConfig, CutGraphSCM, InputNodeFunction, XGraph, observed_writeback,
                                        v2_prior_config)
from iltm2.prior._dataset import STAGES, STREAM_REPLICATES, sample_task, sample_task_shape, task_rng
from iltm2.prior._postprocess import RobustClamp, YTransform
from iltm2.prior._filter import (RejectionLog, all_finite, ancestors, drop_constant_columns, graph_check,
                                 predictability_rejects)
from tabicl.prior._dataset import Prior

N, N_CTX, D = 256, 200, 6       # small stage-1-like tasks
CFG = CutGraphConfig()
FAILS = []


def check(name, ok, detail=""):
    print(f"{'OK  ' if ok else 'FAIL'} {name}  {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def guarded(fn):
    """Run one test section; an exception counts as a FAIL of that section instead of stopping the gate."""
    try:
        fn()
    except Exception:
        traceback.print_exc()
        check(f"{fn.__name__} ran without raising", False)


def small_task(task_idx, seed=0, cfg=CFG, d=D, cat_sizes=None):
    with task_rng(seed, 0, task_idx):
        shape = sample_task_shape(1, cfg, N=N, n_ctx=N_CTX, d=d)
        task, log = CutGraphSCM(cfg).sample(shape, cat_sizes=cat_sizes)
    return task, log


# ------------------------------------------------------------------------------------------- 1. caching
# Proposal §4.3 / CLAUDE.md gotcha: as shipped, every call of a graph_lib graph function redraws the root point
# law, the feature-importance weights and node_importance, so a second call is a different task. `CachedNodeFunction`
# draws them once. Both graphs need it: the X-graph (its ensure_iid fit pass and data pass must be one task) and
# the target graph (replicates must differ only through U2).


def test_caching():
    print("\n# 1. caching: only the base points at roots (the noise) are redrawn between calls")

    # X-graph. cat_modes="neighbor" excludes the softmax categorical modes, whose per-row multinomial draws are
    # legitimate U1 noise that the tape does not capture; everything else must replay exactly.
    cfg_nb = v2_prior_config(cat_modes="neighbor")
    with task_rng(1, 0, 0):
        xg = XGraph(Context(config=cfg_nb), D, cat_sizes=[0, 0, 0, 3, 5, 0])
        xg._run(300)                                         # ensure_iid fit pass
        set_tape(xg.laws(), "record")
        a = xg._run(300)
    set_tape(xg.laws(), "replay")
    torch.manual_seed(12345)
    np.random.seed(12345)
    b = xg._run(300)
    set_tape(xg.laws(), None)
    c = xg._run(300)
    check("X-graph: replaying the base points under another RNG state reproduces X bit for bit",
          torch.equal(a, b), f"({len(xg.laws())} root laws)")
    check("X-graph: fresh base points give different X", not torch.equal(a, c))

    # Target graph: its noise sources are the private roots and the hidden dims of X-input nodes (both U2).
    task, _ = small_task(0)
    tg = task.target
    inputs = tg.observed_inputs(task.x_clamped[:64], task.x_graph.col_index)
    set_tape(tg.laws(), "record")
    y1 = tg.sample(inputs, 64)
    set_tape(tg.laws(), "replay")
    torch.manual_seed(999)
    y2 = tg.sample(inputs, 64)
    set_tape(tg.laws(), None)
    y3 = tg.sample(inputs, 64)
    n_hidden = sum(isinstance(tg.nodes_[i], InputNodeFunction) for i in tg.needed_)
    check("target graph: replaying U2 under another RNG state reproduces y bit for bit", torch.equal(y1, y2),
          f"({len(tg.laws())} U2 sources: {n_hidden} hidden-dim blocks, {tg.n_private_roots_used} private roots)")
    check("target graph: fresh U2 gives different y (the replicates carry noise)", not torch.equal(y1, y3))


# ------------------------------------------------------------------------------------------- 2. replicates
# Replication = hold X, redraw U2, rerun the target graph (proposal §4.3). "Assert bit-identical X across
# replicates" is the construction's only correctness guard: `TargetGraph.sample` snapshots the tiled inputs and
# checks them after the pass, and `CutGraphTask.replicates` checks the task's X.


def test_replicates():
    print("\n# 2. replicates: X bit-identical across replicates, one tiled pass = R separate passes")
    task, _ = small_task(1)
    rows = torch.arange(task.shape.n_ctx, task.shape.N)    # any pool rows will do
    with task_rng(0, 0, 1, STREAM_REPLICATES):
        Y = task.replicates(rows, 8)
    check("replicates run with the X guard on and return (rows, R) finite values",
          Y.shape == (len(rows), 8) and all_finite(Y), f"shape {tuple(Y.shape)}")

    # The guard must fire if a node modifies X in place: make the first X-input node add 1 to its input block.
    tg = task.target
    p = next(i for i in tg.input_hosts if i in tg.needed_)
    node, orig = tg.nodes_[p], tg.nodes_[p]._transform

    def mutate(obs, n):
        obs.add_(1.0)
        return orig(obs, n)

    node._transform = mutate
    try:
        task.replicates(rows, 4)
        fired = False
    except AssertionError:
        fired = True
    finally:
        del node._transform
    check("the X guard fires when a node modifies X in place", fired)

    # One tiled pass vs R separate passes on the same U2. The tiled pass draws base points for R*n rows at once;
    # splitting each recorded draw into its R row blocks and replaying them one pass at a time must give the
    # same y. This holds only if every node function acts row-wise after fitting, which the tiling relies on.
    n, R = 40, 5
    inputs = tg.observed_inputs(task.x_clamped[:n], task.x_graph.col_index)
    set_tape(tg.laws(), "record")
    y_tiled = tg.sample(inputs, n, R)
    for law in tg.laws():
        base = law.tape[0]
        law.tape = [base[r * n:(r + 1) * n] for r in range(R)]
    set_tape(tg.laws(), "replay")
    y_sep = torch.stack([tg.sample(inputs, n, 1)[:, 0] for _ in range(R)], dim=1)
    set_tape(tg.laws(), None)
    err = float((y_tiled - y_sep).abs().max() / y_tiled.abs().max().clamp(min=1e-12))
    check("one tiled pass equals R separate passes given the same U2 (row-wise node functions)", err < 1e-5,
          f"max rel diff {err:.1e}")


# ------------------------------------------------------------------------------------------- 3. exactness
# The target graph may depend on X only through the observed columns (module docstring, "exactness rule"): an
# X-input node holds `observed_writeback(converter, x̃)`, the converter's write-back recomputed from the clamped
# observed value x̃. On rows the clamp leaves alone it must equal what the X-graph wrote back, for every converter
# mode, except the `*_id` modes, whose write-back is the latent value and is replaced by the category's center.

ALL_CAT_MODES = ["neighbor_id", "neighbor_disc", "neighbor_func", "neighbor_int",
                 "softmax_id", "softmax_disc", "softmax_int"]


def test_exactness():
    print("\n# 3. exactness: the target graph sees only observed values")
    num_stats = {"id": [0, 0], "kumar": [0, 0]}               # [columns compared, columns equal]
    for k, mode in enumerate(ALL_CAT_MODES):
        prior = v2_prior_config(cat_modes=mode)
        with task_rng(2, 0, k):
            xg = XGraph(Context(config=prior), 8, cat_sizes=[0, 0, 0, 0, 3, 5, 8, 4])
            xg._run(400)                                     # fit pass
            for nd in xg.graph_fn.nodes_:
                nd.keep_writeback = True
            raw = xg._run(400)
        clamp = RobustClamp.fit(raw)
        xc = clamp.apply(raw)
        free = raw == xc                                     # rows where the clamp does not bind, per column
        n_cols, n_ok = 0, 0
        for host in xg.hosts:
            nd = xg.graph_fn.nodes_[host.node]
            for name, conv in zip(host.names, host.converters):
                j = xg.col_index[name]
                ow = observed_writeback(conv, xc[:, j:j + 1], prior)
                actual, rows = nd.writeback_[name], free[:, j]
                assert ow.shape == actual.shape, (name, ow.shape, actual.shape)
                if isinstance(conv, NumericalConverter):
                    num_stats[conv.mode_][0] += 1
                    num_stats[conv.mode_][1] += bool(torch.equal(ow[rows], actual[rows]))
                    continue
                n_cols += 1
                idx = raw[:, j].long()
                table = conv.disc.centers_ if mode.startswith("neighbor") else conv.disc.embeddings_
                if mode.endswith("_id"):
                    # substitution: the observed category's center / embedding, never the latent write-back
                    n_ok += bool(torch.equal(ow[rows], table[idx][rows]) and not torch.equal(ow, actual))
                else:
                    n_ok += bool(torch.equal(ow[rows], actual[rows]))
        what = ("equals the category's center/embedding (latent write-back replaced)" if mode.endswith("_id")
                else "equals the X-graph's write-back on unclamped rows")
        check(f"categorical {mode}: observed write-back {what}", n_cols > 0 and n_ok == n_cols,
              f"{n_ok}/{n_cols} columns")
    for m, (n_cols, n_ok) in num_stats.items():
        check(f"numerical {m}: observed write-back equals the X-graph's write-back on unclamped rows",
              n_cols > 0 and n_ok == n_cols, f"{n_ok}/{n_cols} columns")

    # Outlier rows: two raw values beyond the clamp bound look identical in X, so the target graph must see
    # identical inputs and (given the same U2) return identical y. A raw write-back would tell them apart.
    task, _ = small_task(3)
    tg, ci = task.target, task.x_graph.col_index
    used = {name for p, h in tg.input_hosts.items() if p in tg.needed_ for name in h.names}
    j = ci[sorted(used)[0]]
    up = task.x_clamp.upper[j]
    xa, xb = task.x_clamped[:20].clone(), task.x_clamped[:20].clone()
    xa[0, j], xb[0, j] = up + 1.0, up + 50.0                # two different raw outliers
    ia = tg.observed_inputs(task.x_clamp.apply(xa), ci)
    ib = tg.observed_inputs(task.x_clamp.apply(xb), ci)
    set_tape(tg.laws(), "record")
    ya = tg.sample(ia, 20)
    set_tape(tg.laws(), "replay")
    yb = tg.sample(ib, 20)
    set_tape(tg.laws(), None)
    check("outliers beyond the clamp: identical observed X gives identical inputs and identical y (same U2)",
          all(torch.equal(ia[p], ib[p]) for p in ia) and torch.equal(ya, yb))


# ------------------------------------------------------------------------------------------- 4. structure
# The target graph copies graph_scm's DAG sampler; X-input nodes sit on random positions with their incoming edges
# cut; y sits on a non-input node; the graph check (some X input is an ancestor of y) holds; and because X-input
# nodes carry hidden U2 dims, private noise always reaches y. Only y's ancestors are evaluated.


def test_structure():
    print("\n# 4. target-graph structure (12 tasks)")
    ok = dict(roots=True, y_off_input=True, check=True, noise=True, prune=True, hosts=True, width=True)
    sizes = []
    for t in range(12):
        task, _ = small_task(100 + t)
        tg = task.target
        pos = list(tg.input_hosts)
        ok["roots"] &= all(tg.dag[p] == [] for p in pos)
        ok["y_off_input"] &= tg.y_node not in pos
        ok["check"] &= graph_check(tg.dag, pos, tg.y_node) and any(p in tg.needed_ for p in pos)
        ok["noise"] &= any(isinstance(tg.nodes_[i], InputNodeFunction) for i in tg.needed_)
        ok["prune"] &= tg.needed_ == ancestors(tg.dag, tg.y_node)
        ok["hosts"] &= len({h.node for h in tg.input_hosts.values()}) == len(pos)
        blocks = tg.observed_inputs(task.x_clamped[:5], task.x_graph.col_index)
        ok["width"] &= all(blocks[p].shape[1] == sum(c.get_n_features() for c in tg.input_hosts[p].converters)
                           for p in blocks)
        sizes.append(task.diagnostics["y_ancestor_size"])
    check("X-input positions have no parents (incoming edges cut)", ok["roots"])
    check("y sits on a non-input node", ok["y_off_input"])
    check("graph check holds: an X-input node is an ancestor of y", ok["check"])
    check("private noise reaches y (an X-input node with hidden U2 dims is an ancestor of y)", ok["noise"])
    check("only ancestors of y are evaluated (pruning)", ok["prune"])
    check("each X-input node is a distinct feature-hosting node of the X-graph", ok["hosts"])
    check("an X-input block has the width its host's converters occupy", ok["width"],
          f"y-ancestor sizes {sorted(sizes)}")


# ------------------------------------------------------------------------------------------- 5. g
# g (TabICLv2's y post-processing: outlier clamp at mean +- 4 sigma, then standard scaling) is fitted once on the
# pool's y^(0) and applied unchanged to every replicate (design §1 step 6; proposal §4.3 "fit/apply").


def test_g():
    print("\n# 5. y transform g: tabicl's post-processing, fitted once, applied unchanged")
    rng = np.random.default_rng(5)
    Xh = torch.from_numpy(rng.standard_t(2, size=(500, 7))).float()  # heavy tails, so the clamp binds
    check("RobustClamp.fit/apply reproduces tabicl's outlier_removing bit for bit",
          torch.equal(RobustClamp.fit(Xh).apply(Xh), outlier_removing(Xh, threshold=4)))
    y = torch.from_numpy(rng.standard_t(2, size=800)).float()
    ref = standard_scaling(outlier_removing(y.view(-1, 1), threshold=4)).view(-1)
    check("YTransform.fit/apply reproduces tabicl's standard_scaling(outlier_removing(y)) bit for bit",
          torch.equal(YTransform.fit(y).apply(y), ref))

    task, _ = small_task(4)
    check("the task's y0 is g(y0_raw) with g fitted on the pool", torch.equal(task.y0, task.g.apply(task.y0_raw)))
    rows = torch.arange(task.shape.n_ctx, task.shape.N)
    inputs = task.target.observed_inputs(task.x_clamped[rows], task.x_graph.col_index)
    with task_rng(0, 0, 4, STREAM_REPLICATES):
        raw = task.target.sample(inputs, len(rows), 8)
    with task_rng(0, 0, 4, STREAM_REPLICATES):
        reps = task.replicates(rows, 8)
    refit = YTransform.fit(raw).apply(raw)
    check("replicates use the pool's g (bit-equal to g.apply on the raw draws) and differ from a refit",
          torch.equal(reps, task.g.apply(raw)) and not torch.equal(reps, refit))


# ------------------------------------------------------------------------------------------- 6. determinism
# Protocol §1.3: a task's randomness depends only on (seed, step, task_idx), never on which process builds it or on
# what was built before. This is what pairs A and B jobs on the same tasks and makes resuming exact.


def _det_worker(task_idx):
    torch.set_num_threads(1)
    task, _ = sample_task(7, 3, task_idx, stage=1, N=N, n_ctx=N_CTX, d=D)
    with task_rng(7, 3, task_idx, STREAM_REPLICATES):
        reps = task.replicates(torch.arange(N_CTX, N), 4)
    return task.X, task.y0, reps


def test_determinism():
    print("\n# 6. determinism per (seed, step, task_idx)")
    torch.set_num_threads(1)                                 # same intra-op threads as a DataLoader worker
    a = _det_worker(11)
    _det_worker(12)                                          # another task in between
    b = _det_worker(11)
    check("same (seed, step, task_idx) in-process gives bit-identical X, y0 and replicates",
          all(torch.equal(u, v) for u, v in zip(a, b)))
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as ex:
        res = list(ex.map(_det_worker, [12, 11]))
    check("same task built in another process (2-worker pool) is bit-identical",
          all(torch.equal(u, v) for u, v in zip(a, res[1])))
    check("a different task_idx gives a different task",
          not (a[0].shape == res[0][0].shape and torch.equal(a[0], res[0][0])))


# ------------------------------------------------------------------------------------------- 7. filters and shapes


def test_filters_and_shapes():
    print("\n# 7. filters and task-shape bookkeeping")
    X = torch.randn(100, 4)
    X[:, 1] = 3.0
    check("drop_constant_columns keeps exactly the non-constant columns",
          drop_constant_columns(X).tolist() == [True, False, True, True])
    check("an all-constant X keeps no column (the task is rejected)",
          not drop_constant_columns(torch.ones(50, 3)).any())

    # Predictability test (tabicl's should_filter): a learnable y passes, pure noise is rejected. The test is a
    # p-value at 0.05, so noise passes it ~5% of the time; 5 seeds, at least 4 rejections required.
    acc_signal, rej_noise = 0, 0
    for s in range(5):
        g = torch.Generator().manual_seed(s)
        Xs = torch.randn(512, 3, generator=g)
        y_sig = torch.sin(2 * Xs[:, 0]) + Xs[:, 1] ** 2 + 0.1 * torch.randn(512, generator=g)
        acc_signal += not predictability_rejects(Xs, y_sig, CFG.prior)
        rej_noise += predictability_rejects(Xs, torch.randn(512, generator=g), CFG.prior)
    check("predictability test accepts y = f(x) + small noise (5/5 seeds)", acc_signal == 5, f"{acc_signal}/5")
    check("predictability test rejects pure-noise y (>= 4/5 seeds)", rej_noise >= 4, f"{rej_noise}/5")
    check("all_finite catches NaN and inf",
          all_finite(torch.zeros(3)) and not all_finite(torch.tensor([0.0, float("nan")]))
          and not all_finite(torch.zeros(2), torch.tensor([float("inf")])))
    log = RejectionLog()
    log.accepted = 3
    log.reject("unpredictable")
    check("RejectionLog: rate = rejections / attempts", abs(log.rate - 0.25) < 1e-12, log.summary())

    # Task shapes: exact bookkeeping of design Training §1 for every stage.
    bad = 0
    for stage in STAGES:
        for i in range(300):
            with task_rng(0, stage, i):
                s = sample_task_shape(stage, CFG)
            bad += not (s.n_ctx + s.m == s.N and s.m_ext + s.m_hole + s.m_iid == s.m
                        and s.P == s.N + (s.T - s.m_ext) + (s.k - s.m_hole)
                        and s.m_ext <= s.T <= max(s.m_ext, 0.2 * s.N) and s.m_hole <= s.k <= max(s.m_hole, 0.2 * s.N)
                        and 1 <= s.d <= Prior.adjust_max_features(s.N, CFG.max_features)
                        and (stage != 1 or s.N == 1024))
    check("task shapes: N = n_ctx + m, m = m_ext + m_hole + m_iid, P = N + (T - m_ext) + (k - m_hole), "
          "region and feature bounds (900 shapes, 3 stages)", bad == 0, f"{bad} bad")

    # End to end: an accepted task has the documented shapes and a standardized y0.
    task, log = small_task(5)
    P = task.shape.P
    ok = (task.X.shape == (P, task.diagnostics["d_kept"]) and task.y0.shape == (P,) and all_finite(task.X, task.y0)
          and abs(float(task.y0.mean())) < 1e-3 and abs(float(task.y0.std()) - 1.0) < 1e-3)
    check("an accepted task: X (P, d_kept), y0 (P,), finite, y0 standardized", ok,
          f"P={P}, d_kept={task.diagnostics['d_kept']}; {log.summary()}")


def main():
    torch.manual_seed(0)
    for fn in (test_caching, test_replicates, test_exactness, test_structure, test_g, test_determinism,
               test_filters_and_shapes):
        guarded(fn)
    print(f"\n{'ALL OK' if not FAILS else f'{len(FAILS)} FAILED: ' + '; '.join(FAILS)}", flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
