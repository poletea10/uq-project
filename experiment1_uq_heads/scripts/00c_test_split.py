"""Gate for step 3 of the order of work (phase1_protocol.md §3, unit tests of §4.3): the split.

WHAT IS TESTED
    iltm2.prior._split   `make_split`: turns an accepted observed-cut task (a pool of P rows) into n_ctx context
                         rows and m queries of four types (extrapolation tail, kNN hole, i.i.d., duplicates), with
                         the P - N region rows discarded (design Training §1.1-1.2). Every training task, the
                         validation bank and the T2 test bank go through it, and it decides which queries are
                         "far from the context", i.e. where the epistemic head learns large divergences.

HOW
    Section 1-2 run on synthetic standardized X (the split only looks at X, so a Gaussian pool is a fair input) for
    many random task shapes of all three stages; sections 3-5 run on real small observed-cut tasks (N = 256); section
    6 checks edge cases and a stage-3-sized pool. Checks, following protocol §4.3:
      1. exact counts, and the four row sets partition the pool;
      2. geometry: the tail is the top-T rows along u, the hole is the k nearest remaining rows to its centre, and
         queries / context come from the right regions;
      3. duplicates copy x only: their source is a context row, and their replicates are fresh draws;
      4. the split never looks at y;
      5. determinism per (seed, step, task_idx) under STREAM_SPLIT;
      6. edge cases (no hole / no tail) and speed on a stage-3 pool.

OUTPUT
    One "OK  <check>" or "FAIL <check>" line per check. Exit code 1 if any check failed.

    sbatch experiment1_uq_heads/scripts/00c_test_split.sh
"""
import inspect
import sys
import time
import traceback
from dataclasses import replace

import torch

from iltm2.prior._cut_graph_scm import CutGraphConfig, CutGraphSCM
from iltm2.prior._dataset import STAGES, STREAM_REPLICATES, STREAM_SPLIT, sample_task_shape, task_rng
from iltm2.prior._split import DUP, EXT, HOLE, IID, make_split

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


def synthetic(stage, i, d=None):
    """A task shape from TabICLv2's stage samplers and a standardized Gaussian pool X of P rows."""
    with task_rng(1, stage, i):
        shape = sample_task_shape(stage, CFG) if d is None else sample_task_shape(stage, CFG, d=d)
        X = torch.randn(shape.P, shape.d)
    return shape, (X - X.mean(0)) / X.std(0)


def split_of(X, shape, i, seed=1):
    with task_rng(seed, shape.stage, i, STREAM_SPLIT):
        return make_split(X, shape, CFG.dup_fraction)


def small_task(t, cfg=CFG):
    with task_rng(2, 0, t):
        shape = sample_task_shape(1, cfg, N=256, n_ctx=200, d=6)
        task, _ = CutGraphSCM(cfg).sample(shape)
    return task


# ------------------------------------------------------------------------------------------- 1. counts
# The model must see exactly TabICLv2's sizes for the stage (n_ctx context rows, m queries), and every pool row must
# end up in exactly one place: context, query (before the duplicate overwrite) or discarded.


def test_counts():
    print("\n# 1. exact counts and a partition of the pool (synthetic pools, 60 shapes over 3 stages)")
    bad_counts, bad_partition, bad_dup = 0, 0, 0
    for stage in STAGES:
        for i in range(20):
            shape, X = synthetic(stage, i, d=8 if stage == 3 else None)
            sp = split_of(X, shape, i)
            qt = sp.query_type
            bad_counts += not (len(sp.ctx_rows) == shape.n_ctx and len(sp.query_rows) == shape.m
                               and len(qt) == shape.m and len(sp.discarded) == shape.P - shape.N)
            # before the duplicate overwrite: ext / hole / iid counts are exact
            before = sp.query_rows_before_dup
            in_tail = torch.isin(before, sp.tail_rows)
            in_hole = torch.isin(before, sp.hole_rows)
            bad_counts += not (int(in_tail.sum()) == shape.m_ext and int(in_hole.sum()) == shape.m_hole
                               and int((~in_tail & ~in_hole).sum()) == shape.m_iid)
            allrows = torch.cat([sp.ctx_rows, before, sp.discarded])
            bad_partition += not (len(allrows) == shape.P and torch.equal(torch.sort(allrows).values,
                                                                          torch.arange(shape.P)))
            expected_dup = int(CFG.dup_fraction * shape.m + 0.5)                          # round half up
            bad_dup += int((qt == DUP).sum()) != expected_dup
    check("context = n_ctx, queries = m (m_ext + m_hole + m_iid before duplicates), discarded = P - N", bad_counts == 0,
          f"{bad_counts} bad")
    check("context, queries (before duplicates) and discarded rows partition the pool", bad_partition == 0,
          f"{bad_partition} bad")
    check("duplicates = round(dup_fraction * m) query slots", bad_dup == 0, f"{bad_dup} bad")


# ------------------------------------------------------------------------------------------- 2. geometry
# The regions must be what design §1.1 says: the tail is the top-T rows along the random direction u, the hole is
# the centre and its k - 1 nearest neighbours among the rows left after the tail, extrapolation / hole queries come
# from their region, and the context never sees either region.


def test_geometry():
    print("\n# 2. region geometry (synthetic pools, 40 stage-1/2 shapes)")
    ok = dict(tail=True, hole=True, centre=True, origin=True, ctx=True, unit=True)
    n_checked = 0
    for stage in (1, 2):
        for i in range(20):
            shape, X = synthetic(stage, 100 + i)
            sp = split_of(X, shape, 100 + i)
            ok["unit"] &= abs(float(sp.u.norm()) - 1.0) < 1e-5
            if shape.T > 0:
                s = X @ sp.u
                tail = torch.zeros(shape.P, dtype=torch.bool)
                tail[sp.tail_rows] = True
                ok["tail"] &= bool(s[tail].min() >= s[~tail].max()) and abs(float(s[tail].min()) - sp.tail_threshold) < 1e-6
            if shape.k > 0:
                remaining = torch.ones(shape.P, dtype=torch.bool)
                remaining[sp.tail_rows] = False
                hole = torch.zeros(shape.P, dtype=torch.bool)
                hole[sp.hole_rows] = True
                dist = (X - X[sp.hole_centre]).square().sum(1)
                ok["hole"] &= bool((hole & ~remaining).sum() == 0)                       # hole comes after the tail
                others = remaining & ~hole
                ok["hole"] &= (not others.any()) or bool(dist[hole].max() <= dist[others].min())
                ok["centre"] &= bool(hole[sp.hole_centre]) and bool(remaining[sp.hole_centre])
            qt, qr = sp.query_type, sp.query_rows_before_dup                             # DUP slots: no constraint
            ext = torch.isin(qr, sp.tail_rows)
            hol = torch.isin(qr, sp.hole_rows)
            ok["origin"] &= bool(((qt == EXT) <= ext).all() and ((qt == HOLE) <= hol).all()
                                 and ((qt == IID) <= (~ext & ~hol)).all())
            ok["ctx"] &= not bool(torch.isin(sp.ctx_rows, torch.cat([sp.tail_rows, sp.hole_rows])).any())
            n_checked += 1
    check("u is a unit vector", ok["unit"])
    check("the tail is exactly the T rows with the largest score x.u (tail_threshold = smallest tail score)",
          ok["tail"])
    check("the hole is the k nearest rows to its centre among the rows left after the tail", ok["hole"])
    check("the hole centre is a remaining row and belongs to its own hole", ok["centre"])
    check("extrapolation queries come from the tail, hole queries from the hole, i.i.d. queries from neither",
          ok["origin"])
    check("no context row lies in the tail or the hole (the context never sees the regions)", ok["ctx"],
          f"{n_checked} splits")


# ------------------------------------------------------------------------------------------- 3. duplicates
# Design §1.2: a duplicate slot gets the x of a random context row; only x is copied. Its replicates are drawn from
# the target graph at that x with fresh U2 (the context row's label is never used as a target).


def test_duplicates():
    print("\n# 3. duplicates copy x only (real observed-cut tasks)")
    ok_src, ok_x, fresh_tasks, n_noisy = True, True, 0, 0
    for t in range(30):
        task = small_task(t)
        sp = split_of(task.X, task.shape, t, seed=2)
        dup = sp.query_type == DUP
        src = sp.query_rows[dup]
        ok_src &= bool(torch.isin(src, sp.ctx_rows).all())
        ok_x &= bool(torch.equal(task.X[sp.query_rows][dup], task.X[src]))
        if task.target.n_private_roots_used > 0 and n_noisy < 5:
            n_noisy += 1
            with task_rng(2, 0, t, STREAM_REPLICATES):
                Y = task.replicates(src, 16)                         # (n_dup, 16), g-transformed
            label = task.y0[src][:, None]
            varies = bool((Y != Y[:, :1]).any())                     # fresh U2: replicates differ across r
            not_label = not bool((Y == label).all())                 # the context label is not reused
            fresh_tasks += varies and not_label
    check("every duplicate's source row is a context row", ok_src)
    check("a duplicate's x equals its context row's x (only x is copied)", ok_x)
    check("on noisy tasks, duplicates' replicates are fresh draws (vary across r, are not the context label)",
          n_noisy > 0 and fresh_tasks == n_noisy, f"{fresh_tasks}/{n_noisy} noisy tasks")


# ------------------------------------------------------------------------------------------- 4. X only
# Protocol §4.3: "the split never looks at y". `make_split` receives no y at all; the check below makes the property
# explicit: the same X and RNG state give the same split whatever y the task carries.


def test_x_only():
    print("\n# 4. the split never looks at y")
    params = list(inspect.signature(make_split).parameters)
    check("make_split takes no y argument", params == ["X", "shape", "dup_fraction"], f"parameters {params}")
    task = small_task(0)
    a = split_of(task.X, task.shape, 0, seed=2)
    task.y0 = torch.randn_like(task.y0)                              # a different y on the same X
    task.y0_raw = torch.randn_like(task.y0_raw)
    b = split_of(task.X, task.shape, 0, seed=2)
    same = all(torch.equal(getattr(a, f), getattr(b, f)) for f in ("ctx_rows", "query_rows", "query_type",
                                                                   "discarded", "tail_rows", "hole_rows"))
    check("two different y vectors on the same X give identical indices", same and a.hole_centre == b.hole_centre)


# ------------------------------------------------------------------------------------------- 5. determinism


def test_determinism():
    print("\n# 5. determinism per (seed, step, task_idx) under STREAM_SPLIT")
    shape, X = synthetic(2, 7)
    a, b = split_of(X, shape, 7), split_of(X, shape, 7)
    with task_rng(1, shape.stage, 7, STREAM_REPLICATES):
        c = make_split(X, shape, CFG.dup_fraction)
    check("same (seed, step, task_idx, STREAM_SPLIT) gives an identical split",
          torch.equal(a.model_rows, b.model_rows) and torch.equal(a.query_type, b.query_type))
    check("another stream gives a different split", not torch.equal(a.model_rows, c.model_rows))


# ------------------------------------------------------------------------------------------- 6. edge cases, scale


def test_edges_and_scale():
    print("\n# 6. edge cases and scale")
    cfg_iid = replace(CFG, query_mix=(1.0, 0.0, 0.0))
    with task_rng(3, 1, 0):
        shape = sample_task_shape(1, cfg_iid)
    X = torch.randn(shape.P, shape.d)
    sp = split_of(X, shape, 0)
    check("query_mix (1, 0, 0): no tail, no hole, P = N, i.i.d. queries and duplicates only",
          shape.T == 0 and shape.k == 0 and shape.P == shape.N and len(sp.discarded) == 0
          and sp.hole_centre == -1 and bool(torch.isin(sp.query_type, torch.tensor([IID, DUP])).all()))
    big = None
    for i in range(200):                                             # a large stage-3 pool
        with task_rng(3, 3, i):
            s = sample_task_shape(3, CFG, d=10)
        if s.P >= 50_000:
            big = s
            break
    X = torch.randn(big.P, big.d)
    t0 = time.time()
    sp = split_of(X, big, 0)
    dt = time.time() - t0
    check("a stage-3 pool splits in seconds", dt < 10 and len(sp.model_rows) == big.N,
          f"P = {big.P}, N = {big.N}, {dt:.2f}s")


def main():
    torch.set_num_threads(1)
    for fn in (test_counts, test_geometry, test_duplicates, test_x_only, test_determinism, test_edges_and_scale):
        guarded(fn)
    print(f"\n{'ALL OK' if not FAILS else f'{len(FAILS)} FAILED: ' + '; '.join(FAILS)}", flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
