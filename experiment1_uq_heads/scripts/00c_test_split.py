"""Gate for step 3 of the order of work (phase1_protocol.md §3, unit tests of §4.3; transfer guide §5 Step 2.1): the
split, thinned regions and subsampled contexts.

WHAT IS TESTED
    iltm2.prior._split   `make_split`: turns an accepted observed-cut task (a pool of P rows) into n_ctx context
                         rows and m queries from three regions (extrapolation tail, kNN hole, i.i.d.) plus duplicates;
                         each carved region is empty or thinned (r of its non-query rows returned to the context), the
                         other region rows are discarded (design Training §1.1-1.2, guide §3.7). Every training task
                         and the validation bank go through it, and it decides which queries are "far from the
                         context", i.e. where the epistemic head learns large divergences.
                         `subsample_context`: stage 1's second pass, n' of the n_ctx context rows (guide §3.9).
    iltm2.prior._dataset `sample_task_shape` (thinning draws), `prior_config` ("iid"), `UQTask.subsampled`,
                         `CutGraphPrior`'s subsampled pass.

HOW
    Sections 1-2 run on synthetic standardized X (the split only looks at X, so a Gaussian pool is a fair input) for
    many random task shapes of all three stages; sections 3-5 run on real small observed-cut tasks (N = 256); section
    6 checks edge cases and a stage-3-sized pool; 7 the thinning draws; 8 the subsampled pass. Checks:
      1. exact counts with thinning: context n_ctx (n_ctx - r_ext - r_hole i.i.d. rows + the returned rows),
         queries m, discarded P - N, and the row sets partition the pool;
      2. geometry and labels: the tail is the top-T rows along u, the hole is the k nearest remaining rows to its
         centre; queries come from the right regions; the only context rows inside a region are its returned rows,
         chosen among its non-query rows; a region's queries are `_thinned` exactly when r > 0;
      3. duplicates copy x only: their source (`dup_src`) is a context row, and their replicates are fresh draws;
      4. the split and the subsampling never look at y;
      5. determinism per (seed, step, task_idx) under STREAM_SPLIT;
      6. edge cases (the "iid" prior: no tail, no hole, no duplicates, P = N) and speed on a stage-3 pool;
      7. thinning draws: half of the regions with non-query rows are thinned, the tail and the hole independently
         (a quarter of the tasks in each of both empty / tail thinned / hole thinned / both thinned), rho within the
         configured range up to rounding (a thinned region returns at least one row), r <= 0.04 N, sizes known
         before generation (P from the shape);
      8. the subsampled pass: n' of the context rows, uniformly, relative order kept; queries, their x and their
         replicates unchanged; a duplicate whose source was dropped takes its source row's type; a thinned region
         with no returned row left is empty; CutGraphPrior's stage-1 groups carry it with one n' in [64, n_ctx] per
         group, stages 2-3 do not.

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

from iltm2.prior._cut_graph_scm import CutGraphConfig, CutGraphSCM, NonFiniteReplicate
from iltm2.prior._dataset import (STAGES, STREAM_REPLICATES, STREAM_SPLIT, STREAM_SUB_ROWS, SUB_MIN_CTX,
                                  CutGraphPrior, TaskShape, prior_config, sample_task_shape, sample_uq_task, task_rng)
from iltm2.prior._split import (DUP, EXT, EXT_THIN, HOLE, HOLE_THIN, IID, QUERY_TYPES, make_split,
                                subsample_context)

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


def synthetic(stage, i, d=None, cfg=CFG):
    """A task shape from TabICLv2's stage samplers and a standardized Gaussian pool X of P rows."""
    with task_rng(1, stage, i):
        shape = sample_task_shape(stage, cfg) if d is None else sample_task_shape(stage, cfg, d=d)
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


def mask(rows, n):
    m = torch.zeros(n, dtype=torch.bool)
    m[rows] = True
    return m


# ------------------------------------------------------------------------------------------- 1. counts
# The model must see exactly TabICLv2's sizes for the stage (n_ctx context rows, m queries), and every pool row must
# end up in exactly one place: context (i.i.d. or returned), query (before the duplicate overwrite) or discarded.


def test_counts():
    print("\n# 1. exact counts with thinning, and a partition of the pool (synthetic pools, 90 shapes over 3 stages)")
    bad_counts, bad_partition, bad_dup, bad_origin, n_thin = 0, 0, 0, 0, 0
    for stage in STAGES:
        for i in range(30):
            shape, X = synthetic(stage, i, d=8 if stage == 3 else None)
            sp = split_of(X, shape, i)
            qt = sp.query_type
            n_thin += (shape.r_ext > 0) + (shape.r_hole > 0)
            bad_counts += not (len(sp.ctx_rows) == shape.n_ctx and len(sp.query_rows) == shape.m
                               and len(qt) == shape.m and len(sp.discarded) == shape.P - shape.N
                               and len(sp.returned_ext) == shape.r_ext and len(sp.returned_hole) == shape.r_hole)
            bad_origin += not (int((sp.ctx_origin == IID).sum()) == shape.n_ctx - shape.r_ext - shape.r_hole
                               and int((sp.ctx_origin == EXT_THIN).sum()) == shape.r_ext
                               and int((sp.ctx_origin == HOLE_THIN).sum()) == shape.r_hole)
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
    check("context = n_ctx, queries = m (m_ext + m_hole + m_iid before duplicates), discarded = P - N, returned = r",
          bad_counts == 0, f"{bad_counts} bad")
    check("context = n_ctx - r_ext - r_hole i.i.d. rows + r_ext tail rows + r_hole hole rows (ctx_origin)",
          bad_origin == 0, f"{bad_origin} bad")
    check("context, queries (before duplicates) and discarded rows partition the pool", bad_partition == 0,
          f"{bad_partition} bad; {n_thin} thinned regions among them")
    check("duplicates = round(dup_fraction * m) query slots", bad_dup == 0, f"{bad_dup} bad")


# ------------------------------------------------------------------------------------------- 2. geometry
# The regions must be what design §1.1 says: the tail is the top-T rows along the random direction u, the hole is
# the centre and its k - 1 nearest neighbours among the rows left after the tail, extrapolation / hole queries come
# from their region, and the context sees a region only through its returned rows.


def test_geometry():
    print("\n# 2. region geometry and labels (synthetic pools, 60 stage-1/2 shapes)")
    ok = dict(tail=True, hole=True, centre=True, origin=True, ctx=True, ret=True, unit=True, label=True)
    n_checked = 0
    for stage in (1, 2):
        for i in range(30):
            shape, X = synthetic(stage, 100 + i)
            sp = split_of(X, shape, 100 + i)
            ok["unit"] &= abs(float(sp.u.norm()) - 1.0) < 1e-5
            if shape.T > 0:
                s = X @ sp.u
                tail = mask(sp.tail_rows, shape.P)
                ok["tail"] &= bool(s[tail].min() >= s[~tail].max()) and abs(float(s[tail].min()) - sp.tail_threshold) < 1e-6
            if shape.k > 0:
                remaining = ~mask(sp.tail_rows, shape.P)
                hole = mask(sp.hole_rows, shape.P)
                dist = (X - X[sp.hole_centre]).square().sum(1)
                ok["hole"] &= bool((hole & ~remaining).sum() == 0)                       # hole comes after the tail
                others = remaining & ~hole
                ok["hole"] &= (not others.any()) or bool(dist[hole].max() <= dist[others].min())
                ok["centre"] &= bool(hole[sp.hole_centre]) and bool(remaining[sp.hole_centre])
            qt, qr = sp.query_type, sp.query_rows_before_dup                             # DUP slots: no constraint
            ext = torch.isin(qr, sp.tail_rows)
            hol = torch.isin(qr, sp.hole_rows)
            ok["origin"] &= bool((((qt == EXT) | (qt == EXT_THIN)) <= ext).all()
                                 and (((qt == HOLE) | (qt == HOLE_THIN)) <= hol).all()
                                 and ((qt == IID) <= (~ext & ~hol)).all())
            in_regions = sp.ctx_rows[torch.isin(sp.ctx_rows, torch.cat([sp.tail_rows, sp.hole_rows]))]
            returned = torch.cat([sp.returned_ext, sp.returned_hole])
            ok["ctx"] &= torch.equal(torch.sort(in_regions).values, torch.sort(returned).values)
            ok["ret"] &= (bool(torch.isin(sp.returned_ext, sp.tail_rows).all())
                          and bool(torch.isin(sp.returned_hole, sp.hole_rows).all())
                          and not bool(torch.isin(returned, qr).any()))
            for r, empty, thin in ((shape.r_ext, EXT, EXT_THIN), (shape.r_hole, HOLE, HOLE_THIN)):
                ok["label"] &= not bool((qt == (empty if r > 0 else thin)).any())
            n_checked += 1
    check("u is a unit vector", ok["unit"])
    check("the tail is exactly the T rows with the largest score x.u (tail_threshold = smallest tail score)",
          ok["tail"])
    check("the hole is the k nearest rows to its centre among the rows left after the tail", ok["hole"])
    check("the hole centre is a remaining row and belongs to its own hole", ok["centre"])
    check("extrapolation queries come from the tail, hole queries from the hole, i.i.d. queries from neither",
          ok["origin"])
    check("the only context rows inside a region are its returned rows", ok["ctx"], f"{n_checked} splits")
    check("returned rows are non-query rows of their own region", ok["ret"])
    check("a region's queries are _thinned exactly when r > 0 (empty-region types otherwise)", ok["label"])


# ------------------------------------------------------------------------------------------- 3. duplicates
# Design §1.2: a duplicate slot gets the x of a random context row; only x is copied. Its replicates are drawn from
# the target graph at that x with fresh U2 (the context row's label is never used as a target).


def test_duplicates():
    print("\n# 3. duplicates copy x only (real observed-cut tasks)")
    ok_src, ok_x, ok_ptr, fresh_tasks, n_noisy, n_flat = True, True, True, 0, 0, 0
    for t in range(30):
        task = small_task(t)
        sp = split_of(task.X, task.shape, t, seed=2)
        dup = sp.query_type == DUP
        src = sp.query_rows[dup]
        ok_src &= bool(torch.isin(src, sp.ctx_rows).all())
        ok_x &= bool(torch.equal(task.X[sp.query_rows][dup], task.X[src]))
        ok_ptr &= (bool((sp.dup_src[~dup] == -1).all()) and bool(torch.equal(sp.ctx_rows[sp.dup_src[dup]], src)))
        if task.target.n_private_roots_used > 0 and n_noisy < 5:
            # A noisy task can be locally deterministic (the private roots reach y only in some regions of x), so
            # the check runs on the duplicate rows whose x is noisy, as told by an INDEPENDENT replicate stream.
            try:
                with task_rng(2, 0, t, STREAM_REPLICATES):
                    Y = task.replicates(src, 16)                     # (n_dup, 16), g-transformed
                with task_rng(2, 0, t, STREAM_REPLICATES + 7):
                    Y_ref = task.replicates(src, 16)
            except NonFiniteReplicate:
                continue
            noisy_x = (Y_ref != Y_ref[:, :1]).any(dim=1)
            if not bool(noisy_x.any()):
                n_flat += 1
                continue
            n_noisy += 1
            Yn, label = Y[noisy_x], task.y0[src][noisy_x][:, None]
            varies = bool((Yn != Yn[:, :1]).any(dim=1).all())        # fresh U2: replicates differ across r
            not_label = not bool((Yn == label).all(dim=1).any())     # the context label is not reused
            fresh_tasks += varies and not_label
    check("every duplicate's source row is a context row", ok_src)
    check("a duplicate's x equals its context row's x (only x is copied)", ok_x)
    check("dup_src points at the copied context row (and is -1 elsewhere)", ok_ptr)
    check("on noisy tasks, at every duplicate whose x is noisy, the replicates are fresh draws (vary across r, are "
          "not the context label)", n_noisy > 0 and fresh_tasks == n_noisy,
          f"{fresh_tasks}/{n_noisy} noisy tasks ({n_flat} skipped: every duplicate in a locally deterministic region)")


# ------------------------------------------------------------------------------------------- 4. X only
# Protocol §4.3: "the split never looks at y". `make_split` and `subsample_context` receive no y at all; the check
# below makes the property explicit: the same X and RNG state give the same split whatever y the task carries.


def test_x_only():
    print("\n# 4. the split and the subsampling never look at y")
    params = list(inspect.signature(make_split).parameters)
    check("make_split takes no y argument", params == ["X", "shape", "dup_fraction"], f"parameters {params}")
    sparams = list(inspect.signature(subsample_context).parameters)
    check("subsample_context takes no y argument", sparams == ["split", "n_sub"], f"parameters {sparams}")
    task = small_task(0)
    a = split_of(task.X, task.shape, 0, seed=2)
    task.y0 = torch.randn_like(task.y0)                              # a different y on the same X
    task.y0_raw = torch.randn_like(task.y0_raw)
    b = split_of(task.X, task.shape, 0, seed=2)
    same = all(torch.equal(getattr(a, f), getattr(b, f)) for f in ("ctx_rows", "ctx_origin", "query_rows",
                                                                   "query_type", "dup_src", "discarded", "tail_rows",
                                                                   "hole_rows", "returned_ext", "returned_hole"))
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
    cfg_iid = prior_config("iid")
    ok = True
    for i in range(20):
        with task_rng(3, 1, i):
            shape = sample_task_shape(1, cfg_iid)
        X = torch.randn(shape.P, shape.d)
        with task_rng(3, 1, i, STREAM_SPLIT):
            sp = make_split(X, shape, cfg_iid.dup_fraction)
        ok &= (shape.T == 0 and shape.k == 0 and shape.r_ext == shape.r_hole == 0 and shape.P == shape.N
               and len(sp.discarded) == 0 and sp.hole_centre == -1 and bool((sp.query_type == IID).all())
               and bool((sp.ctx_origin == IID).all()))
    check("the iid prior: no tail, no hole, no thinning, no duplicates, P = N, only i.i.d. queries", ok,
          f"query_mix {cfg_iid.query_mix}, dup_fraction {cfg_iid.dup_fraction}")
    check("the carved prior is the default config", prior_config("carved") == CFG)
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


# ------------------------------------------------------------------------------------------- 7. thinning draws


def test_thinning_draws():
    print("\n# 7. thinning draws (4,000 shapes per stage, and the figures' stage-1 shape)")
    lo, hi = CFG.thin_rho_range
    cases = [(f"stage {st}", st, {}) for st in STAGES]
    cases.append(("figure shape (N 1,024, n_ctx 819, d 2)", 1, dict(n_ctx=819, d=2)))
    for name, stage, kw in cases:
        n_free = n_thin = 0
        bad_rho = bad_r = bad_p = 0
        joint, n_both = {}, 0                                # (tail thinned, hole thinned) over tasks with both free
        for i in range(4000):
            with task_rng(4, stage, i):
                s = sample_task_shape(stage, CFG, **kw)
            for T, mq, r in ((s.T, s.m_ext, s.r_ext), (s.k, s.m_hole, s.r_hole)):
                free = T - mq
                if free <= 0:
                    bad_r += r != 0
                    continue
                n_free += 1
                n_thin += r > 0
                if r > 0:                                    # realized rho = r / free, up to the rounding of rho free
                    bad_rho += not (r == 1 or lo * free - 0.5 - 1e-9 <= r <= hi * free + 0.5 + 1e-9) or r > free
                bad_r += r > max(1.0, 0.04 * s.N + 0.5)
            if s.T > s.m_ext and s.k > s.m_hole:
                key = (s.r_ext > 0, s.r_hole > 0)
                joint[key] = joint.get(key, 0) + 1
                n_both += 1
            bad_p += s.P != s.N + (s.T - s.m_ext - s.r_ext) + (s.k - s.m_hole - s.r_hole)
        share = n_thin / max(1, n_free)
        check(f"{name}: half of the regions with non-query rows are thinned (0.46-0.54)", 0.46 <= share <= 0.54,
              f"{share:.3f} of {n_free}")
        quarters = {k: joint.get(k, 0) / max(1, n_both) for k in ((False, False), (True, False), (False, True),
                                                                  (True, True))}
        check(f"{name}: tail and hole independent, a quarter each (0.21-0.29) of both empty / tail thinned / hole "
              f"thinned / both thinned", all(0.21 <= v <= 0.29 for v in quarters.values()),
              " / ".join(f"{v:.3f}" for v in quarters.values()) + f" of {n_both} tasks")
        check(f"{name}: realized rho within [{lo}, {hi}] up to rounding (or r = 1), r <= 0.04 N, P from the shape",
              bad_rho == 0 and bad_r == 0 and bad_p == 0, f"bad rho {bad_rho}, bad r {bad_r}, bad P {bad_p}")


# ------------------------------------------------------------------------------------------- 8. subsampled pass


def forced_split(thin_ext: bool, thin_hole: bool, i: int):
    """A stage-1 synthetic pool whose regions are thinned (or empty) as asked, by overriding the shape's r."""
    with task_rng(5, 1, i):
        s = sample_task_shape(1, CFG)
    r_ext = max(1, round(0.1 * (s.T - s.m_ext))) if thin_ext and s.T > s.m_ext else 0
    r_hole = max(1, round(0.1 * (s.k - s.m_hole))) if thin_hole and s.k > s.m_hole else 0
    s = replace(s, r_ext=r_ext, r_hole=r_hole)
    with task_rng(5, 1, i):
        X = torch.randn(s.P, s.d)
    with task_rng(5, 1, i, STREAM_SPLIT):
        return s, make_split(X, s, CFG.dup_fraction)


def test_subsampling():
    print("\n# 8. the subsampled pass")
    ok_keep, ok_dup, ok_thin, ok_kept_dup, n_lost, n_emptied = True, True, True, True, 0, 0
    for i in range(40):
        s, sp = forced_split(True, True, i)
        n_sub = [64, 100, s.n_ctx // 3, s.n_ctx][i % 4]
        with task_rng(5, 1, i, STREAM_SUB_ROWS):
            keep, qt = subsample_context(sp, n_sub)
        kept = mask(keep, s.n_ctx)
        ok_keep &= (len(keep) == n_sub and bool((keep[1:] > keep[:-1]).all()))
        dup = sp.query_type == DUP
        lost = dup & ~kept[sp.dup_src.clamp_min(0)]
        n_lost += int(lost.sum())
        expect = sp.ctx_origin[sp.dup_src[lost]].clone()
        for thin, empty in ((HOLE_THIN, HOLE), (EXT_THIN, EXT)):
            if not bool((kept & (sp.ctx_origin == thin)).any()):
                expect[expect == thin] = empty
        ok_dup &= torch.equal(qt[lost], expect)
        ok_kept_dup &= bool((qt[dup & ~lost] == DUP).all())
        for thin, empty in ((HOLE_THIN, HOLE), (EXT_THIN, EXT)):
            left = bool((kept & (sp.ctx_origin == thin)).any())
            was = sp.query_type == thin
            if was.any():
                ok_thin &= bool((qt[was] == (thin if left else empty)).all())
                n_emptied += not left
    check("n' context rows kept, increasing positions (relative order kept)", ok_keep)
    check("a duplicate whose source was dropped takes its source row's type (empty if its region lost every "
          "returned row)", ok_dup and n_lost > 0, f"{n_lost} relabelled slots")
    check("a duplicate whose source was kept stays a duplicate", ok_kept_dup)
    check("a thinned region with no returned row left becomes empty, otherwise stays thinned", ok_thin,
          f"{n_emptied} regions emptied by the subsampling")

    # through UQTask on real tasks: queries, their x and their replicates unchanged
    ok = True
    for t in range(6):
        task, _ = sample_uq_task(9, 0, t, stage=1, R=8, N=256, n_ctx=200, d=5)
        full = task.view()
        with task_rng(9, 0, t, STREAM_SUB_ROWS):
            sub = task.subsampled(80)
        m = task.split.query_rows.numel()
        ok &= (sub.X.shape[0] == 80 + m and sub.y_ctx.shape[0] == 80 and sub.Y is task.Y
               and torch.equal(sub.X[80:], full.X[200:])
               and bool(torch.isin(sub.y_ctx, full.y_ctx).all()))
    check("UQTask.subsampled: 80 context rows, every query row and its replicates unchanged", ok)

    for stage in (1, 2):
        prior = CutGraphPrior(11, stage, max_steps=1, R=4, batch_size=8 if stage == 1 else 2)
        g = prior.group(0, 0)
        if stage == 1:
            sub = g.sub
            ok = (sub is not None and sub.subsampled and SUB_MIN_CTX <= sub.train_size <= g.train_size
                  and sub.m == g.m and torch.equal(sub.y_reps, g.y_reps) and torch.equal(sub.task_idx, g.task_idx)
                  and torch.equal(sub.X[:, sub.train_size:], g.X[:, g.train_size:]) and len(g.passes) == 2)
            again = prior.group(0, 0)
            ok &= torch.equal(again.sub.X, sub.X) and torch.equal(again.sub.query_type, sub.query_type)
            check("stage 1: each group carries a subsampled pass, one n' in [64, n_ctx], same queries and replicates, "
                  "deterministic", ok, f"n_ctx {g.train_size} -> n' {sub.train_size if sub is not None else None}")
        else:
            check("stage 2: no subsampled pass", g.sub is None and len(g.passes) == 1)
    ns = [CutGraphPrior(12, 1, max_steps=1).group(0, gi).sub.train_size for gi in range(8)]
    check("n' varies across groups and stays in [64, n_ctx]", len(set(ns)) > 1 and min(ns) >= SUB_MIN_CTX, str(ns))
    check("six query types are defined", len(QUERY_TYPES) == 6, str(list(QUERY_TYPES.values())))


def main():
    torch.set_num_threads(1)
    for fn in (test_counts, test_geometry, test_duplicates, test_x_only, test_determinism, test_edges_and_scale,
               test_thinning_draws, test_subsampling):
        guarded(fn)
    print(f"\n{'ALL OK' if not FAILS else f'{len(FAILS)} FAILED: ' + '; '.join(FAILS)}", flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
