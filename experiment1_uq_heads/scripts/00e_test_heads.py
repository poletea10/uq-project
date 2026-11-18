"""Gate for step 7 of the order of work (phase1_protocol.md §3, unit tests of §4.5; transfer guide §5 Step 2.4): heads,
loss, metrics, validation.

WHAT IS TESTED
    iltm2.model._heads      Trunk, Head, HeadPair (arm, LR, p), Fork (B), softplus_inv, the seeded initialization,
                            build_a_job_heads (head LR x p)
    iltm2.train._loss       batch_targets, loss_weight, uq_loss, group_weight
    iltm2.train._metrics    head_metrics and reference_lines in both weightings, train_metrics (per n bin), EMA
    iltm2.train._validate   BankEvaluator (cached and recomputed inputs, guards), BestTracker (per weighting)
    iltm2.train.frozen      warmup_constants

CHECKS
    1. Initialization (protocol §4.5): the last layer's weights are zero and its bias is softplus^-1(mean target),
       so at step 0 every query of both arms gets the same output, the mean target; softplus^-1 is exact over
       1e-6..1e2 and floors non-positive targets; the initialization is a function of (seed, head) only (same for
       every LR and every p, shared by A and B, different between the A and E trunks and between seeds); pair ids
       are A_lr<LR>_p<P>.
    2. Head independence (protocol §4.5): an A job's 6 pairs (3 head LRs x p {0, 1}) share no parameter (nor do the A
       and E heads of a pair); each optimizer holds exactly its own pair's parameters; stepping one pair (a p = 0 and
       a p = 1 pair tried), with every pair's loss at its own p, leaves the other five bit-identical; the A loss gives
       no gradient to the E head and back.
    3. Loss (protocol §4.5, guide §3.6): with p = 0 the weight is exactly 1 and the loss is exactly the plain MSE;
       with p = 1 it is (e(q) + c)^-1, detached, and the loss is the weighted MSE; p > 0 without c is refused;
       lambda_E scales the E term; group weights recombine group means into the mean over all query rows (two
       passes of a group included); no gradient reaches the targets or e(q).
    4. Gradients (protocol §4.5), on a real backbone forward of small observed-cut tasks, for A and B: at step 0
       (zero last layer) only the last layer gets a gradient; after one step every head parameter does, and in B
       the fork; no backbone parameter ever gets one, its state-dict hash and its quantiles are unchanged; the fork's
       output equals h_star at step 0, it holds only copies (not the backbone's tensors), and its distance from
       initialization is 0 before and > 0 after a step.
    5. Metrics (guide §3.4): head_metrics agrees with the bank's reference lines in BOTH weightings (a constant head
       at the bank constant has the constant baseline's MSE / xMSE and skill 0, and at the weighted constant the
       weighted ones and wskill 0; the e(q) proxy's predictions reproduce its skill and its weighted proxy its
       wskill); the weighted constant is the w-weighted mean target, the best constant under that weighting; the
       metrics cover the six query types and the six n bins; train_metrics matches direct formulas, per n bin
       included; the EMA is bias-corrected and resumable.
    6. Validation on the smoke bank's mini-bank (8 small tasks; real backbone): cached and recomputed inputs give
       the same predictions for A and B; predictions line up with the query table (task 0 recomputed directly);
       recomputed e(q) equals the stored one; a head initialized at the bank constant reproduces the ref_mini
       constant-baseline lines; the mini-bank reports its n bins; the guards fire (wrong backbone sha256, altered
       stored e(q), best tracking on the mini-bank); BestTracker tracks each pair under its own weighting and
       resumes; warmup_constants gives the batch's plain and weighted mean targets; the task-record walk matches the
       query table on the smoke bank and on the real bank (no forward).

OUTPUT
    One "OK  <check>" or "FAIL <check>" line per check. Exit code 1 if any check failed. CPU, a few minutes.

    sbatch experiment1_uq_heads/scripts/00e_test_heads.sh [--skip-real-bank]
"""
import argparse
import math
import sys
import traceback

import torch
import torch.nn.functional as F
from torch import nn

from iltm2.model._backbone import DEFAULT_BACKBONE, FrozenBackbone, state_dict_sha256
from iltm2.model._heads import ARMS, HEADS, MIN_TARGET, Fork, HeadPair, build_a_job_heads, seeded, softplus_inv
from iltm2.prior._bank import BANK_DIR, SMOKE_BANK_DIR, BankQueries, iter_tasks
from iltm2.prior._dataset import UQBatch, sample_uq_task
from iltm2.prior._filter import RejectionLog
from iltm2.prior._split import QUERY_TYPES
from iltm2.train._loss import Targets, batch_targets, group_weight, loss_weight, uq_loss
from iltm2.train._metrics import (EMA, N_BIN_NAMES, N_BINS, const_values, eq_weight, head_metrics, n_bin_index,
                                  proxy_predictions, reference_lines, train_metrics, weighted_mean)
from iltm2.train._validate import MINI_SLICES, BankEvaluator, BestTracker
from iltm2.train.frozen import warmup_constants

FAILS = []
ARGS = None
C = 0.05                        # a weight floor for the synthetic sections


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


def maxdiff(a, b):
    return float((a.double() - b.double()).abs().max())


def worst(values):
    """max |value|, NaN-aware: any NaN gives inf, so a NaN can never pass a '< tol' check."""
    values = list(values)
    return max(abs(v) if math.isfinite(v) else math.inf for v in values) if values else math.inf


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    return False


# ------------------------------------------------------------------------------------------- shared fixtures

_CACHE = {}


def backbone():
    """One frozen backbone with the fork input (k = 2), shared by the sections."""
    if "bb" not in _CACHE:
        _CACHE["bb"] = FrozenBackbone(ARGS.checkpoint, "cpu", fork_k=2)
    return _CACHE["bb"]


def batch():
    """A group of 2 small observed-cut tasks (N = 256, n_ctx = 200, d = 6, R = 8), its backbone output and its
    targets. At least one noisy task is wanted so t_A is not identically 0: tasks are drawn until one is noisy."""
    if "batch" not in _CACHE:
        tasks, idx = [], 0
        while len(tasks) < 2 and idx < 20:
            t = sample_uq_task(7, 0, idx, stage=1, R=8, N=256, n_ctx=200, d=6)[0]
            if tasks or not t.deterministic:          # the first task kept is a noisy one
                tasks.append(t)
            idx += 1
        g = UQBatch.stack(tasks, stage=1, step=0, group=0, task_idx=list(range(len(tasks))), log=RejectionLog(),
                          gen_time=0.0)
        o = backbone()(g.X, g.y_ctx)
        _CACHE["batch"] = (g, o, batch_targets(o.q, g.y_reps))
    return _CACHE["batch"]


def input_of(arm, o, fork=None, grad=False):
    """The input a pair of `arm` reads: h_star (A) or the fork's output (B, with or without autograd)."""
    if arm == "A":
        return o.h_star
    with torch.set_grad_enabled(grad):
        return fork(o.fork_input, o.train_size)


def params_equal(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))


# ------------------------------------------------------------------------------------------- 1. initialization


def test_init():
    print("\n# 1. initialization")
    ys = torch.logspace(-6, 2, 41, dtype=torch.float64)
    back = F.softplus(torch.tensor([softplus_inv(y) for y in ys], dtype=torch.float64))
    rel = float(((back - ys) / ys).abs().max())
    check("softplus(softplus_inv(y)) = y on 1e-6..1e2", rel < 1e-10, f"max rel err {rel:.1e}")
    check("non-positive targets are floored at MIN_TARGET",
          softplus_inv(0.0) == softplus_inv(-0.3) == softplus_inv(MIN_TARGET))

    bb = backbone()
    g, o, T = batch()
    mA, mE = float(T.t_A.mean()), float(T.t_E.mean())
    print(f"info batch mean targets: t_A {mA:.4g}, t_E {mE:.4g}", flush=True)
    fork = Fork(bb, k=2)
    check("the arms are A and B", ARMS == ("A", "B"), str(ARMS))
    for arm in ARMS:
        pair = HeadPair(arm, 3e-4, seed=0)
        check(f"{arm}: last layer weights are zero (both heads)",
              all(bool((pair.head(h).trunk.fc2.weight == 0).all()) for h in HEADS))
        pair.init_output_bias(mA, mE)
        with torch.no_grad():
            A, E = pair(input_of(arm, o, fork))
        for h, out, m in (("A", A, mA), ("E", E, mE)):
            target = max(m, MIN_TARGET)
            same = bool((out == out.flatten()[0]).all())
            err = abs(float(out.flatten()[0]) - target) / target
            check(f"{arm}: step-0 {h} output identical for every query and = mean target", same and err < 1e-5,
                  f"value {float(out.flatten()[0]):.6g} vs {target:.6g} (rel {err:.1e}); shape {tuple(out.shape)}")

    p = {k: HeadPair(k[0], k[1], k[2], p=k[3]) for k in [("A", 3e-4, 0, 0.0), ("A", 3e-3, 0, 0.0), ("A", 3e-4, 0, 1.0),
                                                         ("A", 3e-4, 1, 0.0), ("B", 3e-4, 0, 0.0)]}
    P = lambda k: list(p[k].parameters())  # noqa: E731
    check("same seed: identical initialization for every LR and every p",
          params_equal(P(("A", 3e-4, 0, 0.0)), P(("A", 3e-3, 0, 0.0)))
          and params_equal(P(("A", 3e-4, 0, 0.0)), P(("A", 3e-4, 0, 1.0))))
    check("A and B share the initialization (B starts where A starts)",
          params_equal(P(("A", 3e-4, 0, 0.0)), P(("B", 3e-4, 0, 0.0))))
    check("a different seed gives a different initialization", not params_equal(P(("A", 3e-4, 0, 0.0)),
                                                                                P(("A", 3e-4, 1, 0.0))))
    a = p[("A", 3e-4, 0, 0.0)]
    check("the A and E trunks are initialized differently", not torch.equal(a.A.trunk.fc1.weight,
                                                                             a.E.trunk.fc1.weight))
    ids = [x.head_id for x in build_a_job_heads(0)]
    check("an A job's pair ids are A_lr<LR>_p<P> over head LR {3e-4, 1e-3, 3e-3} x p {0, 1} (transfer guide §4)",
          ids == ["A_lr3e-4_p0", "A_lr1e-3_p0", "A_lr3e-3_p0", "A_lr3e-4_p1", "A_lr1e-3_p1", "A_lr3e-3_p1"], str(ids))
    check("only p > 0 pairs are weighted", [x.weighted for x in build_a_job_heads(0)] == [False] * 3 + [True] * 3)


# ------------------------------------------------------------------------------------------- 2. independence


def test_independence():
    print("\n# 2. head independence in an A job (head LR x p)")
    g, o, T = batch()
    pairs = build_a_job_heads(seed=0)
    check("an A job has 6 pairs with distinct ids (3 head LRs x 2 p)",
          len(pairs) == 6 and len({p.head_id for p in pairs}) == 6)
    ptrs = [{h: {t.data_ptr() for t in p.head(h).parameters()} for h in HEADS} for p in pairs]
    flat = [s for d in ptrs for s in d.values()]
    disjoint = all(not (flat[i] & flat[j]) for i in range(len(flat)) for j in range(i + 1, len(flat)))
    check("no parameter shared between pairs, nor between the A and E heads of a pair", disjoint)
    opts = [p.make_optimizer() for p in pairs]
    check("each optimizer holds exactly its own pair's parameters",
          all({id(t) for gr in op.param_groups for t in gr["params"]} == {id(t) for t in p.parameters()}
              for p, op in zip(pairs, opts)))
    for p in pairs:
        p.init_output_bias(float(T.t_A.mean()), float(T.t_E.mean()))
    for k in (1, 4):                                       # one p = 0 pair, one p = 1 pair
        before = [[t.detach().clone() for t in p.parameters()] for p in pairs]
        for p, op in zip(pairs, opts):                     # every pair computes its loss at its own p
            op.zero_grad(set_to_none=True)
            A, E = p(o.h_star)
            uq_loss(A, E, T.t_A, T.t_E, o.e_q, p=p.p, c=C).loss.backward()
        opts[k].step()
        moved = [not params_equal(b, [t.detach() for t in p.parameters()]) for b, p in zip(before, pairs)]
        check(f"stepping pair {pairs[k].head_id} moves it and leaves the other five bit-identical",
              moved[k] and not any(m for i, m in enumerate(moved) if i != k), f"moved: {moved}")
        for op in opts:
            op.zero_grad(set_to_none=True)
    for p in (pairs[0], pairs[4]):
        for h, other in (("A", "E"), ("E", "A")):
            p.zero_grad(set_to_none=True)
            out = dict(zip(HEADS, p(o.h_star)))
            t = T.t_A if h == "A" else T.t_E
            ((out[h] - t.float()) ** 2).mean().backward()
            check(f"{p.head_id}: the {h} loss gives no gradient to the {other} head",
                  all(t.grad is None for t in p.head(other).parameters())
                  and any(t.grad is not None for t in p.head(h).parameters()))
        p.zero_grad(set_to_none=True)


# ------------------------------------------------------------------------------------------- 3. loss


def test_loss():
    print("\n# 3. loss")
    gen = torch.Generator().manual_seed(0)
    n = 200
    e = torch.rand(n, generator=gen, dtype=torch.float64) + 1e-3
    A, E = torch.rand(n, generator=gen), torch.rand(n, generator=gen)
    tA = torch.rand(n, generator=gen, dtype=torch.float64).requires_grad_(True)
    tE = (torch.rand(n, generator=gen, dtype=torch.float64) - 0.3).requires_grad_(True)
    check("p = 0: the weight is exactly 1", torch.equal(loss_weight(e, 0.0), torch.ones_like(e)))
    check("p = 0: the weight is exactly 1 whatever c is given", torch.equal(loss_weight(e, 0.0, c=C),
                                                                           torch.ones_like(e)))
    e_req = e.clone().requires_grad_(True)
    w1 = loss_weight(e_req, 1.0, c=C)
    check("p = 1: w = (e(q) + c)^-1, detached from e(q)",
          torch.allclose(w1, (e + C) ** -1, rtol=1e-12, atol=0) and not w1.requires_grad)
    check("p = 1: the loss weight equals the weighted metrics' weight", torch.allclose(w1, eq_weight(e, C),
                                                                                      rtol=1e-12, atol=0))
    check("p > 0 without a floor c is refused", raises(lambda: loss_weight(e, 1.0), AssertionError))
    out = uq_loss(A, E, tA, tE, e)
    check("p = 0: the loss is exactly the plain MSE of each head",
          torch.equal(out.loss_A, ((A - tA.float()) ** 2).mean()) and torch.equal(out.loss_E,
                                                                                  ((E - tE.float()) ** 2).mean()))
    out1 = uq_loss(A, E, tA, tE, e, p=1.0, c=C)
    w = ((e + C) ** -1).float()
    check("p = 1: the loss is the weighted MSE", torch.allclose(out1.loss_A, (w * (A - tA.float()) ** 2).mean(),
                                                               rtol=1e-6)
          and torch.allclose(out1.loss_E, (w * (E - tE.float()) ** 2).mean(), rtol=1e-6))
    half = uq_loss(A, E, tA, tE, e, lambda_E=0.5)
    check("lambda_E scales only the E term", torch.allclose(half.loss, half.loss_A + 0.5 * half.loss_E, rtol=1e-7)
          and torch.equal(half.loss_A, out.loss_A))
    A_req = A.clone().requires_grad_(True)
    uq_loss(A_req, E, tA, tE, e_req, p=1.0, c=C).loss.backward()
    check("no gradient reaches the targets or e(q)", tA.grad is None and tE.grad is None and e_req.grad is None
          and A_req.grad is not None)
    sizes, x = [30, 50, 120, 30, 50, 120], torch.rand(400, generator=gen, dtype=torch.float64)
    parts, lo = [], 0
    for m in sizes:                                        # 3 groups x 2 passes
        parts.append(group_weight(m, 400) * float(x[lo:lo + m].mean()))
        lo += m
    check("group weights recombine group-pass means into the mean over all query rows (both passes)",
          abs(sum(parts) - float(x.mean())) < 1e-12)


# ------------------------------------------------------------------------------------------- 4. gradients


def test_gradients():
    print("\n# 4. gradients reach the heads (and the fork) and nothing else")
    bb = backbone()
    g, o, T = batch()
    h0, q0 = state_dict_sha256(bb.model), o.q.clone()
    for arm in ARMS:
        pair = HeadPair(arm, 3e-4, seed=0, p=1.0)
        pair.init_output_bias(float(T.t_A.mean()), float(T.t_E.mean()))
        fork = Fork(bb, k=2) if arm == "B" else None
        groups = [{"params": list(pair.parameters()), "lr": 1e-3}]
        if fork is not None:
            with torch.no_grad():
                h_fork = fork(o.fork_input, o.train_size)
            close = maxdiff(h_fork, o.h_star)
            check("B: the fork's output at step 0 equals h_star", close <= 1e-5, f"max |diff| {close:.1e}")
            bb_ptrs = {t.data_ptr() for t in bb.model.parameters()}
            check("B: the fork holds copies only, and its state dict has only its blocks and LN",
                  not ({t.data_ptr() for t in fork.parameters()} & bb_ptrs)
                  and all(k.startswith(("blocks.", "ln.")) for k in fork.state_dict()))
            check("B: distance from initialization is 0 before any step", fork.distance_from_init() == 0.0)
            groups.append({"params": list(fork.parameters()), "lr": 1e-4})
        opt = torch.optim.AdamW(groups, weight_decay=1e-5)
        trainable = [t for gr in groups for t in gr["params"]]
        for step in (0, 1):
            opt.zero_grad(set_to_none=True)
            A, E = pair(input_of(arm, o, fork, grad=True))
            uq_loss(A, E, T.t_A, T.t_E, o.e_q, p=pair.p, c=C).loss.backward()
            has = all(t.grad is not None for t in trainable)
            last = [t for h in HEADS for t in pair.head(h).trunk.fc2.parameters()]
            upstream = [t for t in trainable if all(t is not x for x in last)]
            if step == 0:
                check(f"{arm} step 0: every trainable parameter has a gradient tensor", has)
                check(f"{arm} step 0: the last layer gets a nonzero gradient, everything upstream exactly zero "
                      f"(zero-initialized last layer)", all(bool(t.grad.abs().sum() > 0) for t in last)
                      and all(bool((t.grad == 0).all()) for t in upstream))
            else:
                head_params = list(pair.parameters())
                nz = [bool(t.grad.abs().sum() > 0) for t in head_params]
                check(f"{arm} after one step: every head parameter gets a nonzero gradient", has and all(nz),
                      f"{sum(nz)}/{len(nz)} tensors")
                if fork is not None:
                    fz = [bool(t.grad.abs().sum() > 0) for t in fork.parameters()]
                    gn = float(torch.sqrt(sum((t.grad ** 2).sum() for t in fork.parameters())))
                    check("B after one step: the fork gets a gradient", gn > 0,
                          f"grad norm {gn:.3g}; {sum(fz)}/{len(fz)} tensors nonzero")
            opt.step()
        check(f"{arm}: no backbone parameter has a gradient", all(t.grad is None for t in bb.model.parameters()))
        if fork is not None:
            check("B: distance from initialization > 0 after a step", fork.distance_from_init() > 0,
                  f"{fork.distance_from_init():.3g}")
    check("backbone state-dict hash unchanged by all the steps", state_dict_sha256(bb.model) == h0)
    check("backbone quantiles bit-identical after all the steps", torch.equal(bb(g.X, g.y_ctx).q, q0))


# ------------------------------------------------------------------------------------------- 5. metrics


def synthetic_queries(n_tasks=6, m=600, seed=0):
    """A query table with real signal: a true A and E per query (partly a function of e(q), so the e(q) proxy has
    skill), and two independent noisy halves around it. Without signal the constant baseline's xMSE is ~0 and can
    be negative in a slice, where skill is undefined (NaN). Every query type and every n bin is populated."""
    gen = torch.Generator().manual_seed(seed)
    n = n_tasks * m
    r = lambda: torch.rand(n, generator=gen)  # noqa: E731
    e_q = r() + 0.01
    s_A, s_E = 0.5 * e_q + 0.1 * r(), 0.2 * e_q + 0.3 * r()
    tAa, tAb = s_A + 0.2 * (r() - 0.5), s_A + 0.2 * (r() - 0.5)
    tEa, tEb = s_E + 0.3 * (r() - 0.5), s_E + 0.3 * (r() - 0.5)
    return BankQueries(task_id=torch.arange(n_tasks).repeat_interleave(m),
                       bin_idx=torch.zeros(n, dtype=torch.int8),
                       query_type=torch.tensor(list(QUERY_TYPES))[torch.randint(0, len(QUERY_TYPES), (n,),
                                                                                generator=gen)].to(torch.int8),
                       n_bin=torch.randint(0, len(N_BINS), (n,), generator=gen).to(torch.int8),
                       e_q=e_q, t_A=(tAa + tAb) / 2, t_E=(tEa + tEb) / 2, t_A_a=tAa, t_A_b=tAb,
                       t_E_a=tEa, t_E_b=tEb, score=r())


def test_metrics():
    print("\n# 5. metrics, both weightings")
    Q = synthetic_queries()
    refs = reference_lines(Q, C)
    consts = const_values(Q, C)
    w = eq_weight(Q.e_q, C)
    best_w = weighted_mean(Q.t_A, w)
    grid = torch.linspace(best_w - 0.05, best_w + 0.05, 11, dtype=torch.float64)
    wmse_grid = [float((w * (Q.t_A.double() - c) ** 2).mean()) for c in grid]
    check("the weighted constant is the w-weighted mean target, the best constant under that weighting",
          abs(consts["wA"] - best_w) < 1e-12 and min(range(11), key=lambda i: wmse_grid[i]) == 5
          and abs(refs["ref/wconst_value_A"] - consts["wA"]) < 1e-12, f"{consts['wA']:.5g} vs unweighted "
                                                                    f"{consts['A']:.5g}")
    for pre, name in (("", "unweighted"), ("w", "weighted")):
        const = {h: torch.full((len(Q),), consts[f"{pre}{h}"]) for h in HEADS}
        mc = head_metrics(const, Q, refs, C)
        slices = sorted({k.rsplit("/", 1)[-1] for k in mc})
        wv = worst(x for h in HEADS for s in slices
                   for x in (mc[f"val/{pre}xmse_{h}/{s}"] - refs[f"ref/{pre}const_xmse_{h}/{s}"],
                             mc[f"val/{pre}mse_{h}/{s}"] - refs[f"ref/{pre}const_mse_{h}/{s}"],
                             mc[f"val/{pre}skill_{h}/{s}"]))
        check(f"{name}: a constant head at the bank constant = the constant-baseline lines, {pre}skill 0, every slice",
              wv < 1e-6, f"{len(slices)} slices; worst |diff| {wv:.1e}")
    proxy = proxy_predictions(Q, C)
    for pre, name in (("", "unweighted"), ("w", "weighted")):
        mp = head_metrics({h: proxy[f"{pre}{h}"].float() for h in HEADS}, Q, refs, C)
        wv = worst(mp[f"val/{pre}skill_{h}/{s}"] - refs[f"ref/{pre}eq_proxy_skill_{h}/{s}"]
                   for h in HEADS for s in slices)
        sk = " ".join(f"{h} {refs[f'ref/{pre}eq_proxy_skill_{h}/all']:.3f}" for h in HEADS)
        check(f"{name}: the e(q) proxy's predictions reproduce its reference skill, every slice", wv < 1e-5,
              f"worst |diff| {wv:.1e}; pooled proxy skill {sk}")
    expect = {"all", *QUERY_TYPES.values(), *N_BIN_NAMES}
    check("the slices are 'all', the six query types and the six n bins", set(slices) == expect
          and len(expect) == 13, f"{sorted(set(slices) ^ expect)} differ")
    mc = head_metrics({h: torch.full((len(Q),), consts[h]) for h in HEADS}, Q, refs, C)
    wv = worst(mc[f"val/ratio_tA_eq/{s}"] - refs[f"ref/ratio_tA_eq/{s}"] for s in slices)
    check("val/ratio_tA_eq matches the reference ratio", wv < 1e-9)
    check("every head-metric value is finite", all(math.isfinite(v) for v in mc.values()))
    only = head_metrics({h: torch.full((len(Q),), consts[h]) for h in HEADS}, Q, refs, C, slices=("all",))
    check("slices=('all',) reports only the pooled slice", all(k.endswith("/all") for k in only) and len(only) > 0)
    check("the bins: n_le256 holds 64-256, n_256-512 holds 257-512",
          n_bin_index(64) == 0 and n_bin_index(256) == 0 and n_bin_index(257) == 1 and n_bin_index(60000) == 5)

    gen = torch.Generator().manual_seed(1)
    A, E = torch.rand(50, generator=gen), torch.rand(50, generator=gen)
    T = Targets(*(torch.rand(50, generator=gen, dtype=torch.float64) for _ in range(6)))
    nb = torch.tensor([0] * 20 + [2] * 30)
    tm = train_metrics(A, E, T, torch.rand(50, generator=gen), n_bin=nb)
    ok = (abs(tm["train/mse_A"] - float(((A.double() - T.t_A) ** 2).mean())) < 1e-12 and
          abs(tm["train/xmse_E"] - float(((E.double() - T.t_E_a) * (E.double() - T.t_E_b)).mean())) < 1e-12)
    m0 = nb == 0
    ok_bin = (abs(tm["train/xmse_A/n_le256"] - float(((A[m0].double() - T.t_A_a[m0]) * (A[m0].double()
                                                                                        - T.t_A_b[m0])).mean())) < 1e-12
              and "train/xmse_A/n_512-1K" in tm and "train/xmse_A/n_256-512" not in tm)
    check("train_metrics: mse and xmse match their formulas", ok)
    check("train_metrics: xmse per n bin, only the populated bins", ok_bin)

    ema = EMA(0.99)
    vals = [ema.update(0.37) for _ in range(5)]
    check("EMA of a constant is that constant from the first update (bias correction)",
          all(abs(v - 0.37) < 1e-12 for v in vals))
    e2 = EMA(0.5)
    for x in (1.0, 2.0, 3.0):
        e2.update(x)
    manual = (0.25 * 0.5 * 1 + 0.5 * 0.5 * 2 + 0.5 * 3) / (1 - 0.5 ** 3)
    e3 = EMA()
    e3.load_state_dict(e2.state_dict())
    e2.reset()
    check("EMA matches the bias-corrected formula, resumes from its state, resets per stage",
          abs(e3.get() - manual) < 1e-12 and e2.n == 0 and e2.get() != e2.get())


# ------------------------------------------------------------------------------------------- 6. validation


def test_validation():
    print("\n# 6. validation on the smoke bank's mini-bank")
    root = ARGS.smoke_bank
    from pathlib import Path
    root = Path(root)
    check("the smoke bank exists", (root / "manifest.json").exists(), str(root))
    if not (root / "manifest.json").exists():
        return
    bb = backbone()
    ev_c = BankEvaluator(root, bb, mini=True, cache=("h", "fork"))
    ev_r = BankEvaluator(root, bb, mini=True)
    check("recomputed e(q) matches the stored e(q) on every mini-bank task (within the guard's tolerance)",
          ev_c.max_eq_scaled <= 1e-3,
          f"{len(ev_c.task_ids)} tasks, {len(ev_c.Q)} queries; max |diff| {ev_c.max_eq_diff:.1e} = "
          f"{ev_c.max_eq_scaled:.1e} of the task's mean e(q) (bank built on GPU, recomputed here on CPU: "
          f"round-off, not a different forward)")
    check("the bank carries a positive weight floor c", ev_c.c > 0, f"c = {ev_c.c:.4g}")

    pairs = [HeadPair("A", 3e-4, seed=0, p=0.0), HeadPair("A", 3e-4, seed=0, p=1.0), HeadPair("B", 3e-4, seed=0)]
    with seeded(11), torch.no_grad():                       # nonzero last layers: predictions vary per query
        for p in pairs:
            for h in HEADS:
                nn.init.normal_(p.head(h).trunk.fc2.weight, std=0.05)
    fork = Fork(bb, k=2)
    pc, pr = ev_c.predict(pairs, fork), ev_r.predict(pairs, fork)
    for p in pairs:
        d = max(maxdiff(pc[p.head_id][h], pr[p.head_id][h]) for h in HEADS)
        spread = float(pc[p.head_id]["A"].std())
        check(f"{p.head_id}: cached and recomputed inputs give the same predictions", d <= 1e-6,
              f"max |diff| {d:.1e}; {len(pc[p.head_id]['A'])} predictions (std {spread:.3g})")
    rec0 = next(iter_tasks(root, mini=True))
    with torch.no_grad():
        o0 = bb(rec0["X"][None], rec0["y_ctx"][None])
        direct = {p.head_id: p(input_of(p.arm, o0, fork)) for p in pairs}
    m0 = rec0["m"]
    d = max(maxdiff(direct[p.head_id][i][0], pr[p.head_id][h][:m0]) for p in pairs for i, h in enumerate(HEADS))
    check("predictions line up with the query table (task 0 recomputed directly)", d <= 1e-6,
          f"m = {m0}; max |diff| {d:.1e}")

    refs = ev_c.refs
    cpair = HeadPair("A", 3e-4, seed=0)
    cpair.init_output_bias(refs["ref_mini/const_value_A"], refs["ref_mini/const_value_E"])
    out = ev_c.evaluate([cpair])
    hid = cpair.head_id
    ok = all(abs(out[f"{hid}/mini/{k}_{h}/all"] / refs[f"ref_mini/const_{k}_{h}/all"] - 1) < 1e-4
             for h in HEADS for k in ("mse", "xmse"))
    check("a head at the bank constant reproduces the ref_mini constant-baseline lines", ok,
          " ".join(f"{h}: mse {out[f'{hid}/mini/mse_{h}/all']:.5g} vs {refs[f'ref_mini/const_mse_{h}/all']:.5g}"
                   for h in HEADS))
    got = sorted({k.rsplit("/", 1)[-1] for k in out if k.startswith(f"{hid}/mini/")})
    present = [s for s in MINI_SLICES if f"ref_mini/count/{s}" in refs]
    check("the mini-bank reports the pooled slice and every n bin it populates, both weightings",
          got == sorted(present) and all(f"{hid}/mini/wxmse_A/{s}" in out for s in present), f"{got}")
    check("evaluate logs the reference lines with the head metrics, all finite",
          any(k.startswith("ref_mini/") for k in out) and all(math.isfinite(v) for v in out.values()))
    check("best checkpoints are refused on the mini-bank",
          raises(lambda: ev_c.evaluate([cpair], best=BestTracker(), step=0), AssertionError))

    sha = bb.sha256
    bb.sha256 = "0" * 64
    try:
        check("a backbone with another sha256 is refused", raises(lambda: BankEvaluator(root, bb, mini=True),
                                                                 AssertionError))
    finally:
        bb.sha256 = sha
    ev_bad = BankEvaluator(root, bb, mini=True)
    ev_bad.Q.e_q[0] += 1.0
    check("an altered stored e(q) is detected at the forward", raises(lambda: ev_bad.predict(pairs[:1]),
                                                                       RuntimeError))

    bt = BestTracker()
    seq = [(0, 0.5, 0.9, 5.0, 2.0), (1, 0.4, 1.0, 6.0, 1.0), (2, 0.45, 0.8, 4.0, 3.0)]
    flags0, flags1 = [], []
    for s, a, e, wa, we in seq:
        m = {"val/mse_A/all": a, "val/mse_E/all": e, "val/wmse_A/all": wa, "val/wmse_E/all": we}
        flags0.append(bt.update("A_lr3e-4_p0", m, s, weighted=False))
        flags1.append(bt.update("A_lr3e-4_p1", m, s, weighted=True))
    bt2 = BestTracker()
    bt2.load_state_dict(bt.state_dict())
    check("BestTracker: each pair under its own weighting (p = 0: mse, p = 1: wmse), per head; state round-trips",
          flags0 == [{"A": True, "E": True}, {"A": True, "E": False}, {"A": False, "E": True}]
          and flags1 == [{"A": True, "E": True}, {"A": False, "E": True}, {"A": True, "E": False}]
          and bt2.best == bt.best and bt.best["A_lr3e-4_p1"] == {"A": (4.0, 2), "E": (1.0, 1)})

    g, o, T = batch()
    wc = warmup_constants(bb, [g], "cpu", ev_c.c)
    w = eq_weight(o.e_q, ev_c.c)
    ok = (abs(wc["A"] - float(T.t_A.mean())) < 1e-9 and abs(wc["E"] - float(T.t_E.mean())) < 1e-9
          and abs(wc["wA"] - float((w * T.t_A).sum() / w.sum())) < 1e-9)
    check("warmup_constants: the batch's plain and weighted mean targets", ok,
          " ".join(f"{k} {v:.4g}" for k, v in wc.items()))

    n = BankEvaluator(root, bb).check_alignment()
    check("smoke bank: task records match the query table (ids, query counts)", n == 14, f"{n} tasks")
    if ARGS.skip_real_bank:
        print("info real bank alignment skipped (--skip-real-bank)", flush=True)
        return
    n = BankEvaluator(BANK_DIR, bb).check_alignment()
    check("real bank: task records match the query table (ids, query counts)", n == 2000, f"{n} tasks")
    n = BankEvaluator(BANK_DIR, bb, mini=True).check_alignment()
    check("real mini-bank: task records match its query table", n == 64, f"{n} tasks")


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_BACKBONE)
    ap.add_argument("--smoke-bank", default=str(SMOKE_BANK_DIR))
    ap.add_argument("--skip-real-bank", action="store_true", help="skip the alignment walk over the real bank")
    ARGS = ap.parse_args()
    torch.set_num_threads(4)
    print(f"checkpoint {ARGS.checkpoint}; smoke bank {ARGS.smoke_bank}; torch {torch.__version__}", flush=True)
    for fn in (test_init, test_independence, test_loss, test_gradients, test_metrics, test_validation):
        guarded(fn)
    print(f"\n{'ALL OK' if not FAILS else f'{len(FAILS)} FAILED: ' + '; '.join(FAILS)}", flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
