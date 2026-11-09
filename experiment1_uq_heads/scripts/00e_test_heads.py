"""Gate for step 7 of the order of work (phase1_protocol.md §3, unit tests of §4.5): heads, loss, metrics, validation.

WHAT IS TESTED
    iltm2.model._heads      Trunk, LayerMix (A2), Head, HeadPair, Fork (B), softplus_inv, the seeded initialization
    iltm2.train._loss       batch_targets, loss_weight, uq_loss, group_weight
    iltm2.train._metrics    head_metrics, train_metrics, EMA (the step-7 additions; reference_lines is step 6's)
    iltm2.train._validate   BankEvaluator (cached and recomputed inputs, guards), BestTracker

CHECKS
    1. Initialization (protocol §4.5): the last layer's weights are zero and its bias is softplus^-1(mean target),
       so at step 0 every query of every arm gets the same output, the mean target; softplus^-1 is exact over
       1e-6..1e2 and floors non-positive targets; the initialization is a function of (seed, arm group, head) only
       (same for every LR, shared by A1 and B, different between the A and E trunks and between seeds).
    2. Head independence (protocol §4.5): an A job's 6 pairs share no parameter (nor do the A and E heads of a
       pair, A2's layer mixes included); each optimizer holds exactly its own pair's parameters; stepping one pair
       (A1 and A2 tried) leaves the other five bit-identical; the A loss gives no gradient to the E head and back.
    3. Loss (protocol §4.5): with p = 0 the weight is exactly 1 and the loss is exactly the plain MSE; with p = 2 it
       is (e(q) + c)^-2, detached; lambda_E scales the E term; group weights recombine group means into the mean
       over all query rows; no gradient reaches the targets or e(q).
    4. Gradients (protocol §4.5), on a real backbone forward of small observed-cut tasks, for A1, A2 and B: at step 0
       (zero last layer) only the last layer gets a gradient; after one step every head parameter does, and in B
       the fork; no backbone parameter ever gets one, its state-dict hash and its quantiles are unchanged; the fork's
       output equals h_star at step 0, it holds only copies (not the backbone's tensors), and its distance from
       initialization is 0 before and > 0 after a step.
    5. LayerMix: alpha = softmax(s) starts uniform and sums to 1; the output is sum_l alpha_l LN_l(r_l); a wrong
       number of taps is refused.
    6. Metrics: head_metrics agrees with the bank's reference lines (a constant head at the bank constant has the
       constant baseline's MSE / xMSE and skill 0; the e(q) proxy's predictions reproduce its skill; the target
       ratio matches); train_metrics matches direct formulas; the EMA is bias-corrected and resumable.
    7. Validation on the smoke bank's mini-bank (8 small tasks; real backbone): cached and recomputed inputs give
       the same predictions for A1, A2 and B; predictions line up with the query table (task 0 recomputed
       directly); recomputed e(q) equals the stored one; a head initialized at the bank constant reproduces the
       ref_mini constant-baseline lines; the guards fire (wrong backbone sha256, altered stored e(q), best
       tracking on the mini-bank); BestTracker flags and resumes; the task-record walk matches the query table on
       the smoke bank and on the real bank (no forward).

OUTPUT
    One "OK  <check>" or "FAIL <check>" line per check. Exit code 1 if any check failed. CPU, a few minutes.

    sbatch experiment1_uq_heads/scripts/00e_test_heads.sh [--skip-real-bank]
"""
import argparse
import math
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from iltm2.model._backbone import DEFAULT_BACKBONE, FrozenBackbone, state_dict_sha256
from iltm2.model._heads import (ARMS, HEADS, MIN_TARGET, Fork, HeadPair, LayerMix, build_a_job_heads, seeded,
                                select_input, softplus_inv)
from iltm2.prior._bank import BANK_DIR, BankQueries, iter_tasks
from iltm2.prior._dataset import UQBatch, sample_uq_task
from iltm2.prior._filter import RejectionLog
from iltm2.prior._split import QUERY_TYPES
from iltm2.train._loss import Targets, batch_targets, group_weight, loss_weight, uq_loss
from iltm2.train._metrics import (EMA, N_BINS, eq_proxy_predictions, head_metrics, reference_lines, targets_of,
                                  train_metrics)
from iltm2.train._validate import BankEvaluator, BestTracker

SMOKE_BANK = Path("/private/groups/ioannidislab/UQ_decoupling_exp/val_bank_smoke")
FAILS = []
ARGS = None


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
    """One frozen backbone with the 12 taps and the fork input (k = 2), shared by the sections."""
    if "bb" not in _CACHE:
        _CACHE["bb"] = FrozenBackbone(ARGS.checkpoint, "cpu", tap_layers="all", fork_k=2)
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


def inputs_of(arm, o, fork=None, grad=False):
    """The input a pair of `arm` reads; B goes through `fork` (with or without autograd)."""
    fork_h = None
    if arm == "B":
        with torch.set_grad_enabled(grad):
            fork_h = fork(o.fork_input, o.train_size)
    return select_input(arm, h_star=o.h_star, taps=o.taps, fork_h=fork_h)


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
    for arm in ARMS:
        pair = HeadPair(arm, 3e-4, seed=0)
        check(f"{arm}: last layer weights are zero (both heads)",
              all(bool((pair.head(h).trunk.fc2.weight == 0).all()) for h in HEADS))
        pair.init_output_bias(mA, mE)
        with torch.no_grad():
            A, E = pair(inputs_of(arm, o, fork))
        for h, out, m in (("A", A, mA), ("E", E, mE)):
            target = max(m, MIN_TARGET)
            same = bool((out == out.flatten()[0]).all())
            err = abs(float(out.flatten()[0]) - target) / target
            check(f"{arm}: step-0 {h} output identical for every query and = mean target", same and err < 1e-5,
                  f"value {float(out.flatten()[0]):.6g} vs {target:.6g} (rel {err:.1e}); shape {tuple(out.shape)}")

    p = {k: HeadPair(*k) for k in [("A1", 1e-4, 0), ("A1", 1e-3, 0), ("A1", 1e-4, 1), ("B", 1e-4, 0),
                                   ("A2", 1e-4, 0), ("A2", 1e-3, 0)]}
    P = lambda k: list(p[k].parameters())  # noqa: E731
    check("same seed and arm: identical initialization for every LR (A1 and A2)",
          params_equal(P(("A1", 1e-4, 0)), P(("A1", 1e-3, 0))) and params_equal(P(("A2", 1e-4, 0)),
                                                                                P(("A2", 1e-3, 0))))
    check("A1 and B share the initialization (B starts where A1 starts)",
          params_equal(P(("A1", 1e-4, 0)), P(("B", 1e-4, 0))))
    check("a different seed gives a different initialization", not params_equal(P(("A1", 1e-4, 0)),
                                                                                P(("A1", 1e-4, 1))))
    a = p[("A1", 1e-4, 0)]
    check("the A and E trunks are initialized differently", not torch.equal(a.A.trunk.fc1.weight,
                                                                             a.E.trunk.fc1.weight))
    check("head ids follow protocol §7.1", [x.head_id for x in build_a_job_heads(0)] ==
          ["A1_lr1e-4", "A1_lr3e-4", "A1_lr1e-3", "A2_lr1e-4", "A2_lr3e-4", "A2_lr1e-3"])


# ------------------------------------------------------------------------------------------- 2. independence


def test_independence():
    print("\n# 2. head independence in an A job")
    g, o, T = batch()
    pairs = build_a_job_heads(seed=0)
    check("an A job has 6 pairs with distinct ids (3 A1 + 3 A2)",
          len(pairs) == 6 and len({p.head_id for p in pairs}) == 6)
    ptrs = [{h: {t.data_ptr() for t in p.head(h).parameters()} for h in HEADS} for p in pairs]
    flat = [s for d in ptrs for s in d.values()]
    disjoint = all(not (flat[i] & flat[j]) for i in range(len(flat)) for j in range(i + 1, len(flat)))
    check("no parameter shared between pairs, nor between the A and E heads of a pair (A2 mixes included)",
          disjoint)
    check("each A2 head has its own LayerMix", all(p.A.mix is not None and p.E.mix is not None
                                                   and p.A.mix is not p.E.mix for p in pairs if p.arm == "A2"))
    opts = [p.make_optimizer() for p in pairs]
    check("each optimizer holds exactly its own pair's parameters",
          all({id(t) for gr in op.param_groups for t in gr["params"]} == {id(t) for t in p.parameters()}
              for p, op in zip(pairs, opts)))
    for p in pairs:
        p.init_output_bias(float(T.t_A.mean()), float(T.t_E.mean()))
    for k in (0, 3):                                       # one A1 pair, one A2 pair
        before = [[t.detach().clone() for t in p.parameters()] for p in pairs]
        for p, op in zip(pairs, opts):                     # every pair computes its loss, as in the trainer
            op.zero_grad(set_to_none=True)
            A, E = p(inputs_of(p.arm, o))
            uq_loss(A, E, T.t_A, T.t_E, o.e_q).loss.backward()
        opts[k].step()
        moved = [not params_equal(b, [t.detach() for t in p.parameters()]) for b, p in zip(before, pairs)]
        check(f"stepping pair {pairs[k].head_id} moves it and leaves the other five bit-identical",
              moved[k] and not any(m for i, m in enumerate(moved) if i != k), f"moved: {moved}")
        for op in opts:
            op.zero_grad(set_to_none=True)
    for p in (pairs[0], pairs[3]):
        for h, other in (("A", "E"), ("E", "A")):
            p.zero_grad(set_to_none=True)
            out = dict(zip(HEADS, p(inputs_of(p.arm, o))))
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
    e_req = e.clone().requires_grad_(True)
    w2 = loss_weight(e_req, 2.0, c=0.01)
    check("p = 2: w = (e(q) + c)^-2, detached from e(q)",
          torch.allclose(w2, (e + 0.01) ** -2, rtol=1e-12, atol=0) and not w2.requires_grad)
    check("p > 0 without a floor c is refused", raises(lambda: loss_weight(e, 2.0), AssertionError))
    out = uq_loss(A, E, tA, tE, e)
    check("p = 0: the loss is exactly the plain MSE of each head",
          torch.equal(out.loss_A, ((A - tA.float()) ** 2).mean()) and torch.equal(out.loss_E,
                                                                                  ((E - tE.float()) ** 2).mean()))
    out2 = uq_loss(A, E, tA, tE, e, p=2.0, c=0.01)
    w = ((e + 0.01) ** -2).float()
    check("p = 2: the loss is the weighted MSE", torch.allclose(out2.loss_A, (w * (A - tA.float()) ** 2).mean(),
                                                               rtol=1e-6))
    half = uq_loss(A, E, tA, tE, e, lambda_E=0.5)
    check("lambda_E scales only the E term", torch.allclose(half.loss, half.loss_A + 0.5 * half.loss_E, rtol=1e-7)
          and torch.equal(half.loss_A, out.loss_A))
    A_req = A.clone().requires_grad_(True)
    uq_loss(A_req, E, tA, tE, e_req).loss.backward()
    check("no gradient reaches the targets or e(q)", tA.grad is None and tE.grad is None and e_req.grad is None
          and A_req.grad is not None)
    sizes, x = [30, 50, 120], torch.rand(200, generator=gen, dtype=torch.float64)
    parts, lo = [], 0
    for m in sizes:
        parts.append(group_weight(m, n) * float(x[lo:lo + m].mean()))
        lo += m
    check("group weights recombine group means into the mean over all query rows",
          abs(sum(parts) - float(x.mean())) < 1e-12)


# ------------------------------------------------------------------------------------------- 4. gradients


def test_gradients():
    print("\n# 4. gradients reach the heads (and the fork) and nothing else")
    bb = backbone()
    g, o, T = batch()
    h0, q0 = state_dict_sha256(bb.model), o.q.clone()
    for arm in ARMS:
        pair = HeadPair(arm, 3e-4, seed=0)
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
            A, E = pair(inputs_of(arm, o, fork, grad=True))
            uq_loss(A, E, T.t_A, T.t_E, o.e_q).loss.backward()
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


# ------------------------------------------------------------------------------------------- 5. LayerMix


def test_layer_mix():
    print("\n# 5. LayerMix (A2)")
    with seeded(3):
        mix = LayerMix(12)
        taps = torch.randn(2, 5, 12, 512)
    a = mix.alphas().detach()
    check("alpha starts uniform and sums to 1", torch.allclose(a, torch.full((12,), 1 / 12)) and
          abs(float(a.sum()) - 1) < 1e-6)
    with torch.no_grad():
        out = mix(taps)
        manual = sum(a[l] * mix.norms[l](taps[:, :, l]) for l in range(12))
    check("output = sum_l alpha_l LN_l(r_l), shape (B, m, D)", tuple(out.shape) == (2, 5, 512)
          and maxdiff(out, manual) < 1e-6, f"max |diff| {maxdiff(out, manual):.1e}")
    check("a wrong number of taps is refused", raises(lambda: mix(taps[:, :, :11]), AssertionError))


# ------------------------------------------------------------------------------------------- 6. metrics


def synthetic_queries(n_tasks=6, m=400, seed=0):
    """A query table with real signal: a true A and E per query (partly a function of e(q), so the e(q) proxy has
    skill), and two independent noisy halves around it. Without signal the constant baseline's xMSE is ~0 and can
    be negative in a slice, where skill is undefined (NaN)."""
    gen = torch.Generator().manual_seed(seed)
    n = n_tasks * m
    r = lambda: torch.rand(n, generator=gen)  # noqa: E731
    e_q = r() + 0.01
    s_A, s_E = 0.5 * e_q + 0.1 * r(), 0.2 * e_q + 0.3 * r()
    tAa, tAb = s_A + 0.2 * (r() - 0.5), s_A + 0.2 * (r() - 0.5)
    tEa, tEb = s_E + 0.3 * (r() - 0.5), s_E + 0.3 * (r() - 0.5)
    return BankQueries(task_id=torch.arange(n_tasks).repeat_interleave(m),
                       bin_idx=torch.zeros(n, dtype=torch.int8),
                       query_type=torch.randint(0, len(QUERY_TYPES), (n,), generator=gen).to(torch.int8),
                       n_bin=torch.randint(0, len(N_BINS), (n,), generator=gen).to(torch.int8),
                       e_q=e_q, t_A=(tAa + tAb) / 2, t_E=(tEa + tEb) / 2, t_A_a=tAa, t_A_b=tAb,
                       t_E_a=tEa, t_E_b=tEb, score=r())


def test_metrics():
    print("\n# 6. metrics")
    Q = synthetic_queries()
    refs = reference_lines(Q)
    const = {h: torch.full((len(Q),), refs[f"ref/const_value_{h}"]) for h in HEADS}
    mc = head_metrics(const, Q, refs)
    slices = sorted({k.rsplit("/", 1)[-1] for k in mc})
    w = worst(x for h in HEADS for s in slices for x in (mc[f"val/xmse_{h}/{s}"] - refs[f"ref/const_xmse_{h}/{s}"],
                                                         mc[f"val/mse_{h}/{s}"] - refs[f"ref/const_mse_{h}/{s}"],
                                                         mc[f"val/skill_{h}/{s}"]))
    check("a constant head at the bank constant = the constant-baseline lines (MSE, xMSE), skill 0, every slice",
          w < 1e-6, f"{len(slices)} slices; worst |diff| {w:.1e}")
    proxy = {h: eq_proxy_predictions(Q.e_q, targets_of(Q, h)[0], Q.task_id).float() for h in HEADS}
    mp = head_metrics(proxy, Q, refs)
    w = worst(mp[f"val/skill_{h}/{s}"] - refs[f"ref/eq_proxy_skill_{h}/{s}"] for h in HEADS for s in slices)
    sk = " ".join(f"{h} {refs[f'ref/eq_proxy_skill_{h}/all']:.3f}" for h in HEADS)
    check("the e(q) proxy's predictions reproduce its reference skill, every slice", w < 1e-5,
          f"worst |diff| {w:.1e}; pooled proxy skill {sk}")
    w = worst(mc[f"val/ratio_tA_eq/{s}"] - refs[f"ref/ratio_tA_eq/{s}"] for s in slices)
    check("val/ratio_tA_eq matches the reference ratio", w < 1e-9)
    check("every head-metric value is finite", all(math.isfinite(v) for m in (mc, mp) for v in m.values()))
    only = head_metrics(const, Q, refs, slices=("all",))
    check("slices=('all',) reports only the pooled slice", all(k.endswith("/all") for k in only) and len(only) > 0)

    gen = torch.Generator().manual_seed(1)
    A, E = torch.rand(50, generator=gen), torch.rand(50, generator=gen)
    T = Targets(*(torch.rand(50, generator=gen, dtype=torch.float64) for _ in range(6)))
    tm = train_metrics(A, E, T, torch.rand(50, generator=gen))
    ok = (abs(tm["train/mse_A"] - float(((A.double() - T.t_A) ** 2).mean())) < 1e-12 and
          abs(tm["train/xmse_E"] - float(((E.double() - T.t_E_a) * (E.double() - T.t_E_b)).mean())) < 1e-12)
    check("train_metrics: mse and xmse match their formulas", ok)

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


# ------------------------------------------------------------------------------------------- 7. validation


def test_validation():
    print("\n# 7. validation on the smoke bank's mini-bank")
    root = Path(ARGS.smoke_bank)
    check("the smoke bank exists", (root / "manifest.json").exists(), str(root))
    if not (root / "manifest.json").exists():
        return
    bb = backbone()
    ev_c = BankEvaluator(root, bb, mini=True, cache=("h", "taps", "fork"))
    ev_r = BankEvaluator(root, bb, mini=True)
    check("recomputed e(q) matches the stored e(q) on every mini-bank task (within the guard's tolerance)",
          ev_c.max_eq_scaled <= 1e-3,
          f"{len(ev_c.task_ids)} tasks, {len(ev_c.Q)} queries; max |diff| {ev_c.max_eq_diff:.1e} = "
          f"{ev_c.max_eq_scaled:.1e} of the task's mean e(q) (bank built on GPU, recomputed here on CPU: "
          f"round-off, not a different forward)")

    pairs = [HeadPair(arm, 3e-4, seed=0) for arm in ARMS]
    with seeded(11), torch.no_grad():                       # nonzero last layers: predictions vary per query
        for p in pairs:
            for h in HEADS:
                nn.init.normal_(p.head(h).trunk.fc2.weight, std=0.05)
    fork = Fork(bb, k=2)
    pc, pr = ev_c.predict(pairs, fork), ev_r.predict(pairs, fork)
    for p in pairs:
        d = max(maxdiff(pc[p.head_id][h], pr[p.head_id][h]) for h in HEADS)
        spread = float(pc[p.head_id]["A"].std())
        check(f"{p.arm}: cached and recomputed inputs give the same predictions", d <= 1e-6,
              f"max |diff| {d:.1e}; {len(pc[p.head_id]['A'])} predictions (std {spread:.3g})")
    rec0 = next(iter_tasks(root, mini=True))
    with torch.no_grad():
        o0 = bb(rec0["X"][None], rec0["y_ctx"][None])
        direct = {p.head_id: p(inputs_of(p.arm, o0, fork)) for p in pairs}
    m0 = rec0["m"]
    d = max(maxdiff(direct[p.head_id][i][0], pr[p.head_id][h][:m0]) for p in pairs for i, h in enumerate(HEADS))
    check("predictions line up with the query table (task 0 recomputed directly)", d <= 1e-6,
          f"m = {m0}; max |diff| {d:.1e}")

    refs = ev_c.refs
    cpair = HeadPair("A1", 3e-4, seed=0)
    cpair.init_output_bias(refs["ref_mini/const_value_A"], refs["ref_mini/const_value_E"])
    out = ev_c.evaluate([cpair])
    ok = all(abs(out[f"A1_lr3e-4/mini/{k}_{h}/all"] / refs[f"ref_mini/const_{k}_{h}/all"] - 1) < 1e-4
             for h in HEADS for k in ("mse", "xmse"))
    check("a head at the bank constant reproduces the ref_mini constant-baseline lines", ok,
          " ".join(f"{h}: mse {out[f'A1_lr3e-4/mini/mse_{h}/all']:.5g} vs {refs[f'ref_mini/const_mse_{h}/all']:.5g}"
                   for h in HEADS))
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
    seq = [(0, 0.5, 0.9), (1, 0.4, 1.0), (2, 0.45, 0.8)]
    flags = [bt.update("A1_lr3e-4", {"val/mse_A/all": a, "val/mse_E/all": e}, s) for s, a, e in seq]
    bt2 = BestTracker()
    bt2.load_state_dict(bt.state_dict())
    check("BestTracker: flags only on improvement, per head; state round-trips",
          flags == [{"A": True, "E": True}, {"A": True, "E": False}, {"A": False, "E": True}]
          and bt2.best == bt.best and bt.best["A1_lr3e-4"] == {"A": (0.4, 1), "E": (0.8, 2)})

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
    ap.add_argument("--smoke-bank", default=str(SMOKE_BANK))
    ap.add_argument("--skip-real-bank", action="store_true", help="skip the alignment walk over the real bank")
    ARGS = ap.parse_args()
    torch.set_num_threads(4)
    print(f"checkpoint {ARGS.checkpoint}; smoke bank {ARGS.smoke_bank}; torch {torch.__version__}", flush=True)
    for fn in (test_init, test_independence, test_loss, test_gradients, test_layer_mix, test_metrics,
               test_validation):
        guarded(fn)
    print(f"\n{'ALL OK' if not FAILS else f'{len(FAILS)} FAILED: ' + '; '.join(FAILS)}", flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
