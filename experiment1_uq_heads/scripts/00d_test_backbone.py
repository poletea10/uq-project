"""Gate for step 5 of the order of work (phase1_protocol.md §3, unit tests of §4.4): the frozen backbone wrapper.

WHAT IS TESTED
    iltm2.model._backbone   `FrozenBackbone`: TabICLv2 frozen, one raw training-mode forward (the forward route of
                            design Architecture §1), with forward hooks for h_star, the 12 residual taps (A2) and
                            the fork input (B); `copy_last_blocks` / `run_blocks` for arm B's trainable copy.
    Every head target and every head input comes through it, and the PPD must stay bit-identical to TabICLv2
    (Phase 0 compares against the same forecast), so the hooks may read but never change anything.

CHECKS (protocol §4.4, plus what the hooks need)
    1. frozen: no backbone parameter requires grad, the wrapper is not an nn.Module (it cannot enter an
       optimizer), and the backbone's state-dict hash and outputs are unchanged after an optimizer step on a head;
    2. the PPD: the wrapper's quantiles are torch.equal to tabicl's own training-mode forward of a separately
       loaded, hook-free model (under no_grad; with grad enabled, as tabicl's trainer runs it, they must agree to
       1e-5); enabling taps / the fork input does not change them; two calls give identical outputs; calling it
       in eval mode is refused;
    3. what is read: h_star is the final LN of the last block's output at the query rows, decoder(h_star) is q,
       A2 gets 12 taps of shape (B, m, 12, 512), e_q is finite and >= 0;
    4. arm B, for k = 1 and 2: the copy's output at step 0 is allclose to h_star (also with gradient
       checkpointing); its parameters are new tensors; a training step changes the copy and leaves the backbone,
       its quantiles (the decoder input) and its hash untouched;
    5. a zero-padded micro-batch group: two tasks whose kept d differ (FAIL, never skip, if none is found), the
       narrow one's extra columns exactly zero, each task's forecast in the group equal to its forecast alone
       (batching does not mix tasks); the effect of padding on the forecast is reported.
    Small real observed-cut tasks (N = 256, n_ctx = 200, d = 6), so the gate runs on CPU in a few minutes.

OUTPUT
    One "OK  <check>" or "FAIL <check>" line per check. Exit code 1 if any check failed.

    sbatch experiment1_uq_heads/scripts/00d_test_backbone.sh [--device cuda] [--checkpoint PATH]
"""
import argparse
import sys
import traceback

import torch
from torch import nn

from iltm2.model._backbone import (DEFAULT_BACKBONE, FrozenBackbone, attention_ctx, load_tabicl,
                                   state_dict_sha256)
from iltm2.prior._dataset import UQBatch, sample_uq_task
from iltm2.prior._filter import RejectionLog

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


def close(name, a, b, atol):
    """Pass if bit-equal or within atol; the detail says which."""
    eq = torch.equal(a, b)
    d = 0.0 if eq else maxdiff(a, b)
    check(name, eq or d <= atol, "bit-equal" if eq else f"max |diff| {d:.2e} (atol {atol:g})")


_GROUP = None


def group():
    """A micro-batch group of 2 small observed-cut tasks with the same (N, n_ctx) (cached)."""
    global _GROUP
    if _GROUP is None:
        tasks = [sample_uq_task(5, 0, t, stage=1, R=4, N=256, n_ctx=200, d=6)[0] for t in range(2)]
        _GROUP = UQBatch.stack(tasks, stage=1, step=0, group=0, task_idx=[0, 1], log=RejectionLog(), gen_time=0.0)
    return _GROUP


def one():
    """The first task of the group alone, its padding removed: (X (1, N, d), y_ctx (1, n_ctx))."""
    g = group()
    return g.X[:1, :, :int(g.d[0])], g.y_ctx[:1]


# ------------------------------------------------------------------------------------------- 1. frozen


def test_frozen():
    print("\n# 1. the backbone is frozen")
    bb = FrozenBackbone(ARGS.checkpoint, ARGS.device)
    check("no backbone parameter requires grad", not any(p.requires_grad for p in bb.model.parameters()))
    check("the wrapper is not an nn.Module (it owns no trainable parameter)", not isinstance(bb, nn.Module))
    X, y = one()
    before = bb(X, y)
    h0 = state_dict_sha256(bb.model)
    head = nn.Sequential(nn.Linear(bb.d_model, 64), nn.GELU(), nn.Linear(64, 1)).to(bb.device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-5)
    for _ in range(3):
        loss = nn.functional.softplus(head(bb(X, y).h_star)).square().mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    check("state-dict hash unchanged after optimizer steps on a head", state_dict_sha256(bb.model) == h0)
    check("no backbone parameter received a gradient", all(p.grad is None for p in bb.model.parameters()))
    after = bb(X, y)
    check("quantiles unchanged after the head's optimizer steps", torch.equal(before.q, after.q))


# ------------------------------------------------------------------------------------------- 2. the PPD


def test_ppd():
    print("\n# 2. the PPD is tabicl's own training-mode forward")
    X, y = one()
    bb = FrozenBackbone(ARGS.checkpoint, ARGS.device)
    out = bb(X, y)
    ref = load_tabicl(ARGS.checkpoint, ARGS.device)        # separately loaded, no hooks, parameters trainable
    dev = torch.device(ARGS.device)
    Xd, yd = X.to(dev), y.to(dev)
    with torch.no_grad(), attention_ctx(dev.type):
        q_ref = ref(Xd, yd)
    check("quantiles torch.equal to tabicl's training-mode forward (no_grad)", torch.equal(out.q, q_ref),
          "" if torch.equal(out.q, q_ref) else f"max |diff| {maxdiff(out.q, q_ref):.2e}")
    with attention_ctx(dev.type):
        q_grad = ref(Xd, yd)                                # grad enabled, as tabicl's trainer runs it
    close("quantiles match tabicl's forward with grad enabled", out.q, q_grad.detach(), atol=1e-5)
    plain = FrozenBackbone(ARGS.checkpoint, ARGS.device, tap_layers=None)
    forked = FrozenBackbone(ARGS.checkpoint, ARGS.device, tap_layers=[3, 7], fork_k=2)
    check("hooks do not change the quantiles (no taps vs 12 taps vs 2 taps + fork input)",
          torch.equal(plain(X, y).q, out.q) and torch.equal(forked(X, y).q, out.q))
    again = bb(X, y)
    check("two calls give identical outputs", torch.equal(out.q, again.q) and torch.equal(out.h_star, again.h_star)
          and torch.equal(out.taps, again.taps))
    bb.model.eval()
    try:
        bb(X, y)
        refused = False
    except AssertionError:
        refused = True
    bb.model.train()
    check("the wrapper refuses an eval-mode model (a different code path)", refused)


# ------------------------------------------------------------------------------------------- 3. what is read


def test_reads():
    print("\n# 3. h_star, the taps and e(q)")
    X, y = one()
    bb = FrozenBackbone(ARGS.checkpoint, ARGS.device)
    o = bb(X, y)
    m = X.shape[1] - y.shape[1]
    check("q has shape (B, m, 999)", tuple(o.q.shape) == (1, m, 999), str(tuple(o.q.shape)))
    check("A2 gets 12 taps of shape (B, m, 12, 512)", o.tap_layers == tuple(range(1, 13))
          and tuple(o.taps.shape) == (1, m, 12, 512), f"{tuple(o.taps.shape)}, layers {o.tap_layers}")
    check("h_star has shape (B, m, 512)", tuple(o.h_star.shape) == (1, m, 512))
    with torch.no_grad():
        close("h_star = final LN of the last block's output at the query rows", bb.ln(o.taps[:, :, -1]), o.h_star,
              atol=1e-6)
        close("decoder(h_star) = q", bb.decoder(o.h_star), o.q, atol=1e-5)
    check("e(q) finite and >= 0, shape (B, m)", tuple(o.e_q.shape) == (1, m) and bool(torch.isfinite(o.e_q).all())
          and bool((o.e_q >= 0).all()), f"range [{float(o.e_q.min()):.3g}, {float(o.e_q.max()):.3g}]")
    check("taps differ across layers (each hook reads its own block)",
          all(not torch.equal(o.taps[:, :, i], o.taps[:, :, i + 1]) for i in range(11)))


# ------------------------------------------------------------------------------------------- 4. arm B


def test_fork():
    print("\n# 4. arm B: a trainable copy of the last k blocks")
    X, y = one()
    for k in (1, 2):
        bb = FrozenBackbone(ARGS.checkpoint, ARGS.device, tap_layers=None, fork_k=k)
        o = bb(X, y)
        check(f"k={k}: fork input is the full stream (B, N, 512) after block {12 - k}",
              o.fork_input is not None and tuple(o.fork_input.shape) == (1, X.shape[1], 512))
        blocks, ln = bb.copy_last_blocks(k)
        with torch.no_grad():
            h_fork = bb.run_blocks(blocks, ln, o.fork_input, o.train_size)
            h_fork_ckpt = bb.run_blocks(blocks, ln, o.fork_input, o.train_size, recompute=True)
        close(f"k={k}: fork output at step 0 = h_star", h_fork, o.h_star, atol=1e-5)
        close(f"k={k}: same with gradient checkpointing", h_fork_ckpt, h_fork, atol=0.0)
        orig = {p.data_ptr() for p in bb.model.parameters()}
        new = list(blocks.parameters()) + list(ln.parameters())
        check(f"k={k}: the copy's parameters are new tensors and trainable",
              all(p.data_ptr() not in orig and p.requires_grad for p in new))
        h_before, q_before = state_dict_sha256(bb.model), o.q.clone()
        w_before = [p.detach().clone() for p in new]
        opt = torch.optim.AdamW(new, lr=1e-3)
        for _ in range(2):
            loss = bb.run_blocks(blocks, ln, bb(X, y).fork_input, o.train_size).square().mean()
            loss.backward()
            opt.step()
            opt.zero_grad()
        check(f"k={k}: a training step changes the copy", any(not torch.equal(a, p.detach())
                                                             for a, p in zip(w_before, new)))
        check(f"k={k}: the backbone's hash is unchanged and no backbone parameter has a gradient",
              state_dict_sha256(bb.model) == h_before and all(p.grad is None for p in bb.model.parameters()))
        check(f"k={k}: the decoder input is untouched (backbone quantiles bit-identical)",
              torch.equal(bb(X, y).q, q_before))


# ------------------------------------------------------------------------------------------- 5. padded group


def padded_group(max_tries=10):
    """A group of 2 tasks whose kept feature counts differ, so `UQBatch.stack` must zero-pad one of them.

    Task 0 is drawn with d = 8, task 1 with d = 3. Constant columns are dropped after the draw, so the kept counts
    can still coincide; then task 0 is redrawn (next task_idx). Returns (group, [task0, task1]) or (None, None)
    if no pair with different kept d is found, which the caller turns into a FAIL, never a skip.
    """
    narrow = sample_uq_task(6, 0, 0, stage=1, R=4, N=256, n_ctx=200, d=3)[0]
    for t in range(1, 1 + max_tries):
        wide = sample_uq_task(6, 0, t, stage=1, R=4, N=256, n_ctx=200, d=8)[0]
        if wide.task.X.shape[1] != narrow.task.X.shape[1]:
            tasks = [wide, narrow]
            return UQBatch.stack(tasks, stage=1, step=0, group=0, task_idx=[t, 0], log=RejectionLog(),
                                 gen_time=0.0), tasks
    return None, None


def test_group():
    print("\n# 5. a zero-padded micro-batch group (tasks with different d)")
    g, tasks = padded_group()
    check("found 2 tasks with different kept d (padding is exercised)", g is not None,
          "" if g is None else f"d per task {g.d.tolist()}")
    if g is None:
        return
    d_wide, d_narrow = int(g.d[0]), int(g.d[1])
    check("X has the group's max d as width", tuple(g.X.shape) == (2, 256, d_wide), str(tuple(g.X.shape)))
    check("the narrow task's padded columns are exactly zero", bool((g.X[1, :, d_narrow:] == 0).all()))
    check("each task's real columns are its own rows, unchanged",
          torch.equal(g.X[0], tasks[0].X_model.float()) and torch.equal(g.X[1, :, :d_narrow],
                                                                       tasks[1].X_model.float()))
    bb = FrozenBackbone(ARGS.checkpoint, ARGS.device)
    o = bb(g.X, g.y_ctx)
    check("group of 2 runs; outputs finite", tuple(o.q.shape) == (2, g.m, 999) and bool(torch.isfinite(o.q).all())
          and bool(torch.isfinite(o.h_star).all()))
    # Batching must not mix tasks: each task's forecast in the group equals its forecast run alone with the same
    # input (the narrow one padded to the same width). Different batch shapes may change float32 round-off.
    alone_wide = bb(g.X[:1], g.y_ctx[:1]).q
    alone_narrow = bb(g.X[1:], g.y_ctx[1:]).q
    close("wide task: forecast in the group = forecast alone", o.q[:1], alone_wide, atol=1e-5)
    close("narrow task: forecast in the group = forecast alone, same zero padding", o.q[1:], alone_narrow, atol=1e-5)
    # Reported, not gated: v2 runs with d = None, so the backbone treats zero-padded columns as features (as in
    # TabICLv2's pretraining, where X is padded to max(d) of the micro-batch). Padding therefore changes the
    # forecast; the stream reproduces pretraining's padding rule rather than avoiding it.
    unpadded = bb(g.X[1:, :, :d_narrow], g.y_ctx[1:]).q
    print(f"info narrow task: max |q(padded) - q(unpadded)| = {maxdiff(alone_narrow, unpadded):.3g} "
          f"(nonzero = the backbone sees the padding, as in pretraining)", flush=True)


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--checkpoint", default=DEFAULT_BACKBONE)
    ARGS = ap.parse_args()
    torch.set_num_threads(4)
    print(f"checkpoint {ARGS.checkpoint}; device {ARGS.device}; torch {torch.__version__}", flush=True)
    for fn in (test_frozen, test_ppd, test_reads, test_fork, test_group):
        guarded(fn)
    print(f"\n{'ALL OK' if not FAILS else f'{len(FAILS)} FAILED: ' + '; '.join(FAILS)}", flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
