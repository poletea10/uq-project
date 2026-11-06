"""Step-4 gate (phase1_protocol.md §3): throughput of the training stream, per stage.

WHY
    The training stream is generated on the fly (protocol §2): every step consumes 32 fresh observed-cut tasks,
    each with its split and R = 16 replicate passes. If the DataLoader workers cannot build a step's tasks faster
    than the GPU consumes them, the GPU waits on the prior and the step budget of design Training §2 becomes
    wall-time bound. Gate: prior time per step at most 0.8 x the step's train time, in every stage (the workers
    build tasks while the GPU trains, so a step costs the slower of the two; with >= 2 steps of items in flight the
    remaining 20% absorbs step-to-step variation). On a failure, raise the number of workers first, then lower
    the region bounds in stage 3 (design §1.1, "Cost of the enlarged pool"); the per-component profile below says
    which part of a task dominates.

    The loader delivers out of order, builds each step's largest groups first and restores the stream order in
    `iterate_steps` (`iltm2.prior._dataset.make_dataloader`): with an in-order loader, slow large-N tasks left
    most workers idle in stages 2-3.

WHAT (per stage, default 1 2 3)
    1. Determinism through the DataLoader: the first --det-groups groups built in this process (workers = 0)
       are bit-identical to the same groups delivered out of order by a W-worker DataLoader and put back in order
       by `iterate_steps` (protocol §4.2: same (seed, step, task_idx) gives the same task whatever the number of
       workers or the delivery order).
    2. Throughput, for each W in --workers: `CutGraphPrior` + `make_dataloader` + `iterate_steps`, exactly the
       trainer's code path, over --n-steps steps. Reported:
         steady s/step   (completion of the last step - completion of the first) / (steps - 1): the rate once all
                         workers are running (excludes start-up and the first step's latency). The gated number.
         wall s/step     total wall time / steps, start-up included (pessimistic for few steps)
         ideal s/step    batch * mean CPU s per task / W: perfect parallelism, the floor for this W
         steady/ideal    the scheduling loss (1 = every worker always busy)
         tasks/s         batch / steady s/step
         out of order    share of items that arrived out of stream order (evidence the reordering is exercised)
       plus the rejection rate, redraws after non-finite replicates, deterministic share, query-type shares and
       the t_A zero-mass fraction of the batches (sanity: they must match design Training §1).
    3. Profile: --profile-tasks tasks built piece by piece in this process (one thread, as in a worker): X pool
       and filters (pipeline steps 1-7), split (8-9), replicates (10), with N and P.
    4. Train-time floor (GPU): the frozen TabICLv2 forward (one raw training-mode forward in float32, the forward
       route of design Architecture §1 and TabICLv2's pretraining dtype) on the first --gpu-steps steps of the
       first W run, micro-batch = group, as the trainer will run it. The heads (~1M parameters each) cost nothing
       next to it, so this is a lower bound on an A-job step; arm B adds a backward through k blocks. Peak GPU
       memory is reported.
    Verdict per stage and W: steady prior s/step <= --margin (0.8) * backbone s/step. Since the backbone forward
    is a lower bound on the train time, passing here is conservative. The projected wall time of each stage
    (design steps x max(prior, backbone)) is printed.

OUTPUT
    outputs/audits/throughput/summary.json and throughput.md. One "OK  ..." / "FAIL ..." line per check; exit 1 if
    any check failed (no GPU = the gate is not assessable = FAIL).

    sbatch experiment1_uq_heads/scripts/02_benchmark_prior.sh
    sbatch experiment1_uq_heads/scripts/02_benchmark_prior.sh --stages 3 --n-steps 6
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from tabicl._model.tabicl import TabICL

from iltm2.prior._cut_graph_scm import CutGraphSCM, NonFiniteReplicate
from iltm2.prior._dataset import (BATCH_SIZE, GROUP_SIZE, STAGES, STREAM_REPLICATES, STREAM_SPLIT, STREAM_TASK, CutGraphPrior,
                                  iterate_steps, make_dataloader, sample_group_shape, sample_task_shape,
                                  single_thread, stage_step_key, task_rng)
from iltm2.prior._filter import RejectionLog
from iltm2.prior._split import QUERY_TYPES, make_split
from iltm2.train._targets import t_aleatoric

OUT = Path(__file__).resolve().parents[1] / "outputs" / "audits" / "throughput"
DEFAULT_CKPT = "/private/groups/ioannidislab/ajaras/tabicl-dev/tabicl-regressor-v2-20260212.ckpt"
DESIGN_STEPS = {1: 50_000, 2: 4_000, 3: 1_000}            # design Training §2
FAILS = []


def check(name, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)
    if not ok:
        FAILS.append(name)
    return ok


def q(a, ps=(0.1, 0.5, 0.9)):
    a = np.asarray(a, float)
    return " ".join(f"q{int(p * 100)}={np.quantile(a, p):.3g}" for p in ps) if len(a) else "-"


def same_bits(a, b):
    """Bitwise equality of two UQBatch groups (NaN-aware, like `_cut_graph_scm._same_bits`)."""
    def eq(x, y):
        return (x.shape == y.shape and x.dtype == y.dtype
                and torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)))
    return all(eq(getattr(a, k), getattr(b, k)) for k in ("X", "y_ctx", "y_reps", "query_type", "d", "task_idx"))


# ------------------------------------------------------------------------------------------- 2. throughput


def run_loader(prior, workers, stash_steps, prefetch_factor):
    """Consume the whole prior exactly as the trainer will: `make_dataloader` (out-of-order delivery, largest groups
    first) + `iterate_steps` (the per-step reorder buffer). Returns the completion time of every step, the arrival
    order of the items, the items' metadata, and the groups of the first `stash_steps` steps (for the GPU timing
    and the determinism check)."""
    t_start = time.perf_counter()
    loader = make_dataloader(prior, num_workers=workers, prefetch_factor=prefetch_factor)
    arrivals, step_done, meta, stash, log = [], [], [], [], RejectionLog()

    def on_item(item):
        arrivals.append((item.step, item.group))

    for step, groups in iterate_steps(loader, on_item=on_item):
        step_done.append(time.perf_counter())
        for item in groups:
            tA = t_aleatoric(item.y_reps.double())
            meta.append(dict(step=item.step, group=item.group, N=item.seq_len, B=item.B, gen_time=item.gen_time,
                             d=item.d.tolist(), det=item.deterministic.tolist(),
                             qtype=np.bincount(item.query_type.flatten().numpy(), minlength=len(QUERY_TYPES)),
                             tA_zero=int((tA == 0).sum()), n_q=int(tA.numel())))
            log += item.log
        if step - prior.start_step < stash_steps:
            stash.extend(groups)
    del loader                                               # shuts the persistent workers down
    return dict(t_start=t_start, step_done=np.array(step_done), arrivals=arrivals), meta, stash, log


def summarize_run(prior, workers, timing, meta, log):
    n_items, n_steps = len(meta), prior.max_steps - prior.start_step
    done = timing["step_done"]
    # steady rate: time between the first and the last completed step (excludes start-up and the first step's
    # latency); with a single step, fall back to the wall time
    steady = (done[-1] - done[0]) / (n_steps - 1) if n_steps > 1 else done[0] - timing["t_start"]
    stream_order = [(m["step"], m["group"]) for m in sorted(meta, key=lambda m: (m["step"], m["group"]))]
    out_of_order = float(np.mean([a != b for a, b in zip(timing["arrivals"], stream_order)]))
    n_tasks = sum(m["B"] for m in meta)
    cpu_per_task = sum(m["gen_time"] for m in meta) / n_tasks
    qt = sum(m["qtype"] for m in meta)
    return dict(workers=workers, steps=n_steps, items=n_items, tasks=n_tasks,
                steady_s_per_step=steady,
                wall_s_per_step=(done[-1] - timing["t_start"]) / n_steps,
                first_step_s=done[0] - timing["t_start"],
                items_arrived_out_of_stream_order=out_of_order,
                ideal_s_per_step=cpu_per_task * prior.batch_size / workers,
                tasks_per_s=prior.batch_size / steady,
                cpu_s_per_task=cpu_per_task,
                cpu_s_per_task_q=q([m["gen_time"] / m["B"] for m in meta]),
                rejection_rate=log.rate, rejections=dict(log.rejected), graph_resamples=log.graph_resamples,
                redraws=log.rejected["non_finite_replicate"],
                deterministic_share=float(np.mean([x for m in meta for x in m["det"]])),
                query_type_share={QUERY_TYPES[k]: float(qt[k] / qt.sum()) for k in QUERY_TYPES},
                tA_zero_mass=sum(m["tA_zero"] for m in meta) / sum(m["n_q"] for m in meta),
                N_q=q([m["N"] for m in meta]), d_q=q([x for m in meta for x in m["d"]]))


# ------------------------------------------------------------------------------------------- 3. profile


def profile_task(seed, stage, task_idx, cfg, R):
    """Time the pieces of one task (attempt k = 0 of `sample_uq_task`) on step 1 of the stage, in this process."""
    key = stage_step_key(stage, 1)
    with single_thread():
        group = task_idx // GROUP_SIZE[stage]
        N, n_ctx, d = sample_group_shape(seed, key, group, stage, cfg)
        t0 = time.perf_counter()
        with task_rng(seed, key, task_idx, STREAM_TASK):
            shape = sample_task_shape(stage, cfg, N=N, n_ctx=n_ctx, d=d)
            log = RejectionLog()
            task, _ = CutGraphSCM(cfg).sample(shape, log)
        t1 = time.perf_counter()
        with task_rng(seed, key, task_idx, STREAM_SPLIT):
            split = make_split(task.X, shape, cfg.dup_fraction)
        t2 = time.perf_counter()
        try:
            with task_rng(seed, key, task_idx, STREAM_REPLICATES):
                task.replicates(split.query_rows, R)
        except NonFiniteReplicate:
            pass                                             # timing only; the stream would redraw the task
        t3 = time.perf_counter()
    return dict(N=shape.N, P=shape.P, d=shape.d, m=shape.m, attempts=log.attempts,
                pool_and_filter_s=t1 - t0, split_s=t2 - t1, replicates_s=t3 - t2, total_s=t3 - t0)


# ------------------------------------------------------------------------------------------- 4. GPU


def load_backbone(path, device):
    """TabICLv2 as scripts/00f_difficulty_match.py loads it: `TabICL(**ckpt["config"])`, training mode (dropout 0,
    deterministic), frozen. Step 5's `_backbone.py` replaces this."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = TabICL(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).train()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def backbone_time(model, stash, device):
    """Seconds per step of the frozen forward over the stashed groups (micro-batch = group), and peak memory."""
    torch.cuda.reset_peak_memory_stats(device)
    first = stash[0].to(device)
    model(first.X, first.y_ctx)                              # warm-up (kernel selection), not timed
    per_step = {}
    for g in stash:
        X, y = g.X.to(device), g.y_ctx.to(device)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        model(X, y)                                          # d = None, as in v2 training (ignore_d)
        torch.cuda.synchronize(device)
        per_step[g.step] = per_step.get(g.step, 0.0) + time.perf_counter() - t0
    return float(np.mean(list(per_step.values()))), torch.cuda.max_memory_allocated(device) / 2**30


# ------------------------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--n-steps", type=int, nargs="*", default=[8, 8, 4],
                    help="steps per stage, in --stages order (one value = all stages)")
    ap.add_argument("--workers", type=int, nargs="*", default=[16, 32])
    ap.add_argument("--seed", type=int, default=1000, help="any seed; the benchmark's tasks are not used elsewhere")
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--det-groups", type=int, default=2)
    ap.add_argument("--profile-tasks", type=int, default=4)
    ap.add_argument("--gpu-steps", type=int, default=2)
    ap.add_argument("--margin", type=float, default=0.8, help="gate: prior s/step <= margin * backbone s/step")
    ap.add_argument("--prefetch-factor", type=int, default=4, help="items in flight per worker (make_dataloader)")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    args = ap.parse_args()
    n_steps = dict(zip(args.stages, args.n_steps if len(args.n_steps) > 1 else args.n_steps * len(args.stages)))
    assert len(n_steps) == len(args.stages), "--n-steps needs one value or one per stage"

    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else None
    cpus = len(os.sched_getaffinity(0))
    print(f"CPUs available {cpus}; workers {args.workers}; device {device or 'none (gate not assessable)'}; "
          f"seed {args.seed}; R {args.R}; batch {args.batch_size}", flush=True)
    if max(args.workers) >= cpus:
        print(f"  warning: {max(args.workers)} workers on {cpus} CPUs leaves no core for the main process")
    model, sha = None, None
    if device:
        sha = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
        model = load_backbone(args.checkpoint, device)
        print(f"backbone {args.checkpoint} (sha256 {sha[:16]}...) on {torch.cuda.get_device_name(device)}")

    summary = dict(args=vars(args), cpus=cpus, checkpoint_sha256=sha,
                   gpu=torch.cuda.get_device_name(device) if device else None, stages={})
    for stage in args.stages:
        st = STAGES[stage]
        prior = CutGraphPrior(args.seed, stage, max_steps=n_steps[stage], R=args.R, batch_size=args.batch_size)
        print(f"\n# stage {stage}: N in [{st['min_seq_len'] or st['max_seq_len']}, {st['max_seq_len']}], "
              f"{prior.n_groups} groups of {prior.group_size} per step; {prior!r}", flush=True)
        out = dict(group_size=prior.group_size, n_groups=prior.n_groups, runs=[])

        # 1. determinism: in-process (workers = 0) vs the first W-worker run's items
        t0 = time.perf_counter()
        ref = [prior.group(0, g) for g in range(min(args.det_groups, prior.n_groups))]
        print(f"  in-process build of {len(ref)} groups: {time.perf_counter() - t0:.1f}s")

        # 2. throughput
        stash = None
        for w in args.workers:
            # the first run keeps step 0 at least: the determinism check compares its first groups
            timing, meta, st_items, log = run_loader(prior, w, max(1, args.gpu_steps) if stash is None else 0,
                                                     args.prefetch_factor)
            if stash is None:
                stash = st_items
                same = len(st_items) >= len(ref) and all(same_bits(a, b) for a, b in zip(ref, st_items))
                check(f"stage {stage}: groups delivered out of order by {w} workers and reordered by iterate_steps "
                      "are bit-identical to in-process ones, in group order", same, f"{len(ref)} groups")
            r = summarize_run(prior, w, timing, meta, log)
            out["runs"].append(r)
            in_flight = w * args.prefetch_factor
            print(f"  W={w:3d}: steady {r['steady_s_per_step']:.2f} s/step ({r['tasks_per_s']:.1f} tasks/s), "
                  f"wall {r['wall_s_per_step']:.2f} s/step (first step after {r['first_step_s']:.1f}s), "
                  f"ideal {r['ideal_s_per_step']:.2f} s/step (steady/ideal {r['steady_s_per_step'] / r['ideal_s_per_step']:.2f}); "
                  f"CPU s/task mean {r['cpu_s_per_task']:.2f} ({r['cpu_s_per_task_q']}); items arriving out of "
                  f"stream order {r['items_arrived_out_of_stream_order']:.0%}; items in flight {in_flight} = "
                  f"{in_flight / prior.n_groups:.1f} steps", flush=True)
            if in_flight < 2 * prior.n_groups:
                print(f"    note: fewer than 2 steps of items in flight; during a GPU step the workers can only "
                      f"build {in_flight} items before they idle (raise --prefetch-factor)")
        r = out["runs"][0]
        print(f"  stream sanity ({r['tasks']} tasks): rejection rate {r['rejection_rate']:.1%} {r['rejections']}; "
              f"graph resamples {r['graph_resamples']}; redraws (non-finite replicate) {r['redraws']}; "
              f"deterministic tasks {r['deterministic_share']:.1%}; t_A zero mass {r['tA_zero_mass']:.1%}\n"
              f"                 query types " + " ".join(f"{k} {v:.1%}" for k, v in r["query_type_share"].items())
              + f" (design: iid 66.5 / hole 14.25 / extrapolation 14.25 / duplicate 5); N {r['N_q']}; d {r['d_q']}")

        # 3. profile
        prof = [profile_task(args.seed, stage, i, prior.cfg, args.R) for i in range(args.profile_tasks)]
        out["profile"] = prof
        if prof:
            tot = sum(p["total_s"] for p in prof)
            shares = {k: sum(p[k] for p in prof) / tot for k in ("pool_and_filter_s", "split_s", "replicates_s")}
            print(f"  profile ({len(prof)} tasks, step 1): " + ", ".join(f"{k[:-2]} {v:.0%}" for k, v in shares.items())
                  + "; per task " + "; ".join(f"N={p['N']} P={p['P']} d={p['d']} attempts={p['attempts']} "
                                             f"{p['total_s']:.1f}s" for p in prof))

        # 4. GPU and verdict
        if model is not None:
            try:
                bb, mem = backbone_time(model, stash, device)
            except torch.cuda.OutOfMemoryError:
                bb, mem = None, None
                torch.cuda.empty_cache()
            out.update(backbone_s_per_step=bb, peak_gpu_gb=mem)
            if bb is None:
                check(f"stage {stage}: backbone forward fits on the GPU", False, "CUDA OOM on a stashed group")
            else:
                print(f"  backbone forward: {bb:.2f} s/step over {len({g.step for g in stash})} steps; "
                      f"peak GPU memory {mem:.1f} GiB")
                for r in out["runs"]:
                    ratio = r["steady_s_per_step"] / bb
                    r["prior_over_backbone"] = ratio
                    proj_h = DESIGN_STEPS[stage] * max(r["steady_s_per_step"], bb) / 3600
                    r["projected_stage_hours"] = proj_h
                    check(f"stage {stage}, W={r['workers']}: prior {r['steady_s_per_step']:.2f} s/step <= "
                          f"{args.margin} x backbone {bb:.2f} s/step", ratio <= args.margin,
                          f"ratio {ratio:.2f}; projected stage wall time {proj_h:.1f} h "
                          f"({DESIGN_STEPS[stage]} steps, A job)")
        else:
            check(f"stage {stage}: train-time floor measured", False, "no GPU: prior numbers only")
        summary["stages"][stage] = out
        del stash, ref

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    lines = ["| stage | W | steady s/step | tasks/s | ideal s/step | CPU s/task | backbone s/step | ratio | "
             "projected h |", "|---|---|---|---|---|---|---|---|---|"]
    for stage, o in summary["stages"].items():
        for r in o["runs"]:
            bb = o.get("backbone_s_per_step")
            lines.append(f"| {stage} | {r['workers']} | {r['steady_s_per_step']:.2f} | {r['tasks_per_s']:.1f} | "
                         f"{r['ideal_s_per_step']:.2f} | {r['cpu_s_per_task']:.2f} | "
                         f"{'-' if bb is None else f'{bb:.2f}'} | {r.get('prior_over_backbone', float('nan')):.2f} | "
                         f"{r.get('projected_stage_hours', float('nan')):.1f} |")
    (OUT / "throughput.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nwrote {OUT}\n{'ALL OK' if not FAILS else f'{len(FAILS)} FAILED: ' + '; '.join(FAILS)}", flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
