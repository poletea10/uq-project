"""Step 6 (phase1_protocol.md §3): build the validation bank, the mini-bank and the reference lines.

WHAT
    The 2,000-task validation bank of design Training §3 (1,000 small / 600 medium / 400 large tasks by N; R_val =
    128 replicates; its own seed VAL_BANK_SEED), the 64-task mini-bank of protocol §7.4 (the first 64 small tasks),
    and the reference lines of protocol §8.1 (noise floor, constant baseline, e(q) proxy; per head, per query type
    and per n bin). Layout and contents: `iltm2.prior._bank` docstring. Everything the trainer validates against is
    fixed here, once: the targets and e(q_theta) are precomputed with the frozen backbone's forward route.

HOW
    Tasks are generated on CPU by --n-jobs worker processes (`generate_bank_task`, a pure function of
    (seed, bin, index)), in bank order; as each task arrives, the main process runs the frozen TabICLv2 forward on
    the GPU and computes its targets, so CPU generation and GPU forwards overlap. Tasks are written in shards
    (100 small / 50 medium / 20 large tasks), each atomically: a killed or timed-out job is relaunched with the same
    command and resumes at the first missing shard. The first launch writes build_config.json (seed, R, sizes,
    prior config, backbone sha256); a relaunch with a different configuration is refused, so a bank is never a mix
    of two configurations. When all shards exist: queries.pt, mini/, manifest.json.

    The reference lines are wandb-ready flat {key: float} dicts (`iltm2.train._metrics.reference_lines`):
    manifest["reference_lines"] for the bank and manifest["mini"]["reference_lines"] for the mini-bank (the
    mini-bank uses the bank's constants and the bank's out-of-fold e(q) proxy, as the trainer's heads start from
    the bank-wide constant). Next: scripts/01b_check_val_bank.sh (the bank checks of protocol §3 step 6).

    sbatch experiment1_uq_heads/scripts/01_make_val_bank.sh
    sbatch experiment1_uq_heads/scripts/01_make_val_bank.sh --n-tasks 8 4 2 \
        --out /private/groups/ioannidislab/UQ_decoupling_exp/val_bank_smoke          # a smoke build first
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import torch
from joblib import Parallel, delayed

from iltm2.model._backbone import DEFAULT_BACKBONE, FrozenBackbone, file_sha256
from iltm2.prior._bank import (BANK_DIR, BIN_NAMES, BINS, MINI_SIZE, R_VAL, VAL_BANK_SEED, BankQueries,
                               atomic_save, config_dict, forecast_targets, generate_bank_task, shard_path,
                               shard_ranges, stream_stats, task_meta, task_offsets)
from iltm2.prior._cut_graph_scm import CutGraphConfig
from iltm2.prior._split import QUERY_TYPES
from iltm2.train._metrics import HEADS, N_BINS, eq_proxy_predictions, reference_lines, targets_of

REPO = Path(__file__).resolve().parents[2]                 # UQ_project


def git_state(path):
    try:
        head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True).strip())
        return dict(commit=head, dirty=dirty)
    except Exception as e:                                   # noqa: BLE001
        return dict(error=str(e))


def generate_in_order(jobs, n_jobs, seed, cfg, R):
    """Yield generated task records in submission order, generating up to ~2 n_jobs ahead."""
    calls = (delayed(generate_bank_task)(seed, b, i, cfg, R) for b, i in jobs)
    if tuple(int(x) for x in joblib.__version__.split(".")[:2]) >= (1, 3):
        yield from Parallel(n_jobs=n_jobs, return_as="generator", pre_dispatch="2*n_jobs")(calls)
    else:                                                    # old joblib: one Parallel call per 2 n_jobs tasks
        calls = list(calls)
        for lo in range(0, len(calls), 2 * n_jobs):
            yield from Parallel(n_jobs=n_jobs)(calls[lo:lo + 2 * n_jobs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=BANK_DIR)
    ap.add_argument("--seed", type=int, default=VAL_BANK_SEED)
    ap.add_argument("--n-tasks", type=int, nargs=3, default=[BINS[b]["n_tasks"] for b in BIN_NAMES],
                    metavar=("SMALL", "MEDIUM", "LARGE"))
    ap.add_argument("--R", type=int, default=R_VAL)
    ap.add_argument("--n-jobs", type=int, default=max(1, len(os.sched_getaffinity(0)) - 2))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--checkpoint", default=DEFAULT_BACKBONE)
    args = ap.parse_args()
    assert args.R % 2 == 0 and args.R >= 4, "R must be even (half-split targets)"
    n_tasks = dict(zip(BIN_NAMES, args.n_tasks))
    assert n_tasks["small"] >= MINI_SIZE or args.out != BANK_DIR, "the real bank needs >= 64 small tasks (mini)"
    cfg = CutGraphConfig()
    out = args.out
    (out / "shards").mkdir(parents=True, exist_ok=True)

    # ---- the configuration is fixed at the first launch; relaunches must match it
    sha = file_sha256(args.checkpoint)
    build = dict(seed=args.seed, R=args.R, n_tasks=n_tasks, bins={b: BINS[b] for b in BIN_NAMES},
                 prior_config=config_dict(cfg), checkpoint=args.checkpoint, checkpoint_sha256=sha)
    bc_path = out / "build_config.json"
    if bc_path.exists():
        old = json.loads(bc_path.read_text())
        if old != json.loads(json.dumps(build)):
            sys.exit(f"{bc_path} exists with a different configuration; use another --out or delete the bank.")
        print(f"resuming the build in {out}", flush=True)
    else:
        bc_path.write_text(json.dumps(build, indent=2))

    backbone = FrozenBackbone(args.checkpoint, args.device, tap_layers=None)
    print(f"{backbone!r}\nseed {args.seed}; R {args.R}; tasks {n_tasks}; n_jobs {args.n_jobs}; device "
          f"{torch.cuda.get_device_name() if args.device == 'cuda' else args.device}", flush=True)
    offsets = task_offsets(n_tasks)
    ranges = shard_ranges(n_tasks)
    todo = [r for r in ranges if not shard_path(out, r[0], r[1]).exists()]
    print(f"{len(ranges) - len(todo)}/{len(ranges)} shards already built", flush=True)

    # ---- generate (CPU workers) -> forecast targets (GPU, main process) -> shards
    jobs = [(b, i) for b, _, lo, hi in todo for i in range(lo, hi)]
    stream = generate_in_order(jobs, args.n_jobs, args.seed, cfg, args.R) if jobs else iter(())
    t0 = time.time()
    for b, k, lo, hi in todo:
        ts = time.time()
        tasks, qs = [], []
        for i in range(lo, hi):
            rec = next(stream)
            assert (rec["bin"], rec["index"]) == (b, i), "generation out of order"
            try:
                rec.update(forecast_targets(backbone, rec["X"], rec["y_ctx"], rec["Y"]))
            except torch.cuda.OutOfMemoryError as e:
                raise RuntimeError(f"CUDA OOM on bank task {b}/{i} (N={rec['N']}, d={rec['d']}); the bank needs the "
                                   "forward route on the training GPU, so do not fall back to CPU") from e
            tasks.append(rec)
            qs.append(BankQueries.from_task(rec, offsets[b] + i))
        Q = BankQueries.cat(qs)
        atomic_save(dict(tasks=tasks, queries=Q.to_dict(), meta=[task_meta(r) for r in tasks]), shard_path(out, b, k))
        print(f"  shard {b}_{k:03d}: tasks {lo}-{hi - 1}, {len(Q)} queries, N {tasks[0]['N']}-{tasks[-1]['N']}, "
              f"{time.time() - ts:.0f}s (total {(time.time() - t0) / 60:.1f} min)", flush=True)

    # ---- the whole bank: query table, mini-bank, reference lines, manifest
    metas, parts, mini_tasks = [], [], []
    for b, k, lo, hi in ranges:
        shard = torch.load(shard_path(out, b, k), weights_only=False)
        metas += shard["meta"]
        parts.append(BankQueries.from_dict(shard["queries"]))
        if b == "small" and len(mini_tasks) < MINI_SIZE:
            mini_tasks += shard["tasks"][:MINI_SIZE - len(mini_tasks)]
        del shard
    Q = BankQueries.cat(parts)
    assert len(metas) == sum(n_tasks.values()) and Q.task_id.max() == len(metas) - 1
    atomic_save(Q.to_dict(), out / "queries.pt")

    proxy = {h: eq_proxy_predictions(Q.e_q, targets_of(Q, h)[0], Q.task_id) for h in HEADS}
    refs = reference_lines(Q, eq_proxy=proxy)
    const = {h: refs[f"ref/const_value_{h}"] for h in HEADS}
    mini_mask = Q.task_id < len(mini_tasks)                 # small tasks come first: ids 0..63
    Qm = Q.subset(mini_mask)
    (out / "mini").mkdir(exist_ok=True)
    atomic_save(mini_tasks, out / "mini" / "tasks.pt")
    atomic_save(Qm.to_dict(), out / "mini" / "queries.pt")
    mini_refs = reference_lines(Qm, const=const, eq_proxy={h: proxy[h][mini_mask] for h in HEADS}, prefix="ref_mini")

    qt = torch.bincount(Q.query_type.long(), minlength=len(QUERY_TYPES)).double()
    nb = torch.bincount(Q.n_bin.long(), minlength=len(N_BINS))
    manifest = dict(
        created=datetime.now().isoformat(timespec="seconds"), seed=args.seed, R_val=args.R, n_tasks=n_tasks,
        task_offsets=offsets, n_queries=len(Q), bins={b: BINS[b] for b in BIN_NAMES},
        n_bins=[dict(name=n, lo_exclusive=lo, hi_inclusive=hi) for n, lo, hi in N_BINS],
        n_bin_of="the task's context size n_ctx", query_types=QUERY_TYPES,
        query_type_share={QUERY_TYPES[c]: float(qt[c] / qt.sum()) for c in QUERY_TYPES},
        n_bin_counts={N_BINS[i][0]: int(nb[i]) for i in range(len(N_BINS))},
        prior_config=config_dict(cfg), dup_fraction=cfg.dup_fraction, query_mix=cfg.query_mix,
        backbone=dict(path=args.checkpoint, sha256=sha, forward_route="FrozenBackbone (training-mode, fp32 SDPA)"),
        git=dict(UQ_project=git_state(REPO), tabicl=git_state(REPO / "external" / "tabicl")),
        software=dict(torch=torch.__version__, python=platform.python_version(),
                      device=torch.cuda.get_device_name() if args.device == "cuda" else args.device),
        stream={"all": stream_stats(metas, Q),
                **{b: stream_stats([m for m in metas if m["bin"] == b], Q.subset(Q.bin_idx == BIN_NAMES.index(b)))
                   for b in BIN_NAMES}},
        reference_lines=refs,
        mini=dict(n_tasks=len(mini_tasks), task_ids=list(range(len(mini_tasks))), n_queries=len(Qm),
                  reference_lines=mini_refs),
        shards=[shard_path(out, b, k).name for b, k, _, _ in ranges],
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    print(f"\nbank: {len(metas)} tasks, {len(Q)} queries; query types "
          + " ".join(f"{k} {v:.1%}" for k, v in manifest["query_type_share"].items())
          + f"; n bins {manifest['n_bin_counts']}")
    s = manifest["stream"]["all"]
    print(f"stream: rejection rate {s['rejection_rate']:.1%} {s['rejected']}; redraws {s['redraws']}; deterministic "
          f"tasks {s['deterministic_share']:.1%}; t_A zero mass {s['tA_zero_mass']:.1%}")
    for h in HEADS:
        print(f"reference lines {h} (pooled): noise floor {refs[f'ref/noise_floor_{h}/all']:.4g}, const xMSE "
              f"{refs[f'ref/const_xmse_{h}/all']:.4g}, e(q) proxy xMSE {refs[f'ref/eq_proxy_xmse_{h}/all']:.4g} "
              f"(skill {refs[f'ref/eq_proxy_skill_{h}/all']:.3f})")
    print(f"wrote {out} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
