"""Step 6 (phase1_protocol.md §3; transfer guide §3.4, §5 Step 2.6): build the validation bank val_bank_v2, the
mini-bank, the weight floor c and the reference lines.

WHAT
    The 2,000-task validation bank of transfer guide §3.4 (1,000 small tasks of 256-1,024 rows at 80% context / 600
    medium / 400 large; carved prior with thinned regions; R_val = 128 replicates; its own seed VAL_BANK_SEED), the
    64-task mini-bank of protocol §7.4 (the first 64 small tasks), the weight floor c of the p = 1 loss and metrics
    (fixed at LOSS_C = 0.1, manifest["loss_c"]; transfer guide §3.6), and the reference
    lines of protocol §8.1 for both weightings (noise floor, constant baseline, e(q) proxy; per head, per query type
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
    the bank-wide constant). The e(q) quantiles are printed and stored (manifest["e_q_quantiles"]), with the weights'
    range under c. To change c, relaunch the same command with --loss-c <value>: the shards are kept and only the
    manifest and the reference lines are rewritten (the same holds when c changes in the code). Next: scripts/01b_check_val_bank.sh (the bank checks of protocol §3 step 6).

    sbatch experiment1_uq_heads/scripts/01_make_val_bank.sh
    sbatch experiment1_uq_heads/scripts/01_make_val_bank.sh --n-tasks 8 4 2 \
        --out /private/groups/ioannidislab/UQ_decoupling_exp_new/val_bank_v2_smoke          # a smoke build first
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
from iltm2.prior._bank import (BANK_DIR, BIN_NAMES, BINS, LOSS_C, MINI_SIZE, R_VAL, VAL_BANK_SEED, BankQueries,
                               atomic_save, config_dict, forecast_targets, generate_bank_task, shard_path,
                               shard_ranges, stream_stats, task_meta, task_offsets)
from iltm2.prior._cut_graph_scm import CutGraphConfig
from iltm2.prior._split import QUERY_TYPES
from iltm2.train._metrics import HEADS, N_BINS, const_values, proxy_predictions, reference_lines

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
    ap.add_argument("--loss-c", type=float, default=None,
                    help=f"the weight floor c (default: the fixed {LOSS_C}, transfer guide §3.6)")
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

    backbone = FrozenBackbone(args.checkpoint, args.device)
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

    e_q = Q.e_q.double()
    eq_quant = {str(p): float(torch.quantile(e_q, p)) for p in (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)}
    c_median = float(torch.median(e_q))
    loss_c = LOSS_C if args.loss_c is None else float(args.loss_c)
    assert loss_c > 0, f"the weight floor c must be positive, got {loss_c}"
    det_tasks = torch.tensor([m["deterministic"] for m in metas])
    det_q = det_tasks[Q.task_id]
    below = e_q <= loss_c
    w = 1.0 / (e_q + loss_c)
    proxy = proxy_predictions(Q, loss_c)
    const = const_values(Q, loss_c)
    refs = reference_lines(Q, loss_c, const=const, eq_proxy=proxy)
    mini_mask = Q.task_id < len(mini_tasks)                 # small tasks come first: ids 0..63
    Qm = Q.subset(mini_mask)
    (out / "mini").mkdir(exist_ok=True)
    atomic_save(mini_tasks, out / "mini" / "tasks.pt")
    atomic_save(Qm.to_dict(), out / "mini" / "queries.pt")
    mini_refs = reference_lines(Qm, loss_c, const=const, eq_proxy={k: v[mini_mask] for k, v in proxy.items()},
                                prefix="ref_mini")

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
        loss_c=loss_c, loss_c_rule=f"fixed ({LOSS_C}, transfer guide §3.6)" if args.loss_c is None else "--loss-c",
        e_q_median=c_median, e_q_quantiles=eq_quant,
        e_q_below_c=dict(share_of_queries=float(below.double().mean()),
                         share_from_deterministic_tasks=float(det_q[below].double().mean()),
                         share_t_A_zero=float((Q.t_A[below] == 0).double().mean())),
        weights=dict(max=float(1.0 / loss_c), quantiles={str(p): float(torch.quantile(w, p))
                                                       for p in (0.01, 0.25, 0.5, 0.75, 0.99)}),
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
    for reg, d in s["thinning"].items():
        print(f"thinning {reg}: {d['thinned_share']:.1%} of {d['tasks_with_free_rows']} regions thinned; realized rho "
              f"{d['rho_quantiles']}")
    print("e(q) quantiles: " + " ".join(f"q{float(p) * 100:g}={v:.4g}" for p, v in eq_quant.items()))
    b = manifest["e_q_below_c"]
    print(f"weight floor c = {loss_c:.4g} ({manifest['loss_c_rule']}; median e(q) {c_median:.4g}); {b['share_of_queries']:.1%} "
          f"of the queries have e(q) <= c, {b['share_from_deterministic_tasks']:.1%} of them from deterministic tasks, "
          f"{b['share_t_A_zero']:.1%} with t_A = 0; weights w = 1/(e(q) + c): max {1 / loss_c:.3g}, quantiles "
          + " ".join(f"q{float(p) * 100:g}={v:.3g}" for p, v in manifest["weights"]["quantiles"].items()))
    for h in HEADS:
        for pre, name in (("", "unweighted"), ("w", "weighted")):
            print(f"reference lines {h} {name} (pooled): noise floor {refs[f'ref/{pre}noise_floor_{h}/all']:.4g}, "
                  f"const xMSE {refs[f'ref/{pre}const_xmse_{h}/all']:.4g}, e(q) proxy xMSE "
                  f"{refs[f'ref/{pre}eq_proxy_xmse_{h}/all']:.4g} (skill {refs[f'ref/{pre}eq_proxy_skill_{h}/all']:.3f})")
    print(f"wrote {out} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
