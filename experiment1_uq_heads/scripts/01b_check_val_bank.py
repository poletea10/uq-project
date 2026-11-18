"""Step-6 gate (phase1_protocol.md §3 step 6, "Bank checks, before any training"; transfer guide §5 Step 2.6): is the
validation bank right?

WHY
    Every validation curve, checkpoint choice and arm comparison of Phase 1 is read against this bank. If its
    targets are wrong, or it does not come from the training stream, or the frozen backbone is off-domain on it,
    every later number is silently wrong. These checks run once, after scripts/01_make_val_bank.sh, before any
    training.

CHECKS
    0. Integrity: task and query counts match the manifest; every target finite; t_A >= 0, e(q) >= 0;
       score = t_A + t_E; query-type shares within 1 point of the design's iid 66.5 / hole (empty + thinned) 14.25 /
       extrapolation (empty + thinned) 14.25 / duplicate 5, every query type present, and the empty / thinned split
       of each region reported; every n bin the bank's shapes reach is populated; thinning: r <= 0.04 N per region,
       and a task's region queries are `_thinned` exactly when its r > 0; the mini-bank is exactly the bank's first
       64 small tasks.
    W. The weight floor c (transfer guide §3.6): c = the fixed LOSS_C (0.1) unless set with --loss-c, and the
       reference lines were computed with it; the e(q) quantiles, the share of the queries at or below c (and how
       many of them come from deterministic tasks) and the range of the weights are reported.
    1. Reproducibility: --repro-tasks tasks per bin (the first ones) are regenerated from their keys
       (seed, bin, index) and must be bit-identical to the stored tasks (X, y_ctx, replicates, query types), and
       their forecast targets recomputed with the frozen backbone must equal the stored ones (bit-equal, or within
       1e-6). This ties the bank to the code that generates the training stream and to the backbone's forward route.
    2. Target ordering by query type (protocol §3 step 6; transfer guide §5 Step 2.6): mean t_E ordered
       duplicate < iid < thinned < empty, for the hole and for the extrapolation tail, gated on the pooled bank with
       task-level bootstrap 95% CIs of the differences (each CI must exclude 0 in the right direction); hole vs
       extrapolation is reported (expected similar, not gated); the same orderings per N bin (small / medium /
       large) are reported. Also the broken-stream
       red flag of protocol §8.3: the mean t_A of duplicates and of i.i.d. queries must agree (CI of the difference
       covers 0), since duplicates copy the x of context rows, which are i.i.d. rows of the same pool.
    3. The frozen PPD is sane on the cut prior (protocol §3 step 6): for every bank task (or --max-gs-tasks per
       bin), an unmodified graph_scm task of the same (N, n_ctx, d), generated as TabICLv2 was pretrained
       (`iltm2.prior._graph_scm_ref`) and split at random; TabICLv2's mean CRPS on its queries vs the bank task's
       mean CRPS on its i.i.d. queries (the stored score, the replicate-averaged CRPS, which is unbiased for the
       expected CRPS of one draw). Gate per N bin: the ratio of mean CRPS cut / graph_scm within [1/tol, tol]
       (--crps-ratio-tol 1.25). Reported: medians, the paired median log ratio with a bootstrap CI, rejection rates.
       This reruns the in-domain question of scripts/00f_difficulty_match.py on the bank itself, with its query mix.

OUTPUT
    One "OK  ..." / "FAIL ..." line per check; outputs/audits/val_bank_checks/<bank dir name>/summary.json and
    per-task tables; exit 1 if any check failed.

    sbatch experiment1_uq_heads/scripts/01b_check_val_bank.sh
    sbatch experiment1_uq_heads/scripts/01b_check_val_bank.sh --bank /private/groups/ioannidislab/UQ_decoupling_exp_new/val_bank_v2_smoke
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

from iltm2.model._backbone import DEFAULT_BACKBONE, FrozenBackbone
from iltm2.model._entropy import crps
from iltm2.prior._bank import (BANK_DIR, BIN_NAMES, BINS, LOSS_C, MINI_SIZE, forecast_targets, generate_bank_task,
                               load_manifest, load_queries, shard_path, shard_ranges)
from iltm2.prior._dataset import STREAM_SPLIT, STREAM_TASK, single_thread, task_rng
from iltm2.prior._filter import RejectionLog
from iltm2.prior._graph_scm_ref import graph_scm_task
from iltm2.prior._split import EXT, EXT_THIN, HOLE, HOLE_THIN, IID, QUERY_TYPES
from iltm2.train._metrics import N_BINS

OUT = Path(__file__).resolve().parents[1] / "outputs" / "audits" / "val_bank_checks"
# Design shares of the query types (query_mix 70 / 15 / 15, then 5% duplicates drawn uniformly over the slots); a
# region's empty and thinned types share its 14.25%.
DESIGN_SHARE = {"iid": 0.665, "hole": 0.1425, "extrapolation": 0.1425, "duplicate": 0.05}
COMBINED = {"iid": ("iid",), "hole": ("hole", "hole_thinned"), "extrapolation": ("extrapolation",
                                                                                  "extrapolation_thinned"),
            "duplicate": ("duplicate",)}
GS_STEP_OFFSET = 100            # graph_scm tasks use task_rng(seed, 100 + bin, index): disjoint from the bank's keys
FAILS = []


def check(name, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'} {name}{'  ' + detail if detail else ''}", flush=True)
    if not ok:
        FAILS.append(name)
    return ok


def same_bits(a, b):
    return (a.shape == b.shape and a.dtype == b.dtype
            and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


# ------------------------------------------------------------------------------------------- bootstrap helpers


def per_task_sums(Q, value, mask, n_tasks):
    """Per task: sum of `value` and count over the queries in `mask` (numpy, length n_tasks)."""
    tid = Q.task_id[mask].numpy()
    v = value[mask].double().numpy()
    return np.bincount(tid, weights=v, minlength=n_tasks), np.bincount(tid, minlength=n_tasks).astype(float)


def boot_ratio_means(sums, cnts, idx):
    """Mean over queries of the resampled tasks, for each bootstrap row of task indices `idx` (B, T)."""
    return sums[idx].sum(axis=1) / np.maximum(cnts[idx].sum(axis=1), 1)


def type_means_with_boot(Q, value, task_ids, n_boot, seed):
    """{type: (point mean, bootstrap array)} over the tasks in `task_ids`, resampling tasks (clusters)."""
    n_tasks = int(Q.task_id.max()) + 1
    rng = np.random.default_rng(seed)
    idx = task_ids[rng.integers(0, len(task_ids), size=(n_boot, len(task_ids)))]
    out = {}
    for code, name in QUERY_TYPES.items():
        s, c = per_task_sums(Q, value, Q.query_type == code, n_tasks)
        out[name] = (float(s[task_ids].sum() / max(c[task_ids].sum(), 1)), boot_ratio_means(s, c, idx))
    return out


def diff_ci(a, b):
    """(point a - b, 2.5% and 97.5% of the bootstrap distribution of a - b) for two (point, boot) pairs."""
    d = a[1] - b[1]
    return a[0] - b[0], float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


# ------------------------------------------------------------------------------------------- 0. integrity


def check_integrity(root, man, Q):
    print("\n# 0. integrity")
    n_tasks = man["n_tasks"]
    check("query count matches the manifest", len(Q) == man["n_queries"], f"{len(Q)} queries")
    for b in BIN_NAMES:
        ids = torch.unique(Q.task_id[Q.bin_idx == BIN_NAMES.index(b)])
        check(f"{b}: {n_tasks[b]} tasks with queries", len(ids) == n_tasks[b], f"{len(ids)}")
    vals = [Q.e_q, Q.t_A, Q.t_E, Q.t_A_a, Q.t_A_b, Q.t_E_a, Q.t_E_b, Q.score]
    check("every target, e(q) and score is finite", all(bool(torch.isfinite(v).all()) for v in vals))
    check("t_A >= 0 (full and halves) and e(q) >= 0",
          bool((Q.t_A >= 0).all() and (Q.t_A_a >= 0).all() and (Q.t_A_b >= 0).all() and (Q.e_q >= 0).all()))
    gap = float((Q.score.double() - Q.t_A.double() - Q.t_E.double()).abs().max())
    check("score = t_A + t_E", gap <= 1e-5, f"max |gap| {gap:.1e}")
    qt = torch.bincount(Q.query_type.long(), minlength=len(QUERY_TYPES)).double() / len(Q)
    shares = {QUERY_TYPES[c]: float(qt[c]) for c in QUERY_TYPES}
    comb = {k: sum(shares[t] for t in ts) for k, ts in COMBINED.items()}
    check("query-type shares within 1 point of the design (regions: empty + thinned)",
          all(abs(comb[k] - DESIGN_SHARE[k]) <= 0.01 for k in DESIGN_SHARE),
          " ".join(f"{k} {v:.2%}" for k, v in comb.items()))
    check("every query type is present", all(v > 0 for v in shares.values()),
          " ".join(f"{k} {v:.2%}" for k, v in shares.items()))
    for reg in ("hole", "extrapolation"):
        e, t = shares[reg], shares[f"{reg}_thinned"]
        print(f"      reported: {reg} queries {e / (e + t):.1%} empty / {t / (e + t):.1%} thinned (design: half the "
              f"regions with non-query rows thinned; regions with none are always empty)")
    nb = torch.bincount(Q.n_bin.long(), minlength=len(man["n_bins"]))
    lo_ctx = min(int(BINS[b]["ctx"][0] * BINS[b]["N"][0]) for b in BIN_NAMES)
    reach = [i for i, (_, lo, hi) in enumerate(N_BINS) if hi >= lo_ctx]
    check("every n bin the bank's shapes reach is populated", all(int(nb[i]) > 0 for i in reach),
          " ".join(f"{b['name']} {int(c)}" for b, c in zip(man["n_bins"], nb)))
    metas = []
    for b, k, _, _ in shard_ranges(n_tasks):
        metas += torch.load(shard_path(root, b, k), weights_only=False)["meta"]
    bad_r = [m for m in metas if max(m["r_ext"], m["r_hole"]) > 0.04 * m["N"] + 0.5]
    check("thinning: r <= 0.04 N in every region", not bad_r, f"{len(bad_r)} tasks above")
    qt_task = {}
    for tid, code in zip(Q.task_id.tolist(), Q.query_type.tolist()):
        qt_task.setdefault(tid, set()).add(code)
    bad_lab = 0
    for m in metas:
        codes = qt_task.get(man["task_offsets"][m["bin"]] + m["index"], set())
        for r, empty, thin in ((m["r_ext"], EXT, EXT_THIN), (m["r_hole"], HOLE, HOLE_THIN)):
            bad_lab += (empty in codes) if r > 0 else (thin in codes)
    check("a region's queries are _thinned exactly when its r > 0", bad_lab == 0, f"{bad_lab} bad regions")
    Qm = load_queries(root, mini=True)
    ref = Q.subset(Q.task_id < man["mini"]["n_tasks"])
    check(f"mini-bank = the bank's first {man['mini']['n_tasks']} small tasks",
          man["mini"]["n_tasks"] == min(MINI_SIZE, n_tasks["small"])
          and all(torch.equal(getattr(Qm, k), getattr(ref, k)) for k in Qm.to_dict()))
    return shares


# ------------------------------------------------------------------------------------------- W. the weight floor c


def check_loss_c(man, Q):
    print("\n# W. the weight floor c")
    c, med = man["loss_c"], float(torch.median(Q.e_q.double()))
    check("c > 0", c > 0, f"c = {c:.4g}")
    if man["loss_c_rule"] == "--loss-c":
        print(f"      reported: c set with --loss-c ({c:.4g}) instead of the fixed {LOSS_C}")
    else:
        check(f"c = the fixed {LOSS_C}", c == LOSS_C, f"c {c:.4g}")
    check("the reference lines were computed with this c", man["reference_lines"].get("ref/loss_c") == c
          and man["mini"]["reference_lines"].get("ref_mini/loss_c") == c, f"ref/loss_c {man['reference_lines'].get('ref/loss_c')}")
    print("      e(q) quantiles: " + " ".join(f"q{float(p) * 100:g}={v:.4g}" for p, v in man["e_q_quantiles"].items())
          + f" (median {med:.4g})")
    b = man["e_q_below_c"]
    print(f"      queries with e(q) <= c: {b['share_of_queries']:.1%}, of them {b['share_from_deterministic_tasks']:.1%} "
          f"from deterministic tasks, {b['share_t_A_zero']:.1%} with t_A = 0; weights: max 1/c = {1 / c:.4g}, "
          + " ".join(f"q{float(p) * 100:g}={v:.3g}" for p, v in man["weights"]["quantiles"].items()))
    return dict(loss_c=c, median=med, quantiles=man["e_q_quantiles"], below_c=b, weights=man["weights"])


# ------------------------------------------------------------------------------------------- 1. reproducibility


def check_repro(root, man, backbone, n_per_bin):
    print(f"\n# 1. reproducibility ({n_per_bin} tasks per bin, regenerated from their keys)")
    rows, shards = [], {}
    for b in BIN_NAMES:
        size = BINS[b]["shard"]
        for i in range(min(n_per_bin, man["n_tasks"][b])):
            if (b, i // size) not in shards:
                shards[(b, i // size)] = torch.load(shard_path(root, b, i // size), weights_only=False)["tasks"]
            stored = shards[(b, i // size)][i % size]
            fresh = generate_bank_task(man["seed"], b, i, R=man["R_val"])
            same_task = all(same_bits(stored[k], fresh[k]) for k in ("X", "y_ctx", "Y", "query_type"))
            tg = forecast_targets(backbone, stored["X"], stored["y_ctx"], stored["Y"])
            diffs = {k: float((tg[k].double() - stored[k].double()).abs().max()) for k in tg}
            rows.append(dict(bin=b, index=i, same_task=same_task, max_target_diff=max(diffs.values()),
                             bit_equal_targets=all(torch.equal(tg[k], stored[k]) for k in tg)))
    df = pd.DataFrame(rows)
    check("regenerated tasks are bit-identical to the stored ones (X, y_ctx, replicates, query types)",
          bool(df.same_task.all()), f"{int(df.same_task.sum())}/{len(df)}")
    check("recomputed forecast targets equal the stored ones", bool((df.max_target_diff <= 1e-6).all()),
          f"bit-equal {int(df.bit_equal_targets.sum())}/{len(df)}; max |diff| {df.max_target_diff.max():.1e}")
    return df


# ------------------------------------------------------------------------------------------- 2. target ordering


def check_ordering(man, Q, n_boot):
    print("\n# 2. target ordering by query type (task-level bootstrap 95% CIs)")
    n_tasks = int(Q.task_id.max()) + 1
    out = {}
    groups = {"all": np.arange(n_tasks)}
    for b in BIN_NAMES:
        lo = man["task_offsets"][b]
        groups[b] = np.arange(lo, lo + man["n_tasks"][b])
    for g, ids in groups.items():
        tE = type_means_with_boot(Q, Q.t_E, ids, n_boot, seed=1)
        tA = type_means_with_boot(Q, Q.t_A, ids, n_boot, seed=2)
        d = {"iid - duplicate": diff_ci(tE["iid"], tE["duplicate"]),
             "hole_thinned - iid": diff_ci(tE["hole_thinned"], tE["iid"]),
             "hole - hole_thinned": diff_ci(tE["hole"], tE["hole_thinned"]),
             "extrapolation_thinned - iid": diff_ci(tE["extrapolation_thinned"], tE["iid"]),
             "extrapolation - extrapolation_thinned": diff_ci(tE["extrapolation"], tE["extrapolation_thinned"]),
             "hole - extrapolation": diff_ci(tE["hole"], tE["extrapolation"])}
        dA = diff_ci(tA["duplicate"], tA["iid"])
        out[g] = dict(mean_tE={k: v[0] for k, v in tE.items()}, mean_tA={k: v[0] for k, v in tA.items()},
                      tE_diffs={k: list(v) for k, v in d.items()}, tA_dup_minus_iid=list(dA))
        print(f"  [{g}] mean t_E " + " ".join(f"{k} {v[0]:.4g}" for k, v in tE.items())
              + " | mean t_A " + " ".join(f"{k} {v[0]:.4g}" for k, v in tA.items()))
        for k, (p, lo_, hi_) in d.items():
            print(f"      t_E {k}: {p:+.4g} [{lo_:+.4g}, {hi_:+.4g}]")
        print(f"      t_A duplicate - iid: {dA[0]:+.4g} [{dA[1]:+.4g}, {dA[2]:+.4g}]")
        if g == "all":
            for k in ("iid - duplicate", "hole_thinned - iid", "hole - hole_thinned", "extrapolation_thinned - iid",
                      "extrapolation - extrapolation_thinned"):
                check(f"pooled: mean t_E {k} > 0 (CI excludes 0)", d[k][1] > 0,
                      f"{d[k][0]:+.4g} [{d[k][1]:+.4g}, {d[k][2]:+.4g}]")
            check("pooled: mean t_A duplicate = iid (CI of the difference covers 0; protocol §8.3)",
                  dA[1] <= 0 <= dA[2], f"{dA[0]:+.4g} [{dA[1]:+.4g}, {dA[2]:+.4g}]")
            rel = abs(d["hole - extrapolation"][0]) / max(tE["hole"][0], tE["extrapolation"][0])
            print(f"      reported: hole vs extrapolation mean t_E differ by {rel:.0%} of the larger "
                  f"(expected similar, not gated)")
    return out


# ------------------------------------------------------------------------------------------- 3. PPD sanity


def graph_scm_pair(seed, bin_idx, index, N, n_ctx, d):
    """An unmodified graph_scm task with the bank task's (N, n_ctx, d), and its random split (context first)."""
    from types import SimpleNamespace
    log = RejectionLog()
    with single_thread():
        with task_rng(seed, GS_STEP_OFFSET + bin_idx, index, STREAM_TASK):
            X, y = graph_scm_task(SimpleNamespace(N=N, n_ctx=n_ctx, d=d), log)
        with task_rng(seed, GS_STEP_OFFSET + bin_idx, index, STREAM_SPLIT):
            perm = torch.randperm(N)
    return X[perm].float(), y[perm].float(), log


def check_ppd(root, man, Q, backbone, max_per_bin, n_jobs, tol, n_boot):
    print("\n# 3. the frozen PPD on the bank vs on matched unmodified graph_scm tasks")
    metas = []
    for b, k, _, _ in shard_ranges(man["n_tasks"]):
        metas += torch.load(shard_path(root, b, k), weights_only=False)["meta"]
    sel = [m for b in BIN_NAMES for m in [x for x in metas if x["bin"] == b][:max_per_bin or None]]
    t0 = time.time()
    gens = Parallel(n_jobs=n_jobs)(delayed(graph_scm_pair)(man["seed"], BIN_NAMES.index(m["bin"]), m["index"],
                                                           m["N"], m["n_ctx"], m["d_drawn"]) for m in sel)
    print(f"  generated {len(sel)} graph_scm tasks in {(time.time() - t0) / 60:.1f} min", flush=True)
    n_tasks = int(Q.task_id.max()) + 1
    s_iid, c_iid = per_task_sums(Q, Q.score, Q.query_type == IID, n_tasks)
    rows, gs_log = [], RejectionLog()
    for m, (X, y, log) in zip(sel, gens):
        gs_log += log
        o = backbone(X[None], y[None, :m["n_ctx"]])
        gs = float(crps(o.q[0].double(), y[m["n_ctx"]:].to(o.q.device).double()[:, None]).mean())
        tid = man["task_offsets"][m["bin"]] + m["index"]
        rows.append(dict(bin=m["bin"], index=m["index"], N=m["N"], n_ctx=m["n_ctx"], d=m["d_drawn"],
                         cut_iid_crps=s_iid[tid] / max(c_iid[tid], 1), n_iid=int(c_iid[tid]), gs_crps=gs))
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(3)
    out = dict(gs_rejection_rate=gs_log.rate, gs_rejected=dict(gs_log.rejected), bins={})
    print(f"  graph_scm rejection rate {gs_log.rate:.1%} {dict(gs_log.rejected)} (bank: "
          f"{man['stream']['all']['rejection_rate']:.1%})")
    for b in (*BIN_NAMES, "all"):
        sub = df if b == "all" else df[df.bin == b]
        if sub.empty:
            continue
        cut, gs = sub.cut_iid_crps.to_numpy(), sub.gs_crps.to_numpy()
        idx = rng.integers(0, len(sub), size=(n_boot, len(sub)))
        ratio = cut.mean() / gs.mean()
        boot = cut[idx].mean(1) / gs[idx].mean(1)
        lr = np.log(np.maximum(cut, 1e-12) / np.maximum(gs, 1e-12))
        lr_boot = np.median(lr[idx], axis=1)
        r = dict(n=len(sub), cut_mean=float(cut.mean()), gs_mean=float(gs.mean()), cut_median=float(np.median(cut)),
                 gs_median=float(np.median(gs)), ratio_of_means=float(ratio),
                 ratio_ci=[float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                 median_log_ratio=float(np.median(lr)),
                 median_log_ratio_ci=[float(np.quantile(lr_boot, 0.025)), float(np.quantile(lr_boot, 0.975))])
        out["bins"][b] = r
        text = (f"cut iid {r['cut_mean']:.4g} (median {r['cut_median']:.4g}) vs graph_scm {r['gs_mean']:.4g} "
                f"(median {r['gs_median']:.4g}); ratio {ratio:.3f} [{r['ratio_ci'][0]:.3f}, {r['ratio_ci'][1]:.3f}]; "
                f"paired median log ratio {r['median_log_ratio']:+.3f}; n={len(sub)}")
        if b == "all":
            print(f"  reported [all]: {text}")
        else:
            check(f"[{b}] mean CRPS ratio cut / graph_scm within [{1 / tol:.2f}, {tol:.2f}]",
                  1 / tol <= ratio <= tol, text)
    return df, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, default=BANK_DIR)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--checkpoint", default=DEFAULT_BACKBONE)
    ap.add_argument("--repro-tasks", type=int, default=3, help="regenerated tasks per bin (check 1)")
    ap.add_argument("--max-gs-tasks", type=int, default=0, help="graph_scm tasks per bin, 0 = every bank task")
    ap.add_argument("--crps-ratio-tol", type=float, default=1.25)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-jobs", type=int, default=max(1, len(os.sched_getaffinity(0)) - 2))
    args = ap.parse_args()

    man, Q = load_manifest(args.bank), load_queries(args.bank)
    out_dir = OUT / args.bank.name
    out_dir.mkdir(parents=True, exist_ok=True)
    backbone = FrozenBackbone(args.checkpoint, args.device)
    print(f"bank {args.bank} (seed {man['seed']}, {sum(man['n_tasks'].values())} tasks, {len(Q)} queries)\n"
          f"{backbone!r}", flush=True)
    check("backbone sha256 matches the bank's", backbone.sha256 == man["backbone"]["sha256"],
          f"{backbone.sha256[:16]}... vs {man['backbone']['sha256'][:16]}...")

    summary = dict(bank=str(args.bank), args={k: str(v) for k, v in vars(args).items()})
    summary["query_type_share"] = check_integrity(args.bank, man, Q)
    summary["loss_c"] = check_loss_c(man, Q)
    repro = check_repro(args.bank, man, backbone, args.repro_tasks)
    repro.to_csv(out_dir / "reproducibility.csv", index=False)
    summary["ordering"] = check_ordering(man, Q, args.n_boot)
    ppd_df, summary["ppd"] = check_ppd(args.bank, man, Q, backbone, args.max_gs_tasks, args.n_jobs,
                                       args.crps_ratio_tol, args.n_boot)
    ppd_df.to_parquet(out_dir / "ppd_tasks.parquet")
    summary["fails"] = FAILS
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {out_dir}\n{'ALL OK' if not FAILS else f'{len(FAILS)} FAILED: ' + '; '.join(FAILS)}", flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
