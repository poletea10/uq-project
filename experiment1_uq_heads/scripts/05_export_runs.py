"""Export the Phase 1 runs' relevant metrics as small CSVs for offline analysis (phase1_protocol.md §9.3).

Curves come from the wandb API (only the needed keys); bests and timings from the run folders; reference lines from
the validation bank's manifest. Runs: A_<prior>_seed<S> (6 pairs each: head LR x p) and every B_* run (finished or
still running); the smoke run is skipped. Output (default <iltm2.paths.ROOT>/audits/export/):

    val.csv        full-bank validation, one row per run x pair x validation step x head x slice
    train.csv      training curves in 250-step windows, one row per run x pair x window
    reference.csv  the bank's reference lines, one row per head x slice
    runs.csv       one row per run x pair: configuration, status, best val MSE per head, cost per stage
    README.md      the data dictionary

wandb's scan_history(keys=...) returns only rows holding EVERY requested key, so keys are fetched in the groups that
are logged together: validation (ten per run: global steps 5,000 ... 30,000, 34,000, 38,000, 38,600, 39,200),
training (every step), mini-bank (every 250 steps).

    sbatch experiment1_uq_heads/scripts/05_export_runs.sh
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

from iltm2.inference import BEST
from iltm2.paths import ROOT, RUNS_ROOT
from iltm2.prior._bank import BANK_DIR, load_manifest
from iltm2.train._config import STAGE_STEPS
from iltm2.train._metrics import HEADS, SLICES

ENTITY_PROJECT = "poletea10-universitat-polit-cnica-de-catalunya/TabUD"   # first round: .../iLTMv2
WINDOW = 250
STAGE_END = np.cumsum([STAGE_STEPS[s] for s in (1, 2, 3)])          # global steps 30,000 / 38,000 / 39,200
VAL_METRICS = ("mse", "xmse", "skill", "wmse", "wxmse", "wskill", "mean")
A_RUN = re.compile(r"A_(carved|iid)_seed(\d+)")
B_RUN = re.compile(r"B_h([^_]+)_f([^_]+)_p([^_]+)_(carved|iid)_seed(\d+)")
A_PAIR = re.compile(r"A_lr([^_]+)_p([^_]+)(?:_v\d+)?")      # _v<k>: a repeated (LR, p) in one A job


def stage_of(gstep: int) -> int:
    return int(np.searchsorted(STAGE_END, gstep, side="left")) + 1


def run_config(run_dir: Path):
    """[(pair id, arm, head_lr, fork_lr, p)], the seed and the prior of a run folder."""
    m = A_RUN.fullmatch(run_dir.name)
    if m:
        pairs = sorted(json.loads((run_dir / "best_index.json").read_text())) if (
            run_dir / "best_index.json").exists() else []
        out = []
        for pid in pairs:
            pm = A_PAIR.fullmatch(pid)
            out.append((pid, "A", float(pm[1]), np.nan, float(pm[2])))
        return out, int(m[2]), m[1]
    m = B_RUN.fullmatch(run_dir.name)
    assert m, run_dir.name
    return [("B", "B", float(m[1]), float(m[2]), float(m[3]))], int(m[5]), m[4]


def scan(run, keys, page_size=5000):
    return pd.DataFrame(list(run.scan_history(keys=keys + ["_step"], page_size=page_size)))


def base_row(run_dir, pid, arm, hlr, flr, p, seed, prior):
    return dict(run=run_dir.name, pair=pid, arm=arm, head_lr=hlr, fork_lr=flr, p=p, prior=prior, seed=seed)


def export_val(run, run_dir, pairs, seed, prior, slices):
    rows = []
    for pid, arm, hlr, flr, p in pairs:
        keys = [f"{pid}/val/{m}_{h}/{s}" for h in HEADS for s in slices for m in VAL_METRICS]
        keys += [f"{pid}/val/{m}/{s}" for s in slices for m in ("ratio_A_eq", "ratio_tA_eq")]
        df = scan(run, keys)
        for _, r in df.iterrows():
            g = int(r["_step"])
            for h in HEADS:
                for s in slices:
                    row = base_row(run_dir, pid, arm, hlr, flr, p, seed, prior)
                    row.update(stage=stage_of(g), global_step=g, head=h, slice=s,
                               **{("mean_pred" if m == "mean" else m): r[f"{pid}/val/{m}_{h}/{s}"]
                                  for m in VAL_METRICS},
                               ratio_A_eq=r[f"{pid}/val/ratio_A_eq/{s}"] if h == "A" else np.nan,
                               ratio_tA_eq=r[f"{pid}/val/ratio_tA_eq/{s}"] if h == "A" else np.nan)
                    rows.append(row)
    return rows


def export_train(run, run_dir, pairs, seed, prior):
    """250-step windows: EMA at the window's end, gradient norms' mean and max, the mini-bank xMSE logged in it.
    All pairs' training metrics are logged in one call per step, so one scan fetches them (and the run's _runtime)."""
    gkeys = {pid: [f"{pid}/train/grad_norm_A", f"{pid}/train/grad_norm_E"]
             + ([f"{pid}/train/grad_norm_fork", f"{pid}/train/fork_dist_from_init"] if arm == "B" else [])
             for pid, arm, _, _, _ in pairs}
    keys = [k for pid, *_ in pairs for k in (f"{pid}/train/xmse_A_ema", f"{pid}/train/xmse_E_ema", *gkeys[pid])]
    tr = scan(run, keys + ["_runtime"])
    if tr.empty:
        return [], None
    mini = scan(run, [f"{pid}/mini/{m}_{h}/all" for pid, *_ in pairs for h in HEADS
                      for m in ("xmse", "wxmse")]).set_index("_step")
    tr["window_end"] = (np.ceil(tr["_step"] / WINDOW) * WINDOW).astype(int)
    rows = []
    for w, d in tr.groupby("window_end"):
        last = d.loc[d["_step"].idxmax()]
        hit = mini.loc[mini.index[(mini.index > w - WINDOW) & (mini.index <= w)]] if not mini.empty else mini
        for pid, arm, hlr, flr, p in pairs:
            row = base_row(run_dir, pid, arm, hlr, flr, p, seed, prior)
            row.update(stage=stage_of(int(w)), window_end_step=int(w), n_steps=len(d),
                       xmse_A_ema=last[f"{pid}/train/xmse_A_ema"], xmse_E_ema=last[f"{pid}/train/xmse_E_ema"])
            for h in HEADS:
                g = d[f"{pid}/train/grad_norm_{h}"]
                row.update({f"grad_norm_{h}_mean": g.mean(), f"grad_norm_{h}_max": g.max()})
            if arm == "B":
                g = d[f"{pid}/train/grad_norm_fork"]
                row.update(grad_norm_fork_mean=g.mean(), grad_norm_fork_max=g.max(),
                           fork_dist_from_init=last[f"{pid}/train/fork_dist_from_init"])
            for h in HEADS:
                for m in ("xmse", "wxmse"):
                    row[f"mini_{m}_{h}"] = hit[f"{pid}/mini/{m}_{h}/all"].iloc[-1] if len(hit) else np.nan
            rows.append(row)
    return rows, tr[["_step", "_runtime"]]


def export_reference(bank: Path, slices):
    refs = load_manifest(bank)["reference_lines"]
    rows = []
    for h in HEADS:
        for s in slices:
            if f"ref/count/{s}" not in refs:
                continue
            g = lambda k: refs.get(k, np.nan)  # noqa: E731
            row = dict(head=h, slice=s, count=g(f"ref/count/{s}"), mean_target=g(f"ref/mean_t{h}/{s}"),
                       mean_eq=g(f"ref/mean_eq/{s}"), ratio_target_eq=g(f"ref/ratio_t{h}_eq/{s}"),
                       loss_c=g("ref/loss_c"))
            for pre in ("", "w"):
                row.update({f"{pre}noise_floor": g(f"ref/{pre}noise_floor_{h}/{s}"),
                            f"{pre}const_value": g(f"ref/{pre}const_value_{h}"),
                            f"{pre}const_mse": g(f"ref/{pre}const_mse_{h}/{s}"),
                            f"{pre}const_xmse": g(f"ref/{pre}const_xmse_{h}/{s}"),
                            f"{pre}eq_proxy_mse": g(f"ref/{pre}eq_proxy_mse_{h}/{s}"),
                            f"{pre}eq_proxy_xmse": g(f"ref/{pre}eq_proxy_xmse_{h}/{s}"),
                            f"{pre}eq_proxy_skill": g(f"ref/{pre}eq_proxy_skill_{h}/{s}")})
            rows.append(row)
    return rows


def selected_heads():
    """{(run, pair, head): arm} from iltm2.inference.BEST (empty until the current round's selection)."""
    out = {}
    for arm, paths in BEST.items():
        for h, p in zip(HEADS, paths):
            run = Path(p).parents[2].name
            pair = "B" if arm == "B" else Path(p).stem.rsplit("_", 1)[0]
            out[(run, pair, h)] = arm
    return out


def export_runs(run, run_dir, pairs, seed, prior, runtime, selected):
    best = json.loads((run_dir / "best_index.json").read_text()) if (run_dir / "best_index.json").exists() else {}
    finished = [s for s in (1, 2, 3) if (run_dir / f"stage{s}" / "final.pt").exists()]
    stage_cost = {}
    for s in (1, 2, 3):
        f = run_dir / f"stage{s}" / "summary.json"
        d = json.loads(f.read_text()) if f.exists() else {}
        hours = np.nan
        if runtime is not None and not runtime.empty:
            r = runtime[runtime["_step"].map(lambda g: stage_of(int(g))) == s]
            if len(r) > 1:
                hours = (r["_runtime"].max() - r["_runtime"].min()) / 3600
        stage_cost[s] = dict(hours=hours, median_step_s=d.get("median_step_s", np.nan),
                             full_val_s=d.get("full_val_s", np.nan), cache_build_s=d.get("cache_build_s", np.nan),
                             peak_gpu_gb=d.get("peak_gpu_gb", np.nan), rejection_rate=d.get("rejection_rate", np.nan))
    rows = []
    for pid, arm, hlr, flr, p in pairs:
        row = base_row(run_dir, pid, arm, hlr, flr, p, seed, prior)
        row.update(wandb_state=run.state if run is not None else "missing", stages_finished=",".join(map(str, finished)),
                   best_weighting="weighted (wmse)" if p > 0 else "unweighted (mse)")
        for h in HEADS:
            b = best.get(pid, {}).get(h, {})
            row.update({f"best_val_{h}": b.get("val_mse", np.nan), f"best_stage_{h}": b.get("stage", np.nan),
                        f"best_global_step_{h}": b.get("global_step", np.nan),
                        f"selected_{h}": selected.get((run_dir.name, pid, h), "")})
        for s, c in stage_cost.items():
            row.update({f"s{s}_{k}": v for k, v in c.items()})
        rows.append(row)
    return rows


README = """# Phase 1 runs: exported metrics (iLTMv2, experiment1_uq_heads)

Two scalar heads on a frozen TabICLv2 regressor: Â (aleatoric, target t_A = CRPS entropy from R replicates) and
Ê_dir (epistemic, target t_E = the forecast's mean CRPS minus t_A). Arms: A = MLP trunk on h* (the final-LN output
the quantile decoder reads); B = trainable copy (fork) of the last 2 transformer blocks + LN feeding the heads. Loss:
mean of w (prediction - target)^2 with w = (e(q) + c)^-p, p in {0, 1}, c = 0.1 (fixed). A jobs:
head LR {3e-4, 1e-3, 3e-3} x p {0, 1} in one job; B: 18 configurations (head LR x fork LR {1e-5, 3e-5, 1e-4} x p) on
seed 0. Prior: carved (holes, extrapolation tails, half of them thinned, duplicates) or iid (TabICLv2's random split
only; ablation). Seeds fix the task stream and the initialization, so runs with the same seed see the same tasks.

Training: 3-stage curriculum, global steps 1-30,000 (stage 1, N = 1,024 rows; every task used twice: its full
context and a context subsampled to 64..n_ctx rows), 30,001-38,000 (stage 2, N 400-10,240), 38,001-39,200 (stage 3,
N 400-60,000); batch 32 tasks; AdamW, cosine schedule restarted per stage, max LR = swept LR / 1, / 8, / 40 per
stage. Every training task is fresh (no epochs). Validation: a fixed bank of 2,000 tasks (50/30/20 %
small/medium/large N; small tasks 256-1,024 rows at 80% context; 128 replicates per query) at global steps 5,000 ...
30,000, 34,000, 38,000, 38,600, 39,200. Best checkpoints per pair and head, under the pair's own weighting (p = 0:
mse, p = 1: wmse).

Definitions:
- mse: mean (prediction - target)^2 against the 128-replicate target (includes the targets' replicate noise).
- xmse: mean (prediction - t^(a)) (prediction - t^(b)) with the targets of the two halves of the replicates: the error
  against the realized task with the replicate noise removed (unbiased; single values can be negative).
- skill = 1 - xmse / xmse of the constant baseline (the bank-wide mean target, where every head starts); 1 = perfect.
- wmse, wxmse, wskill: the same with each query's term multiplied by w = (e(q) + c)^-1 (count-normalized), against the
  weighted constant baseline (the w-weighted mean target). Every run reports both weightings.
- Slices: all; query types iid (random queries), hole / hole_thinned (queries inside a carved-out region with no /
  few context rows), extrapolation / extrapolation_thinned (queries in a tail beyond the context, empty / thinned),
  duplicate (queries whose x copies a context row); context-size bins n_le256, n_256-512, n_512-1K, n_1K-4K,
  n_4K-16K, n_16K-60K, by the task's number of CONTEXT rows. Pooled values are query-weighted.
- e(q): the CRPS entropy of the backbone's own forecast. The e(q) proxy predicts the target from e(q) alone (50 bins,
  cross-fitted on task halves): its skill is what the backbone's spread gives for free.

Files:
- val.csv: run, pair (A: e.g. A_lr1e-3_p1; B: "B"), arm, head_lr, fork_lr (B only), p, prior, seed, stage,
  global_step, head (A/E), slice, mse, xmse, skill, wmse, wxmse, wskill, mean_pred; ratio_A_eq / ratio_tA_eq (head A
  only): mean of Â / e(q) and of t_A / e(q).
- train.csv: per 250-step window (window_end_step): xmse_A_ema, xmse_E_ema (exponential moving average, smoothing 0.99,
  of the per-step training xMSE, computed before each optimizer step on unseen tasks; reset at each stage start);
  grad_norm_{A,E}_{mean,max} (pre-clipping, clip at 5.0); B also grad_norm_fork_{mean,max}, fork_dist_from_init;
  mini_{xmse,wxmse}_{A,E}: on a fixed 64-task mini-bank (the bank's first small tasks), logged every 250 steps.
- reference.csv: per head and slice: count (queries), mean_target, mean_eq, ratio_target_eq, loss_c, and for both
  weightings ([w] prefix): noise_floor (the replicate-noise part of mse), const_value, const_mse / const_xmse
  (constant baseline), eq_proxy_mse / eq_proxy_xmse / eq_proxy_skill.
- runs.csv: per run and pair: wandb_state, stages_finished, best_weighting, best_val_{A,E} (under that weighting) with
  best_stage / best_global_step, selected_{A,E} (the arm whose selected checkpoint it is), and per stage s{1,2,3}_:
  hours (wall time), median_step_s, full_val_s (one full-bank validation), cache_build_s (input caches), peak_gpu_gb,
  rejection_rate (prior tasks rejected by the filters).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=Path, default=RUNS_ROOT)
    ap.add_argument("--bank", type=Path, default=BANK_DIR)
    ap.add_argument("--out", type=Path, default=ROOT / "audits" / "export")
    ap.add_argument("--project", default=ENTITY_PROJECT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=120)
    selected = selected_heads()
    refs = load_manifest(args.bank)["reference_lines"]
    slices = [s for s in SLICES if f"ref/count/{s}" in refs]
    run_dirs = sorted(p for p in args.runs_root.iterdir()
                      if p.is_dir() and (A_RUN.fullmatch(p.name) or B_RUN.fullmatch(p.name)))
    val, train, runs = [], [], []
    for rd in run_dirs:
        pairs, seed, prior = run_config(rd)
        try:
            run = api.run(f"{args.project}/{rd.name}")
        except Exception as e:                                   # noqa: BLE001
            print(f"{rd.name}: no wandb run ({type(e).__name__}); local files only", flush=True)
            runs += export_runs(None, rd, pairs, seed, prior, None, selected)
            continue
        v = export_val(run, rd, pairs, seed, prior, slices)
        t, runtime = export_train(run, rd, pairs, seed, prior)
        val += v
        train += t
        runs += export_runs(run, rd, pairs, seed, prior, runtime, selected)
        print(f"{rd.name} ({run.state}): {len(pairs)} pairs; {len(v)} val rows, {len(t)} train rows", flush=True)
    frames = dict(val=pd.DataFrame(val), train=pd.DataFrame(train),
                  reference=pd.DataFrame(export_reference(args.bank, slices)), runs=pd.DataFrame(runs))
    for name, df in frames.items():
        path = args.out / f"{name}.csv"
        df.to_csv(path, index=False, float_format="%.6g")
        print(f"wrote {path}: {len(df)} rows x {df.shape[1]} columns ({path.stat().st_size / 1e6:.2f} MB)", flush=True)
    (args.out / "README.md").write_text(README)
    print(f"wrote {args.out / 'README.md'}", flush=True)


if __name__ == "__main__":
    main()
