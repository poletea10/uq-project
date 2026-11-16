"""Export the Phase 1 runs' relevant metrics as small CSVs for offline analysis (phase1_protocol.md §9.3).

Curves come from the wandb API (only the needed keys); bests and timings from the run folders; reference lines from
the validation bank's manifest. Runs: A_seed* (6 pairs each) and every B_* run (finished or still running); the smoke
run is skipped. Output (default /private/groups/ioannidislab/UQ_decoupling_exp/audits/export/):

    val.csv        full-bank validation, one row per run x pair x validation step x head x slice
    train.csv      training curves in 250-step windows, one row per run x pair x window
    reference.csv  the bank's reference lines, one row per head x slice
    runs.csv       one row per run x pair: configuration, status, best val MSE per head, cost per stage
    alphas.csv     A2's layer weights at the end of each stage, one row per run x pair x stage x head x layer
    README.md      the data dictionary

wandb's scan_history(keys=...) returns only rows holding EVERY requested key, so keys are fetched in the groups that
are logged together: validation (every 5,000 stage-1 steps + stage ends), training (every step), mini-bank (every 250
steps; A2's alphas ride on it).

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
from iltm2.prior._bank import BANK_DIR, load_manifest
from iltm2.prior._split import QUERY_TYPES
from iltm2.train._config import ROOT, RUNS_ROOT, STAGE_STEPS
from iltm2.train._metrics import HEADS, N_BINS

ENTITY_PROJECT = "poletea10-universitat-polit-cnica-de-catalunya/iLTMv2"
SLICES = ["all", *QUERY_TYPES.values(), *(n for n, _, _ in N_BINS)]
WINDOW = 250
STAGE_END = np.cumsum([STAGE_STEPS[s] for s in (1, 2, 3)])          # global steps 50,000 / 54,000 / 55,000
N_TAPS = 12


def stage_of(gstep: int) -> int:
    return int(np.searchsorted(STAGE_END, gstep, side="left")) + 1


def run_config(run_dir: Path):
    """[(pair id, arm, head_lr, fork_lr)] and the seed of a run folder."""
    name = run_dir.name
    if name.startswith("A_seed"):
        pairs = sorted(json.loads((run_dir / "best_index.json").read_text()))
        out = [(p, p.split("_lr")[0], float(p.split("_lr")[1]), np.nan) for p in pairs]
        return out, int(name[len("A_seed"):])
    m = re.fullmatch(r"B_h([^_]+)_f([^_]+)_seed(\d+)", name)
    assert m, name
    return [("B", "B", float(m[1]), float(m[2]))], int(m[3])


def scan(run, keys, page_size=5000):
    return pd.DataFrame(list(run.scan_history(keys=keys + ["_step"], page_size=page_size)))


def export_val(run, run_dir, pairs, seed):
    rows = []
    for pid, arm, hlr, flr in pairs:
        keys = [f"{pid}/val/{m}_{h}/{s}" for h in HEADS for s in SLICES for m in ("mse", "xmse", "skill", "mean")]
        keys += [f"{pid}/val/{m}/{s}" for s in SLICES for m in ("ratio_A_eq", "ratio_tA_eq")]
        df = scan(run, keys)
        for _, r in df.iterrows():
            g = int(r["_step"])
            for h in HEADS:
                for s in SLICES:
                    rows.append(dict(run=run_dir.name, pair=pid, arm=arm, head_lr=hlr, fork_lr=flr, seed=seed,
                                     stage=stage_of(g), global_step=g, head=h, slice=s,
                                     mse=r[f"{pid}/val/mse_{h}/{s}"], xmse=r[f"{pid}/val/xmse_{h}/{s}"],
                                     skill=r[f"{pid}/val/skill_{h}/{s}"], mean_pred=r[f"{pid}/val/mean_{h}/{s}"],
                                     ratio_A_eq=r[f"{pid}/val/ratio_A_eq/{s}"] if h == "A" else np.nan,
                                     ratio_tA_eq=r[f"{pid}/val/ratio_tA_eq/{s}"] if h == "A" else np.nan))
    return rows


def export_train(run, run_dir, pairs, seed):
    """250-step windows: EMA at the window's end, gradient norms' mean and max, the mini-bank xMSE logged in it.
    All pairs' training metrics are logged in one call per step, so one scan fetches them (and the run's _runtime)."""
    gkeys = {pid: [f"{pid}/train/grad_norm_A", f"{pid}/train/grad_norm_E"]
             + ([f"{pid}/train/grad_norm_fork", f"{pid}/train/fork_dist_from_init"] if arm == "B" else [])
             for pid, arm, _, _ in pairs}
    keys = [k for pid, _, _, _ in pairs for k in (f"{pid}/train/xmse_A_ema", f"{pid}/train/xmse_E_ema", *gkeys[pid])]
    tr = scan(run, keys + ["_runtime"])
    if tr.empty:
        return [], None
    mini = scan(run, [f"{pid}/mini/xmse_{h}/all" for pid, _, _, _ in pairs for h in HEADS]).set_index("_step")
    tr["window_end"] = (np.ceil(tr["_step"] / WINDOW) * WINDOW).astype(int)
    rows = []
    for w, d in tr.groupby("window_end"):
        last = d.loc[d["_step"].idxmax()]
        hit = mini.loc[mini.index[(mini.index > w - WINDOW) & (mini.index <= w)]] if not mini.empty else mini
        for pid, arm, hlr, flr in pairs:
            row = dict(run=run_dir.name, pair=pid, arm=arm, head_lr=hlr, fork_lr=flr, seed=seed,
                       stage=stage_of(int(w)), window_end_step=int(w), n_steps=len(d),
                       xmse_A_ema=last[f"{pid}/train/xmse_A_ema"], xmse_E_ema=last[f"{pid}/train/xmse_E_ema"])
            for h in HEADS:
                g = d[f"{pid}/train/grad_norm_{h}"]
                row.update({f"grad_norm_{h}_mean": g.mean(), f"grad_norm_{h}_max": g.max()})
            if arm == "B":
                g = d[f"{pid}/train/grad_norm_fork"]
                row.update(grad_norm_fork_mean=g.mean(), grad_norm_fork_max=g.max(),
                           fork_dist_from_init=last[f"{pid}/train/fork_dist_from_init"])
            for h in HEADS:
                row[f"mini_xmse_{h}"] = hit[f"{pid}/mini/xmse_{h}/all"].iloc[-1] if len(hit) else np.nan
            rows.append(row)
    return rows, tr[["_step", "_runtime"]]


def export_alphas(run, run_dir, pairs, seed):
    rows = []
    for pid, arm, hlr, flr in pairs:
        if arm != "A2":
            continue
        keys = [f"{pid}/train/alpha_{h}/L{l}" for h in HEADS for l in range(1, N_TAPS + 1)]
        df = scan(run, keys)
        if df.empty:
            continue
        df["stage"] = df["_step"].map(lambda g: stage_of(int(g)))
        for st, d in df.groupby("stage"):
            last = d.loc[d["_step"].idxmax()]
            for h in HEADS:
                for l in range(1, N_TAPS + 1):
                    rows.append(dict(run=run_dir.name, pair=pid, arm=arm, head_lr=hlr, seed=seed, stage=int(st),
                                     global_step=int(last["_step"]), head=h, layer=l,
                                     alpha=last[f"{pid}/train/alpha_{h}/L{l}"]))
    return rows


def export_reference(bank: Path):
    refs = load_manifest(bank)["reference_lines"]
    rows = []
    for h in HEADS:
        for s in SLICES:
            if f"ref/count/{s}" not in refs:
                continue
            g = lambda k: refs.get(k, np.nan)  # noqa: E731
            rows.append(dict(head=h, slice=s, count=g(f"ref/count/{s}"), mean_target=g(f"ref/mean_t{h}/{s}"),
                             mean_eq=g(f"ref/mean_eq/{s}"), noise_floor=g(f"ref/noise_floor_{h}/{s}"),
                             const_value=g(f"ref/const_value_{h}"), const_mse=g(f"ref/const_mse_{h}/{s}"),
                             const_xmse=g(f"ref/const_xmse_{h}/{s}"), eq_proxy_mse=g(f"ref/eq_proxy_mse_{h}/{s}"),
                             eq_proxy_xmse=g(f"ref/eq_proxy_xmse_{h}/{s}"),
                             eq_proxy_skill=g(f"ref/eq_proxy_skill_{h}/{s}"),
                             ratio_target_eq=g(f"ref/ratio_t{h}_eq/{s}")))
    return rows


def selected_heads():
    """{(run, pair, head): "A1" | "A2" | "B"} from iltm2.inference.BEST."""
    out = {}
    for arm, paths in BEST.items():
        for h, p in zip(HEADS, paths):
            run = Path(p).parents[2].name
            pair = "B" if arm == "B" else Path(p).stem.rsplit("_", 1)[0]
            out[(run, pair, h)] = arm
    return out


def export_runs(run, run_dir, pairs, seed, runtime, selected):
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
                             full_val_s=d.get("full_val_s", np.nan), peak_gpu_gb=d.get("peak_gpu_gb", np.nan),
                             rejection_rate=d.get("rejection_rate", np.nan))
    rows = []
    for pid, arm, hlr, flr in pairs:
        row = dict(run=run_dir.name, pair=pid, arm=arm, head_lr=hlr, fork_lr=flr, seed=seed,
                   wandb_state=run.state if run is not None else "missing",
                   stages_finished=",".join(map(str, finished)))
        for h in HEADS:
            b = best.get(pid, {}).get(h, {})
            row.update({f"best_val_mse_{h}": b.get("val_mse", np.nan), f"best_stage_{h}": b.get("stage", np.nan),
                        f"best_global_step_{h}": b.get("global_step", np.nan),
                        f"selected_{h}": selected.get((run_dir.name, pid, h), "")})
        for s, c in stage_cost.items():
            row.update({f"s{s}_{k}": v for k, v in c.items()})
        rows.append(row)
    return rows


README = """# Phase 1 runs: exported metrics (iLTMv2, experiment1_uq_heads)

Two scalar heads on a frozen TabICLv2 regressor: Â (aleatoric, target t_A = CRPS entropy from R replicates) and
Ê_dir (epistemic, target t_E = the forecast's mean CRPS minus t_A). Arms: A1 = MLP trunk on h* (the final-LN output
the quantile decoder reads); A2 = learned softmax mix of the 12 per-layer-normalized residual taps, then the trunk;
B = trainable copy (fork) of the last 2 transformer blocks + LN feeding the heads. A jobs: 3 seeds x {A1, A2} x head LR
{1e-4, 3e-4, 1e-3}; B: 9 configurations (head LR x fork LR {3e-6, 1e-5, 3e-5}) on seed 0. Seeds fix the task stream
and the initialization, so runs with the same seed see the same tasks.

Training: 3-stage curriculum, global steps 1-50,000 (stage 1, N = 1,024 rows), 50,001-54,000 (stage 2, N 400-10,240),
54,001-55,000 (stage 3, N 400-60,000); batch 32 tasks; AdamW, cosine schedule restarted per stage, max LR = swept LR
/ 1, / 8, / 40 per stage. Every training task is fresh (no epochs). Validation: a fixed bank of 2,000 tasks (3.1M
queries; 50/30/20 % small/medium/large N; 128 replicates per query), every 5,000 steps in stage 1 and at each stage
end. Selection: per arm and head, the LR with the lowest seed-mean pooled val MSE, then its best seed (B: one
configuration as a whole); the selected heads are marked in runs.csv.

Definitions:
- mse: mean (prediction - target)^2 against the 128-replicate target (includes the targets' replicate noise).
- xmse: mean (prediction - t^(a)) (prediction - t^(b)) with the targets of the two halves of the replicates: the error
  against the realized task with the replicate noise removed (unbiased; single values can be negative).
- skill = 1 - xmse / xmse of the constant baseline (the bank-wide mean target, where every head starts); 1 = perfect.
- Slices: all; query types iid (random queries), hole (queries inside a carved-out region with no context), extrapolation
  (queries in a tail beyond the context), duplicate (queries whose x copies a context row); context-size bins n_le512,
  n_512-1K, n_1K-4K, n_4K-16K, n_16K-60K, by the task's number of CONTEXT rows. Pooled values are query-weighted, and
  over half of the bank's queries are in the 16K-60K bin.
- e(q): the CRPS entropy of the backbone's own forecast. The e(q) proxy predicts the target from e(q) alone (50 bins,
  cross-fitted on task halves): its skill is what the backbone's spread gives for free.

Files:
- val.csv: run, pair (A: e.g. A2_lr1e-3; B: "B"), arm, head_lr, fork_lr (B only), seed, stage, global_step, head (A/E),
  slice, mse, xmse, skill, mean_pred; ratio_A_eq / ratio_tA_eq (head A only): mean of Â / e(q) and of t_A / e(q).
- train.csv: per 250-step window (window_end_step): xmse_A_ema, xmse_E_ema (exponential moving average, smoothing 0.99,
  of the per-step training xMSE, computed before each optimizer step on unseen tasks; reset at each stage start);
  grad_norm_{A,E}_{mean,max} (pre-clipping, clip at 5.0); B also grad_norm_fork_{mean,max}, fork_dist_from_init;
  mini_xmse_{A,E}: xMSE on a fixed 64-task mini-bank (all N = 1,024, the stage-1 shape), logged every 250 steps.
- reference.csv: per head and slice: count (queries), mean_target, mean_eq, noise_floor (the replicate-noise part of
  mse), const_value, const_mse / const_xmse (constant baseline), eq_proxy_mse / eq_proxy_xmse / eq_proxy_skill,
  ratio_target_eq.
- runs.csv: per run and pair: wandb_state, stages_finished, best_val_mse_{A,E} with best_stage / best_global_step,
  selected_{A,E} (the arm whose selected checkpoint it is), and per stage s{1,2,3}_: hours (wall time), median_step_s,
  full_val_s (one full-bank validation), peak_gpu_gb, rejection_rate (prior tasks rejected by the filters).
- alphas.csv: A2 pairs: softmax layer weight alpha per head and tapped layer (1-12) at each stage's last log. A training
  diagnostic, not a localisation: per-layer LN gains can offset them, and the residual stream is cumulative.
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
    run_dirs = sorted(p for p in args.runs_root.iterdir()
                      if p.is_dir() and (p.name.startswith("A_seed") or p.name.startswith("B_")))
    val, train, alphas, runs = [], [], [], []
    for rd in run_dirs:
        pairs, seed = run_config(rd)
        try:
            run = api.run(f"{args.project}/{rd.name}")
        except Exception as e:                                   # noqa: BLE001
            print(f"{rd.name}: no wandb run ({type(e).__name__}); local files only", flush=True)
            runs += export_runs(None, rd, pairs, seed, None, selected)
            continue
        v = export_val(run, rd, pairs, seed)
        t, runtime = export_train(run, rd, pairs, seed)
        a = export_alphas(run, rd, pairs, seed)
        val += v
        train += t
        alphas += a
        runs += export_runs(run, rd, pairs, seed, runtime, selected)
        print(f"{rd.name} ({run.state}): {len(pairs)} pairs; {len(v)} val rows, {len(t)} train rows, {len(a)} alpha "
              f"rows", flush=True)
    frames = dict(val=pd.DataFrame(val), train=pd.DataFrame(train), reference=pd.DataFrame(export_reference(args.bank)),
                  runs=pd.DataFrame(runs), alphas=pd.DataFrame(alphas))
    for name, df in frames.items():
        path = args.out / f"{name}.csv"
        df.to_csv(path, index=False, float_format="%.6g")
        print(f"wrote {path}: {len(df)} rows x {df.shape[1]} columns ({path.stat().st_size / 1e6:.2f} MB)", flush=True)
    (args.out / "README.md").write_text(README)
    print(f"wrote {args.out / 'README.md'}", flush=True)


if __name__ == "__main__":
    main()
