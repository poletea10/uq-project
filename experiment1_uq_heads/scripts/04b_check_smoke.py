"""Checks of the A-trainer smoke run (phase1_protocol.md §6; transfer guide §5 Step 2.7), run at the end of
scripts/04_smoke_A.sh.

The smoke job trains a short version of every stage into <runs_root>/smoke_A and this script reads what it left:

CHECKS
    1. Stage files: stage{1,2,3}/final.pt exist; stage 1's and stage 2's final.pt are byte-identical to what they
       were when their stage ended (sha256 recorded in that stage's summary): a final checkpoint is never
       overwritten by a later stage; best_index.json lists both heads of the 6 pairs (head LR x p), and every listed
       file exists.
    2. Stage chaining: stage 2 starts from stage 1's final.pt and stage 3 from stage 2's; the global step is
       continuous across stages (each stage's first global step = the previous stage's final global step + 1).
    3. Resume (protocol §6, run 3; stage 1 was stopped after 100 steps and relaunched): two launches, the first
       ending at step 99 and the second starting at step 100; every stage step 0..199 trained exactly once; one
       wandb id for every launch of every stage; the LR of every pair at every step equals tabicl's schedule
       (cosine_with_restarts, recomputed here); the batches of steps 100 and 101 (the first two after the restart),
       both passes, are bit-identical to the same steps rebuilt in this process from (seed, stage, step).
    4. Logging of the new design (stage<s>/last_{train,mini,val}.json): stage 1 trains two passes per group (16
       forwards per step) and stages 2-3 one (32); stage 1's training xMSE is logged per n bin, including n_le256
       (the subsampled pass); the full-bank validation of stage 3 (real bank) logs, for every pair and head, mse /
       xmse / skill and their weighted versions wmse / wxmse / wskill on "all", on every query type and on every n
       bin the bank populates; the mini-bank logs every n bin it populates.
    5. Resources and wall time (protocol §6, run 2): per stage, the median prior wait, train time and step time,
       the validation and cache times and the peak GPU memory (gate: < 22 GiB on the 24 GB A5500); then the
       projected wall time of a full A job at the design budget (30,000 / 8,000 / 1,200 steps; full bank every
       5,000 / 4,000 / 600 steps; h_star cached once per stage launch), with the validation and cache times measured
       on the real bank in stage 3 (stages 1-2 of the smoke run validate on the small smoke bank). Gate: the
       projection fits in the --time of train_A.sh.

OUTPUT: one "OK  ..." / "FAIL ..." line per check, the timing table, exit 1 if any check failed.
    python experiment1_uq_heads/scripts/04b_check_smoke.py --run /private/groups/ioannidislab/UQ_decoupling_exp_new/runs/smoke_A
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from iltm2.prior._dataset import CutGraphPrior
from iltm2.prior._split import QUERY_TYPES
from iltm2.train._base import batch_fingerprint, file_sha256, get_scheduler
from iltm2.train._config import STAGE_STEPS, VAL_EVERY
from iltm2.train._metrics import HEADS, N_BIN_NAMES

PEAK_GB_MAX = 22.0
N_PAIRS = 6
METRICS = ("mse", "xmse", "skill", "wmse", "wxmse", "wskill")
FAILS = []


def check(name, ok, detail=""):
    print(f"{'OK  ' if ok else 'FAIL'} {name}  {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def trace(run: Path, stage: int):
    return [json.loads(l) for l in (run / f"stage{stage}" / "trace.jsonl").read_text().splitlines() if l.strip()]


def last(run: Path, stage: int, kind: str) -> dict:
    path = run / f"stage{stage}" / f"last_{kind}.json"
    return json.loads(path.read_text())["logs"] if path.exists() else {}


def lr_of(head_id: str) -> float:
    m = re.fullmatch(r"A_lr([^_]+)_p[^_]+(?:_v\d+)?", head_id)
    assert m, head_id
    return float(m[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--job_days", type=float, default=4.0, help="the --time of train_A.sh, in days")
    args = ap.parse_args()
    run = args.run
    torch.set_num_threads(4)

    # ---- 1. stage files
    print("\n# 1. stage files")
    finals = {s: run / f"stage{s}" / "final.pt" for s in (1, 2, 3)}
    check("stage{1,2,3}/final.pt exist", all(p.exists() for p in finals.values()),
          " ".join(f"stage{s}:{'yes' if p.exists() else 'NO'}" for s, p in finals.items()))
    summ = {s: json.loads((run / f"stage{s}" / "summary.json").read_text()) for s in (1, 2, 3)
            if (run / f"stage{s}" / "summary.json").exists()}
    for s in (1, 2):
        rec = summ.get(s, {}).get("final_sha256")
        now = file_sha256(finals[s]) if finals[s].exists() else None
        check(f"stage {s}'s final.pt unchanged after the later stages (never overwritten)", rec is not None
              and rec == now, f"recorded {str(rec)[:12]}... now {str(now)[:12]}...")
    idx_path = run / "best_index.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    listed = [(hid, h) for hid in idx for h in idx[hid]]
    check(f"best_index.json lists both heads of {N_PAIRS} pairs, and every listed file exists",
          len(idx) == N_PAIRS and len(listed) == 2 * N_PAIRS
          and all(Path(idx[hid][h]["path"]).exists() for hid, h in listed),
          f"{len(idx)} pairs ({', '.join(sorted(idx))}), {len(listed)} heads")

    # ---- 2. chaining
    print("\n# 2. stage chaining")
    cks = {s: torch.load(p, map_location="cpu", weights_only=False) for s, p in finals.items() if p.exists()}
    if len(cks) == 3:
        check("stage 2 starts from stage 1's final.pt, stage 3 from stage 2's",
              cks[2]["init_from"] == str(finals[1]) and cks[3]["init_from"] == str(finals[2]),
              f"{cks[2]['init_from']} | {cks[3]['init_from']}")
        tr = {s: trace(run, s) for s in (1, 2, 3)}
        cont = all(tr[s][0]["global_step"] == cks[s - 1]["global_step"] + 1 for s in (2, 3))
        check("global step continuous across stages", cont,
              " ".join(f"stage{s}: {tr[s][0]['global_step']}..{tr[s][-1]['global_step']}" for s in (1, 2, 3)))
        ids = {r["wandb_id"] for s in (1, 2, 3) for r in tr[s]}
        check("one wandb id for every launch of every stage", len(ids) == 1, str(sorted(ids)))

    # ---- 3. resume
    print("\n# 3. resume (stage 1 stopped after 100 steps, relaunched)")
    t1 = trace(run, 1)
    launches = list(dict.fromkeys(r["launch"] for r in t1))
    steps = [r["step"] for r in t1]
    by_launch = {l: [r["step"] for r in t1 if r["launch"] == l] for l in launches}
    if len(launches) == 2:
        a, b = by_launch[launches[0]], by_launch[launches[1]]
        ok = a[-1] == 99 and b[0] == 100
    else:
        ok = False
    check("two launches: the first ends at step 99, the second starts at step 100", ok,
          " | ".join(f"{l}: {by_launch[l][0]}..{by_launch[l][-1]}" for l in launches))
    n1 = cks[1]["config"]["max_steps"] if 1 in cks else 200
    check(f"every stage-1 step 0..{n1 - 1} trained exactly once", sorted(steps) == list(range(n1)),
          f"{len(steps)} steps, {len(set(steps))} distinct")
    if 1 in cks:
        cfg = SimpleNamespace(**cks[1]["config"])
        worst = 0.0
        for hid in t1[0]["lr"]:
            lr0 = lr_of(hid) / cfg.lr_divisor
            opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=lr0)
            sched = get_scheduler(cfg, opt)
            expect = []
            for _ in range(n1):
                expect.append(opt.param_groups[0]["lr"])
                opt.step()
                sched.step()
            got = {r["step"]: r["lr"][hid] for r in t1}
            worst = max(worst, max(abs(got[s] - expect[s]) / max(expect[s], 1e-12) if expect[s] else abs(got[s])
                                   for s in got))
        check("every pair's LR at every step equals tabicl's schedule (across the restart)", worst < 1e-9,
              f"worst relative diff {worst:.1e}")
        prior = CutGraphPrior(cfg.seed, 1, n1, R=cfg.R, batch_size=cfg.batch_size, prior=cfg.prior,
                              subsample=cfg.subsample)
        fps = {r["step"]: r["fingerprint"] for r in t1}
        same = [batch_fingerprint(prior.group(s, 0)) == fps[s] for s in (100, 101)]
        check("steps 100 and 101 trained on the batches (both passes) rebuilt from (seed, stage, step)", all(same),
              str(same))

    # ---- 4. logging of the new design
    print("\n# 4. passes, slices and weightings in the logs")
    n_pass = {s: last(run, s, "train").get("train/n_passes") for s in (1, 2, 3)}
    check("stage 1 runs 2 passes per group (16 forwards per step), stages 2-3 one (32)",
          n_pass == {1: 16.0, 2: 32.0, 3: 32.0}, str(n_pass))
    tr1 = last(run, 1, "train")
    pairs = sorted(idx) if idx else []
    bins1 = sorted({k.rsplit("/", 1)[-1] for k in tr1 if "/train/xmse_A/" in k})
    check("stage 1: training xMSE logged per n bin for every pair, n_le256 included (the subsampled pass)",
          bool(pairs) and "n_le256" in bins1
          and all(f"{hid}/train/xmse_{h}/{b}" in tr1 for hid in pairs for h in HEADS for b in bins1),
          f"bins {bins1}")
    val3, mini3 = last(run, 3, "val"), last(run, 3, "mini")
    slices = [s for s in ("all", *QUERY_TYPES.values(), *N_BIN_NAMES) if f"ref/count/{s}" in val3]
    missing = [f"{hid}/val/{m}_{h}/{s}" for hid in pairs for h in HEADS for m in METRICS for s in slices
               if f"{hid}/val/{m}_{h}/{s}" not in val3]
    check("stage 3 full bank: mse / xmse / skill and wmse / wxmse / wskill for every pair, head and populated slice",
          bool(pairs) and len(slices) > 1 + len(QUERY_TYPES) and not missing,
          f"{len(slices)} slices ({', '.join(slices)}); {len(missing)} keys missing {missing[:3]}")
    qt_ok = all(f"ref/count/{q}" in val3 for q in QUERY_TYPES.values())
    check("the real bank populates all six query types", qt_ok,
          " ".join(f"{q} {int(val3.get(f'ref/count/{q}', 0))}" for q in QUERY_TYPES.values()))
    mslices = [s for s in ("all", *N_BIN_NAMES) if f"ref_mini/count/{s}" in mini3]
    missing = [f"{hid}/mini/{m}_{h}/{s}" for hid in pairs for h in HEADS for m in ("xmse", "wxmse") for s in mslices
               if f"{hid}/mini/{m}_{h}/{s}" not in mini3]
    check("mini-bank: xmse and wxmse per populated n bin for every pair", len(mslices) > 1 and not missing,
          f"slices {mslices}; {len(missing)} keys missing")

    # ---- 5. resources and wall time
    print("\n# 5. resources and projected wall time")
    print(f"{'stage':>5} {'steps':>6} {'prior wait':>10} {'train':>8} {'step':>8} {'full val':>9} {'mini':>7} "
          f"{'caches':>8} {'peak GiB':>8}  val bank", flush=True)
    for s, d in summ.items():
        print(f"{s:>5} {d['steps_done']:>6} {d['median_prior_wait_s']:>9.2f}s {d['median_train_s']:>7.2f}s "
              f"{d['median_step_s']:>7.2f}s {d['full_val_s']:>8.1f}s {d['mini_val_s']:>6.2f}s "
              f"{d['cache_build_s']:>7.0f}s {d['peak_gpu_gb']:>8.2f}  {Path(d['val_bank']).name}", flush=True)
    peaks = {s: d["peak_gpu_gb"] for s, d in summ.items()}
    check(f"peak GPU memory below {PEAK_GB_MAX} GiB in every stage", len(peaks) == 3 and all(
        v < PEAK_GB_MAX for v in peaks.values()), str({s: round(v, 2) for s, v in peaks.items()}))
    if 3 in summ:
        full_val, cache = summ[3]["full_val_s"], summ[3]["cache_build_s"]      # measured on the real bank
        total = 0.0
        for s, d in summ.items():
            steps_s = STAGE_STEPS[s]
            n_full = math.ceil(steps_s / VAL_EVERY[s])
            hours = (steps_s * d["median_step_s"] + n_full * full_val + (steps_s // 250) * d["mini_val_s"]
                     + cache) / 3600
            total += hours
            print(f"projected stage {s}: {steps_s} steps x {d['median_step_s']:.2f}s + {n_full} full validations x "
                  f"{full_val / 60:.1f} min + h_star cache {cache / 60:.1f} min + mini-bank = {hours:.1f} h",
                  flush=True)
        check(f"a full A job is projected to fit in train_A.sh's time limit ({args.job_days:g} days)",
              total < args.job_days * 24, f"projected {total:.1f} h = {total / 24:.2f} days")

    print(f"\n{'ALL OK' if not FAILS else f'{len(FAILS)} FAILED: ' + '; '.join(FAILS)}", flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
