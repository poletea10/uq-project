"""Shared trainer skeleton of Phase 1 (phase1_protocol.md §1.3, §2, §7; design Training §2-3, §5).

`BaseTrainer.run()` trains ONE stage of one job. The arm-specific trainers (`frozen.FrozenTrainer` for the A jobs,
`fork.ForkTrainer` for B in step 9) supply the model, the optimizers and the step; this class owns the stage logic,
copied from tabicl's `Trainer.train` loop (Timer, prior_time / train_time, latest + permanent checkpoints, the
only-load-model restart of scripts/train_v2_reg_stage2.sh) without its DDP / AMP / Muon machinery.

FILES (protocol §2), under <runs_root>/<run_name>/
    stage<s>/latest.pt      full state (heads, optimizers, schedulers, EMAs, best tracker, step), overwritten
                            every --save_temp_every steps and after every full validation
    stage<s>/final.pt       written ONCE at the end of stage s, never overwritten (a second write is an error):
                            the stage's result, kept when the next stage runs
    stage<s>/best/<id>_<h>.pt   the best head h of pair <id> so far (lowest pooled full-bank validation MSE),
                            written in the stage where it was reached
    best_index.json         where each head's current best lives (stage, global step, val MSE, path)
    stage<s>/trace.jsonl    one line per step: stage step, global step, LR per pair, batch fingerprint, launch id
    stage<s>/summary.json   timings (prior wait, train, validation), peak GPU memory, projected stage wall time

HOW A STAGE STARTS (so a relaunched job always does the right thing)
    1. stage<s>/final.pt exists      -> the stage is done: exit 0 at once (a resubmitted job skips it)
    2. stage<s>/latest.pt exists     -> resume: full state, the task stream restarts at the saved step
    3. s > 1                         -> new stage from stage<s-1>/final.pt: the heads' weights and the best tracker
                                        only; optimizers and schedulers start fresh (tabicl's --only_load_model):
                                        new max LR, 1% warmup, one cosine cycle (design Training §2)
    4. s = 1                         -> fresh heads; output biases from the first batch's mean targets (design §1)
                                        or the bank's constants (--init_bias bank)
    Global (wandb) step = previous stage's final global step + stage step, so curves are continuous (§7.1).

PER STEP: train_step on the step's micro-batch groups (the task stream of `iltm2.prior._dataset`, delivered out of
order and put back in step order), then: wandb log every --log_every steps; mini-bank every --mini_every steps;
full bank every --val_every steps and always at the stage end (best-per-head checkpoints); latest.pt every
--save_temp_every steps. --stop_after n stops the launch after n steps with latest.pt saved, as a killed job would
(the smoke run's resume test).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple

import torch
from tabicl.train._optim import get_scheduler       # noqa: F401  (re-exported for the trainers)
from tabicl.train._run import Timer

from iltm2.prior._bank import atomic_save, load_manifest
from iltm2.prior._dataset import CutGraphPrior, UQBatch, iterate_steps, make_dataloader
from iltm2.train._config import STAGE_STEPS
from iltm2.train._validate import BankEvaluator, BestTracker

REPO = Path(__file__).resolve().parents[4]            # UQ_project


def git_state(path) -> dict:
    try:
        head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True).strip())
        return dict(commit=head, dirty=dirty)
    except Exception as e:                                   # noqa: BLE001
        return dict(error=str(e))


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def batch_fingerprint(g: UQBatch) -> str:
    """Short hash of a group's context labels and replicates: the same step gives the same fingerprint in every
    launch (used by the smoke run's resume check)."""
    h = hashlib.sha256()
    for t in (g.y_ctx, g.y_reps):
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


class StageFiles:
    def __init__(self, runs_root, run_name: str, stage: int):
        self.run = Path(runs_root) / run_name
        self.stage = stage
        self.dir = self.run / f"stage{stage}"
        self.latest, self.final = self.dir / "latest.pt", self.dir / "final.pt"
        self.best_dir, self.trace, self.summary = self.dir / "best", self.dir / "trace.jsonl", self.dir / "summary.json"
        self.best_index = self.run / "best_index.json"

    def final_of(self, stage: int) -> Path:
        return self.run / f"stage{stage}" / "final.pt"


class BaseTrainer:
    """One stage of one job. Subclasses implement the hooks below; `run()` is the stage."""

    arm = "A"

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.files = StageFiles(args.runs_root, args.run_name, args.stage)
        self.best = BestTracker()
        self.step = 0                      # completed steps of this stage
        self.global_offset = 0             # global step at the start of this stage
        self.init_from: Optional[str] = None
        self.wandb = None
        self.launch_id = f"{int(time.time())}-{os.getpid()}"
        self.timing: Dict[str, List[float]] = dict(prior=[], train=[], val_full=[], val_mini=[])
        self.rejections = dict(accepted=0, rejected=0)

    # ----- hooks (subclasses)

    def build(self):
        """Backbone, trainable modules, optimizers, schedulers. Sets self.backbone."""
        raise NotImplementedError

    def heads_state(self) -> dict:
        raise NotImplementedError

    def load_heads_state(self, d: dict):
        raise NotImplementedError

    def optim_state(self) -> dict:
        raise NotImplementedError

    def load_optim_state(self, d: dict):
        raise NotImplementedError

    def extra_state(self) -> dict:
        return {}

    def load_extra_state(self, d: dict):
        pass

    def init_output_bias(self, mean_t_A: float, mean_t_E: float):
        raise NotImplementedError

    def warmup_means(self, groups: List[UQBatch]) -> Tuple[float, float]:
        raise NotImplementedError

    def train_step(self, groups: List[UQBatch]) -> Dict[str, float]:
        raise NotImplementedError

    def build_evaluators(self) -> Tuple[BankEvaluator, BankEvaluator]:
        raise NotImplementedError

    def evaluate(self, ev: BankEvaluator, best: Optional[BestTracker], step: int) -> Dict[str, float]:
        raise NotImplementedError

    def mini_extra_logs(self) -> Dict[str, float]:
        return {}

    def best_payload(self, head_id: str, h: str) -> dict:
        raise NotImplementedError

    def overfit(self) -> bool:
        raise NotImplementedError

    # ----- checkpoints

    def _state(self, kind: str) -> dict:
        return dict(kind=kind, arm=self.arm, stage=self.args.stage, step=self.step,
                    global_offset=self.global_offset, global_step=self.global_offset + self.step,
                    heads=self.heads_state(), optim=self.optim_state(), extra=self.extra_state(),
                    best=self.best.state_dict(), config=vars(self.args), backbone_sha256=self.backbone.sha256,
                    wandb_id=self.args.wandb_id, init_from=self.init_from, saved=time.time())

    def _load(self, path: Path) -> dict:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert ck["backbone_sha256"] == self.backbone.sha256, f"{path} was trained on another backbone"
        assert ck["arm"] == self.arm, f"{path} is an arm-{ck['arm']} checkpoint"
        return ck

    def save_latest(self):
        atomic_save(self._state("latest"), self.files.latest)

    def save_final(self):
        if self.files.final.exists():
            raise RuntimeError(f"{self.files.final} exists: a stage's final checkpoint is never overwritten")
        atomic_save(self._state("final"), self.files.final)

    def save_best(self, head_id: str, h: str, gstep: int):
        val, _ = self.best.best[head_id][h]
        self.files.best_dir.mkdir(parents=True, exist_ok=True)
        path = self.files.best_dir / f"{head_id}_{h}.pt"
        atomic_save(dict(self.best_payload(head_id, h), stage=self.args.stage, step=self.step, global_step=gstep,
                         val_mse=val, backbone_sha256=self.backbone.sha256, config=vars(self.args)), path)
        idx = json.loads(self.files.best_index.read_text()) if self.files.best_index.exists() else {}
        idx.setdefault(head_id, {})[h] = dict(stage=self.args.stage, global_step=gstep, val_mse=val, path=str(path))
        tmp = self.files.best_index.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(idx, indent=2))
        os.replace(tmp, self.files.best_index)

    # ----- logging

    def _init_wandb(self):
        a = self.args
        if not a.wandb_log or a.wandb_mode == "disabled":
            return
        import wandb
        os.makedirs(a.wandb_dir, exist_ok=True)
        cfg = dict(vars(a), git=dict(UQ_project=git_state(REPO), tabicl=git_state(REPO / "external" / "tabicl")),
                   backbone_sha256=self.backbone.sha256)
        self.wandb = wandb.init(project=a.wandb_project, entity=a.wandb_entity, name=a.wandb_name, id=a.wandb_id,
                                group=a.wandb_group, tags=a.wandb_tags, resume="allow", dir=a.wandb_dir,
                                mode=a.wandb_mode, config=cfg)

    def log(self, d: Dict[str, float], gstep: int):
        if self.wandb is not None:
            self.wandb.log(d, step=gstep)

    def _trace(self, rec: dict):
        with open(self.files.trace, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def lrs(self) -> Dict[str, float]:
        return {}

    def summary(self, finished: bool) -> dict:
        a, t = self.args, self.timing
        skip = 5 if len(t["train"]) > 10 else 0           # the first steps include worker start-up
        prior, train = t["prior"][skip:], t["train"][skip:]
        step_s = median([p + q for p, q in zip(prior, train)]) if train else float("nan")
        full = sum(t["val_full"]) / len(t["val_full"]) if t["val_full"] else float("nan")
        mini = sum(t["val_mini"]) / len(t["val_mini"]) if t["val_mini"] else float("nan")
        steps = STAGE_STEPS[a.stage]
        n_full, n_mini = math.ceil(steps / a.val_every), steps // a.mini_every
        proj = steps * step_s + n_full * (0 if math.isnan(full) else full) + n_mini * (0 if math.isnan(mini) else mini)
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if self.device.type == "cuda" else float("nan")
        out = dict(stage=a.stage, finished=finished, steps_done=self.step, max_steps=a.max_steps,
                   global_step=self.global_offset + self.step, launch=self.launch_id, init_from=self.init_from,
                   median_prior_wait_s=median(prior) if prior else float("nan"),
                   median_train_s=median(train) if train else float("nan"), median_step_s=step_s,
                   full_val_s=full, mini_val_s=mini, n_full_val=len(t["val_full"]), peak_gpu_gb=peak,
                   rejection_rate=self.rejections["rejected"] / max(1, sum(self.rejections.values())),
                   projected_hours_at_design_budget=proj / 3600, design_steps=steps,
                   val_bank=a.val_bank, batch_size=a.batch_size, num_workers=a.num_workers)
        if finished and self.files.final.exists():
            out["final_sha256"] = file_sha256(self.files.final)
        return out

    def write_summary(self, finished: bool):
        s = self.summary(finished)
        self.files.summary.write_text(json.dumps(s, indent=2))
        print("summary " + json.dumps(s), flush=True)

    # ----- validation

    def _eval(self, ev: BankEvaluator, gstep: int, full: bool):
        with Timer() as t:
            out = self.evaluate(ev, self.best if full else None, gstep)
        self.timing["val_full" if full else "val_mini"].append(t.elapsed)
        if full:
            for k, v in list(out.items()):
                if "/is_best_" in k and v:
                    self.save_best(k.split("/")[0], k[-1], gstep)
        else:
            out.update(self.mini_extra_logs())
        out[f"time/{'val' if full else 'mini'}_s"] = t.elapsed
        self.log(out, gstep)
        print(f"  [{'val ' if full else 'mini'}] global step {gstep}: {t.elapsed:.1f}s", flush=True)

    # ----- the stage

    def run(self) -> int:
        a = self.args
        torch.manual_seed(a.seed)
        if a.overfit_steps > 0:
            self.build()
            return 0 if self.overfit() else 1
        if self.files.final.exists():
            print(f"{self.files.final} exists: stage {a.stage} of {a.run_name} is complete, nothing to do", flush=True)
            return 0
        self.files.dir.mkdir(parents=True, exist_ok=True)
        self.build()
        init_bias_pending = False
        if self.files.latest.exists():
            ck = self._load(self.files.latest)
            assert ck["stage"] == a.stage
            self.load_heads_state(ck["heads"])
            self.load_optim_state(ck["optim"])
            self.load_extra_state(ck["extra"])
            self.best.load_state_dict(ck["best"])
            self.step, self.global_offset, self.init_from = ck["step"], ck["global_offset"], ck["init_from"]
            print(f"resuming stage {a.stage} of {a.run_name} at step {self.step} (global {self.global_offset + self.step})",
                  flush=True)
        elif a.stage > 1:
            prev = self.files.final_of(a.stage - 1)
            if not prev.exists():
                raise FileNotFoundError(f"stage {a.stage} starts from {prev}, which does not exist (stage {a.stage - 1} "
                                        "not finished)")
            ck = self._load(prev)
            self.load_heads_state(ck["heads"])
            self.best.load_state_dict(ck["best"])
            self.global_offset, self.init_from = ck["global_step"], str(prev)
            print(f"stage {a.stage} of {a.run_name}: heads from {prev} (global step {self.global_offset}); fresh "
                  f"optimizers and schedulers", flush=True)
        else:
            if a.init_bias == "bank":
                refs = load_manifest(a.val_bank)["reference_lines"]
                self.init_output_bias(refs["ref/const_value_A"], refs["ref/const_value_E"])
            else:
                init_bias_pending = True
            print(f"stage 1 of {a.run_name}: fresh heads (seed {a.seed}), output bias from "
                  f"{'the first batch' if init_bias_pending else 'the bank constants'}", flush=True)
        self._init_wandb()
        full_ev, mini_ev = self.build_evaluators()

        if self.step < a.max_steps:
            prior = CutGraphPrior(a.seed, a.stage, a.max_steps, start_step=self.step, R=a.R,
                                  batch_size=a.batch_size)
            loader = make_dataloader(prior, a.num_workers, a.prefetch_factor)
            steps = iterate_steps(loader)
            n_launch = 0
            print(f"{prior!r}; {a.num_workers} workers", flush=True)
            while self.step < a.max_steps:
                with Timer() as tp:
                    step, groups = next(steps)
                assert step == self.step, f"stream at step {step}, trainer at {self.step}"
                for g in groups:
                    self.rejections["accepted"] += g.log.accepted
                    self.rejections["rejected"] += sum(g.log.rejected.values())
                if init_bias_pending:
                    self.init_output_bias(*self.warmup_means(groups))
                    init_bias_pending = False
                fp = batch_fingerprint(groups[0])
                lrs = self.lrs()
                with Timer() as tt:
                    logs = self.train_step(groups)
                self.step += 1
                n_launch += 1
                gstep = self.global_offset + self.step
                self.timing["prior"].append(tp.elapsed)
                self.timing["train"].append(tt.elapsed)
                self._trace(dict(launch=self.launch_id, stage=a.stage, step=step, global_step=gstep, lr=lrs,
                                 fingerprint=fp, wandb_id=a.wandb_id))
                last = self.step == a.max_steps
                if self.step % a.log_every == 0 or last:
                    logs.update({"stage": a.stage, "prior_time": tp.elapsed, "train_time": tt.elapsed,
                                 "prior/rejection_rate": self.rejections["rejected"] / max(
                                     1, sum(self.rejections.values())),
                                 "prior/deterministic_share": float(torch.cat([g.deterministic for g in groups])
                                                                    .float().mean())})
                    self.log(logs, gstep)
                if self.step % 100 == 0 or last or self.step <= 3:
                    print(f"stage {a.stage} step {self.step}/{a.max_steps} (global {gstep}): prior wait "
                          f"{tp.elapsed:.2f}s, train {tt.elapsed:.2f}s", flush=True)
                if self.step % a.mini_every == 0 or last:
                    self._eval(mini_ev, gstep, full=False)
                if self.step % a.val_every == 0 or last:
                    self._eval(full_ev, gstep, full=True)
                    self.save_latest()
                elif self.step % a.save_temp_every == 0:
                    self.save_latest()
                if a.stop_after is not None and n_launch >= a.stop_after and not last:
                    self.save_latest()
                    self.write_summary(finished=False)
                    print(f"stopped after {n_launch} steps of this launch (--stop_after); latest.pt at step "
                          f"{self.step}", flush=True)
                    return 0
            del steps, loader
        self.save_final()
        self.write_summary(finished=True)
        if self.wandb is not None:
            self.wandb.finish()
        print(f"stage {a.stage} of {a.run_name} complete: {self.files.final}", flush=True)
        return 0
