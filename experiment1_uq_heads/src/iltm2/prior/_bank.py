"""The validation bank: 2,000 fixed observed-cut tasks with precomputed targets (design Training §3, protocol §2,
§3 step 6, §7.4-7.5, §8.1).

WHAT IT IS
    A fixed set of tasks every run is validated on, so every comparison between checkpoints, runs, arms and seeds is
    paired (same tasks). Same pipeline as training (`sample_uq_task`: observed-cut prior, filters, split with holes,
    extrapolation and duplicates), with R_val = 128 replicates per query. Stratified by N (design Training §3):
        small   1,000 tasks  N = 1,024                    context 30-90%   (stage-1 shapes)
        medium    600 tasks  N log-uniform 1,024-10,240   context ~80%     (stage 3's context rule)
        large     400 tasks  N log-uniform 10,240-60,000  context ~80%
    d follows TabICLv2's feature sampler for the task's N. The bank has its own seed (VAL_BANK_SEED), disjoint from
    the training seeds {0, 1, 2}; the T2 test bank (and T1, T3, T4) must use other seeds. The mini-bank (§7.4) is
    the first 64 small tasks, stored again on its own so it loads in a second.

WHAT IS STORED PER QUERY (`BankQueries`)
    The PPD is frozen in every arm, so everything that depends only on the task and q_theta is computed once, here
    (protocol §1.3, "precomputed bank targets"): query_type, n bin (of the task's context size n_ctx), e(q_theta),
    t_A and t_E from all 128 replicates, the half-split targets from replicates 1-64 and 65-128 (xMSE and the noise
    floor), and score = mean_r CRPS(q_theta, y^(r)) = t_A + t_E (the forecast's expected CRPS; bank check 2). The
    forecast is the forward route of design Architecture §1 (`FrozenBackbone`). Validation then costs a backbone
    forward (for the heads' inputs) plus the heads; no target is ever recomputed during training.

    Per task (`tasks` in the shards) the model's input is stored too: X (N, d) context rows first, y_ctx, the
    replicates Y (m, 128) (to audit or recompute targets; never needed by the trainer), query_type, and the task's
    key (seed, bin, index) from which `sample_uq_task` regenerates it bit for bit.

LAYOUT (under BANK_DIR = /private/groups/ioannidislab/UQ_decoupling_exp/val_bank, protocol §2)
    shards/<bin>_<k:03d>.pt   {"tasks": [task record, ...], "queries": BankQueries as a dict}; written atomically,
                              so an interrupted build resumes at the first missing shard
    queries.pt                the whole bank's query table (what validation and the reference lines read)
    mini/tasks.pt, mini/queries.pt   the 64-task mini-bank
    manifest.json             seed, sizes, config, git hashes, backbone sha256, stream statistics, and the
                              reference lines of §8.1 for the bank and the mini-bank (wandb-ready flat dicts)
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch

from tabicl.prior._dataset import Prior

from iltm2.model._entropy import crps
from iltm2.prior._cut_graph_scm import CutGraphConfig
from iltm2.prior._dataset import STREAM_SHAPE, sample_task_shape, sample_uq_task, single_thread, task_rng
from iltm2.prior._filter import RejectionLog
from iltm2.train._metrics import n_bin_index
from iltm2.train._targets import half_split_targets, t_aleatoric

BANK_DIR = Path("/private/groups/ioannidislab/UQ_decoupling_exp/val_bank")
VAL_BANK_SEED = 20261103            # chosen once (design Training §3: any seed disjoint from training and test banks)
R_VAL = 128
MINI_SIZE = 64

# name -> (number of tasks, N range, log-uniform N, the stage whose context-fraction rule applies, shard size)
BINS = {
    "small": dict(n_tasks=1000, N=(1024, 1024), log=False, ctx_stage=1, shard=100),
    "medium": dict(n_tasks=600, N=(1024, 10240), log=True, ctx_stage=3, shard=50),
    "large": dict(n_tasks=400, N=(10240, 60000), log=True, ctx_stage=3, shard=20),
}
BIN_NAMES = tuple(BINS)


def task_offsets(n_tasks: Dict[str, int]) -> Dict[str, int]:
    """Global task id of each bin's first task (bins in BIN_NAMES order)."""
    out, acc = {}, 0
    for b in BIN_NAMES:
        out[b], acc = acc, acc + n_tasks[b]
    return out


# ------------------------------------------------------------------------------------------- one task (CPU)


def bank_task_shape(seed: int, bin_name: str, index: int, cfg: CutGraphConfig):
    """(ctx_stage, N, n_ctx, d) of bank task (bin, index), under task_rng(seed, bin index, index, STREAM_SHAPE)."""
    spec = BINS[bin_name]
    with task_rng(seed, BIN_NAMES.index(bin_name), index, STREAM_SHAPE):
        lo, hi = spec["N"]
        N = lo if lo == hi else int(Prior.sample_seq_len(lo, hi, log=spec["log"]))
        s = sample_task_shape(spec["ctx_stage"], cfg, N=N)
    return spec["ctx_stage"], s.N, s.n_ctx, s.d


def generate_bank_task(seed: int, bin_name: str, index: int, cfg: Optional[CutGraphConfig] = None,
                       R: int = R_VAL) -> dict:
    """The CPU part of one bank task: `sample_uq_task` with the bank's shape and R replicates (pipeline steps 1-10).
    Pure function of (seed, bin, index), single-threaded as in the training stream's workers."""
    cfg = cfg or CutGraphConfig()
    with single_thread():
        stage, N, n_ctx, d = bank_task_shape(seed, bin_name, index, cfg)
        log = RejectionLog()
        t, _ = sample_uq_task(seed, BIN_NAMES.index(bin_name), index, stage, cfg, R=R, log=log,
                              N=N, n_ctx=n_ctx, d=d)
    return dict(bin=bin_name, index=index, key=(seed, BIN_NAMES.index(bin_name), index), N=N, n_ctx=n_ctx,
                d=int(t.task.X.shape[1]), d_drawn=d, m=N - n_ctx, X=t.X_model.float().contiguous(),
                y_ctx=t.y_ctx.float().contiguous(), Y=t.Y.float().contiguous(),
                query_type=t.split.query_type.to(torch.int8), deterministic=t.deterministic, redraws=t.redraws,
                accepted=log.accepted, rejected=dict(log.rejected), graph_resamples=log.graph_resamples,
                diagnostics=dict(t.task.diagnostics))


# ------------------------------------------------------------------------------------------- forecast targets (GPU)


@torch.no_grad()
def forecast_targets(backbone, X: torch.Tensor, y_ctx: torch.Tensor, Y: torch.Tensor) -> Dict[str, torch.Tensor]:
    """e(q_theta), t_A, t_E, the half-split targets and the score of one task's queries, from one forward of the
    frozen backbone (forward route of design Architecture §1). All computed in float64, stored as float32."""
    o = backbone(X[None], y_ctx[None])
    q = o.q[0].double()                                     # (m, K)
    Yd = Y.to(q.device).double()                            # (m, R)
    t_A = t_aleatoric(Yd)
    score = crps(q, Yd).mean(dim=-1)
    t_E = score - t_A                                       # = _targets.t_epistemic(q, Yd, t_A), sharing score
    (tAa, tEa), (tAb, tEb) = half_split_targets(q, Yd)
    out = dict(e_q=o.e_q[0], t_A=t_A, t_E=t_E, t_A_a=tAa, t_A_b=tAb, t_E_a=tEa, t_E_b=tEb, score=score)
    return {k: v.float().cpu() for k, v in out.items()}


# ------------------------------------------------------------------------------------------- the query table


@dataclass
class BankQueries:
    """One row per bank query (flat over tasks). int8 codes, float32 values; metrics cast to float64."""
    task_id: torch.Tensor          # (Q,) long, global task id (small, then medium, then large)
    bin_idx: torch.Tensor          # (Q,) int8, index in BIN_NAMES
    query_type: torch.Tensor       # (Q,) int8, IID / HOLE / EXT / DUP
    n_bin: torch.Tensor            # (Q,) int8, index in iltm2.train._metrics.N_BINS (of the task's n_ctx)
    e_q: torch.Tensor
    t_A: torch.Tensor
    t_E: torch.Tensor
    t_A_a: torch.Tensor
    t_A_b: torch.Tensor
    t_E_a: torch.Tensor
    t_E_b: torch.Tensor
    score: torch.Tensor

    def __len__(self) -> int:
        return len(self.task_id)

    @classmethod
    def from_task(cls, rec: dict, task_id: int) -> "BankQueries":
        m = rec["m"]
        return cls(task_id=torch.full((m,), task_id, dtype=torch.long),
                   bin_idx=torch.full((m,), BIN_NAMES.index(rec["bin"]), dtype=torch.int8),
                   query_type=rec["query_type"].to(torch.int8),
                   n_bin=torch.full((m,), n_bin_index(rec["n_ctx"]), dtype=torch.int8),
                   **{k: rec[k] for k in ("e_q", "t_A", "t_E", "t_A_a", "t_A_b", "t_E_a", "t_E_b", "score")})

    @classmethod
    def cat(cls, parts: List["BankQueries"]) -> "BankQueries":
        return cls(**{f.name: torch.cat([getattr(p, f.name) for p in parts]) for f in fields(cls)})

    def subset(self, mask: torch.Tensor) -> "BankQueries":
        return BankQueries(**{f.name: getattr(self, f.name)[mask] for f in fields(self)})

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: dict) -> "BankQueries":
        return cls(**{f.name: d[f.name] for f in fields(cls)})


# ------------------------------------------------------------------------------------------- storage


def atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def shard_path(root: Path, bin_name: str, k: int) -> Path:
    return Path(root) / "shards" / f"{bin_name}_{k:03d}.pt"


def shard_ranges(n_tasks: Dict[str, int]) -> List[tuple]:
    """[(bin, shard k, first index, last index + 1)] in build order."""
    out = []
    for b in BIN_NAMES:
        size = BINS[b]["shard"]
        for k, lo in enumerate(range(0, n_tasks[b], size)):
            out.append((b, k, lo, min(lo + size, n_tasks[b])))
    return out


def load_manifest(root: Path = BANK_DIR) -> dict:
    return json.loads((Path(root) / "manifest.json").read_text())


def load_queries(root: Path = BANK_DIR, mini: bool = False) -> BankQueries:
    path = Path(root) / ("mini" if mini else "") / "queries.pt"
    return BankQueries.from_dict(torch.load(path, weights_only=False))


def iter_tasks(root: Path = BANK_DIR, mini: bool = False) -> Iterator[dict]:
    """Task records in bank order (shard by shard, so the whole bank never sits in memory). Each record has the
    model input (X, y_ctx), the task's query rows in the global query table are those with its `task_id`."""
    root = Path(root)
    if mini:
        yield from torch.load(root / "mini" / "tasks.pt", weights_only=False)
        return
    man = load_manifest(root)
    for b, k, _, _ in shard_ranges(man["n_tasks"]):
        yield from torch.load(shard_path(root, b, k), weights_only=False)["tasks"]


def config_dict(cfg: CutGraphConfig) -> dict:
    return json.loads(json.dumps(asdict(cfg), default=str))


def stream_stats(records: List[dict], Q: BankQueries) -> dict:
    """Rejections, redraws, deterministic share and the t_A distribution of a set of tasks (proposal §4.3: log the
    empirical distribution of the aleatoric target, incl. its zero mass)."""
    acc = sum(r["accepted"] for r in records)
    rejected: Dict[str, int] = {}
    for r in records:
        for k, v in r["rejected"].items():
            rejected[k] = rejected.get(k, 0) + v
    n_rej = sum(rejected.values())
    tA = Q.t_A.double().numpy()
    return dict(tasks=len(records), attempts=acc + n_rej, rejection_rate=n_rej / max(1, acc + n_rej),
                rejected=rejected, graph_resamples=sum(r["graph_resamples"] for r in records),
                redraws=sum(r["redraws"] for r in records),
                deterministic_share=float(np.mean([r["deterministic"] for r in records])),
                tA_zero_mass=float(np.mean(tA == 0)),
                tA_quantiles={str(p): float(np.quantile(tA, p)) for p in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)},
                tasks_tA_all_zero=float(np.mean([r["tA_all_zero"] for r in records])))


def task_meta(rec: dict) -> dict:
    """The tensor-free part of a task record (kept in each shard for the manifest's stream statistics)."""
    keys = ("bin", "index", "key", "N", "n_ctx", "d", "d_drawn", "m", "deterministic", "redraws", "accepted",
            "rejected", "graph_resamples", "diagnostics")
    return {**{k: rec[k] for k in keys}, "tA_all_zero": bool((rec["t_A"] == 0).all())}
