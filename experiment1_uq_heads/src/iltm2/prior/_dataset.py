"""Task-level sampling of the observed-cut prior: RNG, task shapes and one accepted task (design Training §1, protocol §1.3).

This module is the dataset side of the prior. It holds what turns (seed, step, task_idx) into one task:
      task_rng            deterministic global RNG per (seed, step, task_idx, stream)
      STAGES, TaskShape,  pipeline steps 1-3: stage sizes (N, n_ctx, d), query budgets (m_ext, m_hole, m_iid),
        sample_task_shape region sizes (T, k) and the pool size P
      sample_task         shape + `CutGraphSCM.sample` (pipeline steps 4-7, in `_cut_graph_scm.py`) under task_rng
    Step 4 of the order of work (protocol §3) adds the training iterator here (`CutGraphPrior` -> `UQBatch`,
    mirroring tabicl's `GraphPrior.get_batch`), on top of `sample_task` and the split of `_split.py`.

DETERMINISM
    graph_lib draws from the global numpy and torch RNGs, so `task_rng` seeds them from a `SeedSequence` of
    (seed, step, task_idx, stream) and restores them afterwards. A task is then a pure function of those integers,
    whatever process or DataLoader worker builds it: A job s and the B jobs on seed s see the same tasks, and a
    resumed run continues on exactly the tasks it would have seen (design Training §5, protocol §1.3). Streams keep
    the stages of a task independent of each other (task, split, replicates).

TASK SHAPES
    Every task has exactly TabICLv2's sizes for the current stage (scripts/train_v2_reg_stage{1,2,3}.sh): N rows
    the model sees, n_ctx of them context. Hole and extrapolation queries are carved out of regions whose other
    rows are discarded so that the context never sees them; to keep n_ctx and m = N - n_ctx exact, the X-graph
    generates an enlarged pool of P = N + (T - m_ext) + (k - m_hole) rows: the N kept rows plus exactly the rows
    the split will discard (design Training §1, "Fixed sizes").
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from tabicl.prior._dataset import Prior

from iltm2.prior._cut_graph_scm import CutGraphConfig, CutGraphSCM, CutGraphTask
from iltm2.prior._filter import RejectionLog


# =========================================================================================== determinism

STREAM_TASK, STREAM_SPLIT, STREAM_REPLICATES = 0, 1, 2


@contextlib.contextmanager
def task_rng(seed: int, step: int, task_idx: int, stream: int = STREAM_TASK):
    """Seed the global numpy and torch RNGs from (seed, step, task_idx, stream); restore their state on exit.

    graph_lib draws from the *global* numpy and torch RNGs (and each `Context` seeds its `GlobalSampler` from the
    global numpy RNG), so seeding them makes a task a pure function of the four integers, whatever process or
    DataLoader worker builds it (protocol §1.3). The streams keep stages of a task independent:
      STREAM_TASK        task shape, both graphs, pool and y^(0) (`sample_task`)
      STREAM_SPLIT       hole / tail / i.i.d. split and duplicates (step 3, `_split.py`)
      STREAM_REPLICATES  the R replicate draws (`CutGraphTask.replicates`)
    Bit-identical results across processes also need the same intra-op thread count (torch reductions may sum in
    a different order); DataLoader workers use 1 thread, and the determinism test sets 1 thread everywhere.
    Python's `set` iteration order (PYTHONHASHSEED) is not involved here: this module never iterates a set.
    """
    np_state, torch_state = np.random.get_state(), torch.get_rng_state()
    s_np, s_torch = np.random.SeedSequence([seed, step, task_idx, stream]).generate_state(2)
    np.random.seed(int(s_np))
    torch.manual_seed(int(s_torch))
    try:
        yield
    finally:
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)


# =========================================================================================== task shape [1-3]

# Row counts and context fractions per stage (scripts/train_v2_reg_stage{1,2,3}.sh; design Training §0, §2).
STAGES = {
    1: dict(min_seq_len=None, max_seq_len=1024, log_seq_len=False, min_train_size=0.3, max_train_size=0.9),
    2: dict(min_seq_len=400, max_seq_len=10240, log_seq_len=True, min_train_size=0.79, max_train_size=0.81),
    3: dict(min_seq_len=400, max_seq_len=60000, log_seq_len=True, min_train_size=0.79, max_train_size=0.81),
}


def _round(x: float) -> int:
    """Round half up (Python's round() rounds half to even)."""
    return int(math.floor(x + 0.5))


@dataclass(frozen=True)
class TaskShape:
    """Sizes of one task (design Training §1, "Fixed sizes" and pipeline steps 1-3).

    The model sees N = n_ctx + m rows. The m queries split into m_ext extrapolation, m_hole hole and m_iid i.i.d.
    queries. The tail (T rows) and the hole (k rows) are carved out of an enlarged pool; their rows that are not
    selected as queries are discarded, so the context never sees those regions. The pool therefore has
    P = N + (T - m_ext) + (k - m_hole) rows: the N kept rows plus exactly the rows that will be discarded.
    """
    stage: int
    N: int
    n_ctx: int
    d: int
    m_ext: int
    m_hole: int
    m_iid: int
    T: int
    k: int

    @property
    def m(self) -> int:
        return self.N - self.n_ctx

    @property
    def P(self) -> int:
        return self.N + (self.T - self.m_ext) + (self.k - self.m_hole)


def _log_uniform_int(lo: int, hi: float) -> int:
    """Integer log-uniform on [lo, hi]; 0 when lo = 0 (no queries of that type, so no region either)."""
    if lo == 0:
        return 0
    hi = max(lo, int(math.floor(hi)))
    if hi == lo:
        return lo
    return int(np.clip(_round(math.exp(np.random.uniform(math.log(lo), math.log(hi)))), lo, hi))


def sample_task_shape(stage: int, cfg: CutGraphConfig, N: Optional[int] = None, n_ctx: Optional[int] = None,
                      d: Optional[int] = None) -> TaskShape:
    """Pipeline steps 1-3. N, n_ctx and d follow TabICLv2's own samplers unless given (the gate matches them).

    1. N: `Prior.sample_seq_len` (stage 1: always 1,024; stages 2-3: log-uniform). n_ctx: `Prior.sample_train_size`
       (TabICLv2 truncates f N to an int). d: uniform on [min_features, `Prior.adjust_max_features(N)`], rounded,
       as `GraphPrior.get_batch` does for each subgroup.
    2. Query budgets from the mixture: m_ext = round(p_extrap m), m_hole = round(p_hole m), m_iid = the rest.
    3. Region sizes: T ~ log-uniform [m_ext, max(m_ext, 0.2 N)], k ~ log-uniform [m_hole, max(m_hole, 0.2 N)].
    """
    st = STAGES[stage]
    if N is None:
        N = int(Prior.sample_seq_len(st["min_seq_len"], st["max_seq_len"], log=st["log_seq_len"]))
    if n_ctx is None:
        n_ctx = int(Prior.sample_train_size(st["min_train_size"], st["max_train_size"], N))
    if d is None:
        d = _round(np.random.uniform(cfg.min_features, Prior.adjust_max_features(N, cfg.max_features)))
    m = N - n_ctx
    _, p_hole, p_ext = cfg.query_mix
    m_ext, m_hole = _round(p_ext * m), _round(p_hole * m)
    m_iid = m - m_ext - m_hole
    assert m_iid >= 0 and 0 < n_ctx < N and d >= 1
    T = _log_uniform_int(m_ext, cfg.region_max_frac * N)
    k = _log_uniform_int(m_hole, cfg.region_max_frac * N)
    return TaskShape(stage, N, n_ctx, d, m_ext, m_hole, m_iid, T, k)


def sample_task(seed: int, step: int, task_idx: int, stage: int, cfg: Optional[CutGraphConfig] = None,
                **shape_overrides) -> Tuple[CutGraphTask, RejectionLog]:
    """One accepted task as a pure function of (seed, step, task_idx): shape (steps 1-3) and task (steps 4-7),
    under `task_rng(..., STREAM_TASK)`. `shape_overrides` (N, n_ctx, d) are passed to `sample_task_shape`."""
    cfg = cfg or CutGraphConfig()
    with task_rng(seed, step, task_idx, STREAM_TASK):
        shape = sample_task_shape(stage, cfg, **shape_overrides)
        return CutGraphSCM(cfg).sample(shape)
