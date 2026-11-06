"""Task-level sampling of the observed-cut prior: RNG, task shapes, one accepted task, and the training stream
(design Training §1, protocol §1.3, §3 step 4).

This module is the dataset side of the prior. It holds what turns (seed, step, task_idx) into one task, and the
tasks into the batches a trainer consumes:
      task_rng            deterministic global RNG per (seed, step, task_idx, stream)
      STAGES, TaskShape,  pipeline steps 1-3: stage sizes (N, n_ctx, d), query budgets (m_ext, m_hole, m_iid),
        sample_task_shape region sizes (T, k) and the pool size P
      sample_task         shape + `CutGraphSCM.sample` (pipeline steps 4-7, in `_cut_graph_scm.py`) under task_rng
      UQTask, sample_uq_task  one training task: sample_task + split + duplicates + R replicates (steps 1-10)
      UQBatch             one micro-batch group: tasks sharing (N, n_ctx, d), stacked (protocol §1.3)
      CutGraphPrior       map-style Dataset: item i = group i of the stream (step = i // n_groups); mirrors tabicl's
                          `GraphPrior.get_batch` + `PriorDataset`, with a deterministic stream
      LargestFirstSampler, make_dataloader, iterate_steps   dispatch order, the DataLoader (tabicl's settings,
                          out-of-order delivery) and the per-step reorder buffer

THE TRAINING STREAM (step 4)
    tabicl's `GraphPrior.get_batch` draws a step's datasets (64 in TabICLv2; `BATCH_SIZE` = 32 here, design
    Training §2) in groups of `batch_size_per_gp` that share
    (seq_len, train_size, max_features) and, per subgroup, the feature count; the trainer then cuts the batch into
    micro-batches of `micro_batch_size` = the group size (4 in stage 1, 1 in stages 2-3, scripts/
    train_v2_reg_stage*.sh) and requires one (seq_len, train_size) per micro-batch (`validate_micro_batch`). The
    same structure is kept here: a step is `n_groups = batch_size / group_size` groups; group g of step s draws its
    (N, n_ctx, d) once (under task_rng(seed, key, g, STREAM_SHAPE)), and each of its tasks draws everything else
    (T, k, graphs, pool, split, replicates) from its own (seed, key, task_idx) with task_idx = g * group_size + j.
    `key = stage_step_key(stage, step)` puts the stage into the RNG key, so stage 2's step 0 is not stage 1's.
    One DataLoader item is one group: a stage-3 step is 32 independent single-task items, so its tasks are spread
    over the workers instead of one worker building 32 tasks of up to 60K rows in a row. Nothing depends on the
    number of workers, so A job s, the B jobs on seed s and a resumed run all see the same tasks.
    Items are built largest group first and delivered out of order (`make_dataloader`), so that no worker idles
    behind a slow 60K-row task; `iterate_steps` puts them back in stream order (a per-step reorder buffer), so the
    trainer receives exactly the steps an in-order loader would give.

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
import time
from dataclasses import dataclass, field, fields
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import threadpoolctl
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from tabicl.prior._dataset import Prior

from iltm2.prior._cut_graph_scm import CutGraphConfig, CutGraphSCM, CutGraphTask, NonFiniteReplicate
from iltm2.prior._filter import RejectionLog


# =========================================================================================== determinism

# STREAM_SHAPE seeds a micro-batch group's shared (N, n_ctx, d); its task_idx slot holds the group index.
# Redraws after a non-finite replicate use stream + REDRAW_STRIDE * k (as scripts/00f_difficulty_match.py does).
STREAM_TASK, STREAM_SPLIT, STREAM_REPLICATES, STREAM_SHAPE = 0, 1, 2, 3
REDRAW_STRIDE = 100


@contextlib.contextmanager
def task_rng(seed: int, step: int, task_idx: int, stream: int = STREAM_TASK):
    """Seed the global numpy and torch RNGs from (seed, step, task_idx, stream); restore their state on exit.

    graph_lib draws from the *global* numpy and torch RNGs (and each `Context` seeds its `GlobalSampler` from the
    global numpy RNG), so seeding them makes a task a pure function of the four integers, whatever process or
    DataLoader worker builds it (protocol §1.3). The streams keep stages of a task independent:
      STREAM_TASK        task shape, both graphs, pool and y^(0) (`sample_task`)
      STREAM_SPLIT       hole / tail / i.i.d. split and duplicates (step 3, `_split.py`)
      STREAM_REPLICATES  the R replicate draws (`CutGraphTask.replicates`)
      STREAM_SHAPE       a micro-batch group's shared (N, n_ctx, d); task_idx is then the group index
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


# =========================================================================================== one training task [1-10]


@dataclass
class UQTask:
    """One training task after the split and the replicates (pipeline steps 1-10).

    task      the accepted pool (`CutGraphTask`)
    split     its split (`_split.Split`: pool rows of the context and the queries, query types)
    Y         (m, R) g-transformed replicates of the m query slots (fresh U2 draws; duplicates included)
    redraws   how many whole tasks were redrawn because a replicate was non-finite (design §1.4)
    """
    task: CutGraphTask
    split: "Split"  # noqa: F821  (iltm2.prior._split imports this module, so it is imported lazily below)
    Y: torch.Tensor
    redraws: int

    @property
    def X_model(self) -> torch.Tensor:
        """(N, d_kept) the rows the model sees: context first, then the query slots."""
        return self.task.X[self.split.model_rows]

    @property
    def y_ctx(self) -> torch.Tensor:
        """(n_ctx,) the context labels g(y^(0)); query y^(0) are never used past the filter."""
        return self.task.y0[self.split.ctx_rows]

    @property
    def deterministic(self) -> bool:
        """True if no private root reaches y: y is an exact function of X and t_A = 0 at every query."""
        return self.task.target.n_private_roots_used == 0


def sample_uq_task(seed: int, step: int, task_idx: int, stage: int, cfg: Optional[CutGraphConfig] = None,
                   R: int = 16, max_redraws: int = 10, log: Optional[RejectionLog] = None,
                   **shape_overrides) -> Tuple[UQTask, RejectionLog]:
    """Pipeline steps 1-10 for one task, as a pure function of (seed, step, task_idx).

    Attempt k = 0 is exactly `sample_task(seed, step, task_idx, stage, cfg, **shape_overrides)`, then
    `make_split` under STREAM_SPLIT and `replicates(query_rows, R)` under STREAM_REPLICATES. A non-finite replicate
    is only seen here, after acceptance: the task is revoked in the log (reason "non_finite_replicate") and the
    whole task is redrawn (attempt k uses streams + REDRAW_STRIDE * k), with the same (N, n_ctx, d) so the
    micro-batch group keeps one shape (design §1.4: rejected tasks are redrawn entirely, never repaired).
    """
    from iltm2.prior._split import make_split            # lazy: _split imports TaskShape from this module

    cfg = cfg or CutGraphConfig()
    log = log if log is not None else RejectionLog()
    for k in range(max_redraws + 1):
        off = REDRAW_STRIDE * k
        with task_rng(seed, step, task_idx, STREAM_TASK + off):
            shape = sample_task_shape(stage, cfg, **shape_overrides)
            task, _ = CutGraphSCM(cfg).sample(shape, log)
        with task_rng(seed, step, task_idx, STREAM_SPLIT + off):
            split = make_split(task.X, shape, cfg.dup_fraction)
        try:
            with task_rng(seed, step, task_idx, STREAM_REPLICATES + off):
                Y = task.replicates(split.query_rows, R)
        except NonFiniteReplicate:
            log.revoke("non_finite_replicate", noisy=task.target.n_private_roots_used > 0)
            continue
        return UQTask(task, split, Y, redraws=k), log
    raise RuntimeError(f"task (seed {seed}, step {step}, task {task_idx}): non-finite replicates in "
                       f"{max_redraws + 1} draws ({log.summary()})")


# =========================================================================================== training stream (step 4)

# Tasks per optimizer step (design Training §2): half of TabICLv2's 64, to halve the wall time per step.
BATCH_SIZE = 32
# Tasks per micro-batch group (tabicl `--batch_size_per_gp` = `--micro_batch_size`, scripts/train_v2_reg_stage*.sh).
GROUP_SIZE = {1: 4, 2: 1, 3: 1}
# Stage 1 runs 50K steps (design Training §2); any offset above the longest stage keeps the stages' keys disjoint.
STAGE_KEY_OFFSET = 10_000_000


def stage_step_key(stage: int, step: int) -> int:
    """The `step` slot of task_rng for (stage, stage-local step): the stages draw disjoint tasks."""
    assert 0 <= step < STAGE_KEY_OFFSET
    return stage * STAGE_KEY_OFFSET + step


def sample_group_shape(seed: int, key: int, group: int, stage: int, cfg: CutGraphConfig) -> Tuple[int, int, int]:
    """(N, n_ctx, d) shared by a micro-batch group: TabICLv2's per-group seq_len / train_size and per-subgroup
    feature count (`GraphPrior.get_batch`; subgroup = group, as `batch_size_per_subgp` defaults to the group size)."""
    with task_rng(seed, key, group, STREAM_SHAPE):
        s = sample_task_shape(stage, cfg)
    return s.N, s.n_ctx, s.d


@contextlib.contextmanager
def single_thread():
    """1 torch intra-op thread and 1 BLAS / OpenMP thread (sklearn's ExtraTrees in the filter) while building
    tasks: bit-identical tasks across processes need the same thread count (module docstring of `task_rng`), and
    tabicl's `get_batch` does the same. Restores the previous torch setting on exit."""
    n_old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with threadpoolctl.threadpool_limits(limits=1):
            yield
    finally:
        torch.set_num_threads(n_old)


@dataclass
class UQBatch:
    """One micro-batch group of B tasks sharing (seq_len = N, train_size = n_ctx) (protocol §1.3).

    X            (B, N, d_max) float32  rows the model sees, context first; columns past a task's d are zero
                                        padding (tabicl pads to max_features and truncates to max(d))
    y_ctx        (B, n_ctx)    float32  context labels g(y^(0))
    y_reps       (B, m, R)     float32  g-transformed replicates of the query slots: the heads' targets
    query_type   (B, m)        long     IID / HOLE / EXT / DUP (`_split.QUERY_TYPES`)
    d            (B,)          long     active features per task (after dropping constant columns)
    seq_len, train_size        int      N and n_ctx
    stage, step, group         int      where the group sits in the stream (step is stage-local)
    task_idx     (B,)          long     task index within the step
    deterministic (B,)         bool     no private root reaches y (t_A = 0 at every query)
    log                                 the group's RejectionLog (rejections and redraws behind its B tasks)
    gen_time                   float    seconds spent building the group (in its DataLoader worker)
    """
    X: torch.Tensor
    y_ctx: torch.Tensor
    y_reps: torch.Tensor
    query_type: torch.Tensor
    d: torch.Tensor
    seq_len: int
    train_size: int
    stage: int
    step: int
    group: int
    task_idx: torch.Tensor
    deterministic: torch.Tensor
    log: RejectionLog = field(default_factory=RejectionLog)
    gen_time: float = 0.0

    @property
    def B(self) -> int:
        return self.X.shape[0]

    @property
    def m(self) -> int:
        return self.seq_len - self.train_size

    @property
    def R(self) -> int:
        return self.y_reps.shape[-1]

    def _map(self, fn) -> "UQBatch":
        kw = {}
        for f in fields(self):
            v = getattr(self, f.name)
            kw[f.name] = fn(v) if isinstance(v, torch.Tensor) else v
        return UQBatch(**kw)

    def pin_memory(self) -> "UQBatch":
        """Called by the DataLoader when pin_memory=True."""
        return self._map(lambda t: t.pin_memory())

    def to(self, device, non_blocking: bool = False) -> "UQBatch":
        return self._map(lambda t: t.to(device, non_blocking=non_blocking))

    @classmethod
    def stack(cls, tasks: List[UQTask], stage: int, step: int, group: int, task_idx: List[int],
              log: RejectionLog, gen_time: float) -> "UQBatch":
        N, n_ctx = tasks[0].task.shape.N, tasks[0].task.shape.n_ctx
        assert all(t.task.shape.N == N and t.task.shape.n_ctx == n_ctx for t in tasks), "one shape per group"
        d = torch.tensor([t.task.X.shape[1] for t in tasks], dtype=torch.long)
        X = torch.zeros(len(tasks), N, int(d.max()), dtype=torch.float32)
        for b, t in enumerate(tasks):
            X[b, :, :int(d[b])] = t.X_model.float()
        return cls(X=X, y_ctx=torch.stack([t.y_ctx.float() for t in tasks]),
                   y_reps=torch.stack([t.Y.float() for t in tasks]),
                   query_type=torch.stack([t.split.query_type.long() for t in tasks]), d=d,
                   seq_len=N, train_size=n_ctx, stage=stage, step=step, group=group,
                   task_idx=torch.tensor(task_idx, dtype=torch.long),
                   deterministic=torch.tensor([t.deterministic for t in tasks]), log=log, gen_time=gen_time)


class CutGraphPrior(Dataset):
    """The training stream of one stage: map-style, item i = micro-batch group i (module docstring).

    Steps [start_step, max_steps) of stage `stage` on seed `seed`; item i is group i % n_groups of step
    start_step + i // n_groups, whatever the number of workers or the order items are built in
    (`LargestFirstSampler`, `iterate_steps`); a resumed run passes its next step as `start_step` (protocol §6,
    smoke run 3).
    """

    def __init__(self, seed: int, stage: int, max_steps: int, start_step: int = 0, R: int = 16,
                 batch_size: int = BATCH_SIZE, group_size: Optional[int] = None,
                 cfg: Optional[CutGraphConfig] = None):
        self.seed, self.stage, self.R = seed, stage, R
        self.start_step, self.max_steps = start_step, max_steps
        self.cfg = cfg or CutGraphConfig()
        self.group_size = group_size or GROUP_SIZE[stage]
        assert batch_size % self.group_size == 0, "batch_size must be a multiple of the group size"
        self.batch_size, self.n_groups = batch_size, batch_size // self.group_size
        assert 0 <= start_step <= max_steps

    def __len__(self) -> int:
        return (self.max_steps - self.start_step) * self.n_groups

    def __getitem__(self, i: int) -> UQBatch:
        step, group = self.start_step + i // self.n_groups, i % self.n_groups
        return self.group(step, group)

    def group(self, step: int, group: int) -> UQBatch:
        """Build group `group` of stage-local step `step` (independent of any other item)."""
        t0 = time.perf_counter()
        key = stage_step_key(self.stage, step)
        with single_thread():
            N, n_ctx, d = sample_group_shape(self.seed, key, group, self.stage, self.cfg)
            log, tasks, idx = RejectionLog(), [], []
            for j in range(self.group_size):
                task_idx = group * self.group_size + j
                t, _ = sample_uq_task(self.seed, key, task_idx, self.stage, self.cfg, self.R, log=log,
                                      N=N, n_ctx=n_ctx, d=d)
                tasks.append(t)
                idx.append(task_idx)
            out = UQBatch.stack(tasks, self.stage, step, group, idx, log, gen_time=0.0)
        out.gen_time = time.perf_counter() - t0
        return out

    def __repr__(self) -> str:
        return (f"CutGraphPrior(seed={self.seed}, stage={self.stage}, steps=[{self.start_step}, {self.max_steps}), "
                f"batch_size={self.batch_size}, group_size={self.group_size}, n_groups={self.n_groups}, R={self.R})")


class LargestFirstSampler(Sampler):
    """Dispatch order of the stream's items: step by step, and within a step the groups by decreasing N (ties in
    group order; stage 1, where N is always 1,024, keeps the plain group order).

    Task cost grows steeply with N (stage 3: N from 400 to 60,000), and a step is only complete when its slowest
    group is. Starting the largest groups first lets their workers finish while the small groups fill the other
    workers, instead of a 60K-row task starting last and holding the step back (longest-processing-time-first
    scheduling). Only the ORDER IN WHICH ITEMS ARE BUILT changes: item i is still group i % n_groups of step
    start_step + i // n_groups, a pure function of (seed, stage, step, group). The group shapes are recomputed here
    with `sample_group_shape` (under task_rng, which restores the global RNG state; microseconds per group).
    """

    def __init__(self, prior: CutGraphPrior):
        self.prior = prior

    def __len__(self) -> int:
        return len(self.prior)

    def __iter__(self):
        p = self.prior
        for s in range(p.max_steps - p.start_step):
            key = stage_step_key(p.stage, p.start_step + s)
            sizes = [sample_group_shape(p.seed, key, g, p.stage, p.cfg)[0] for g in range(p.n_groups)]
            for g in sorted(range(p.n_groups), key=lambda g: -sizes[g]):   # sorted() is stable
                yield s * p.n_groups + g


def _worker_init(worker_id: int):
    # No reseeding (every task seeds itself through task_rng); one intra-op thread, as in the determinism tests.
    torch.set_num_threads(1)


def make_dataloader(prior: CutGraphPrior, num_workers: int, prefetch_factor: int = 4,
                    pin_memory: Optional[bool] = None) -> DataLoader:
    """tabicl's on-the-fly DataLoader (train/_run.py `configure_prior`: batch_size=None, persistent workers,
    pinned memory when CUDA is available), with three changes that keep every worker busy (protocol §3 step 4):

      in_order=False         a finished item is handed over at once, and a new item is only queued on a worker that
                             holds fewer than `prefetch_factor` items (torch's dispatch rule when in_order=False).
                             With in_order=True one slow item blocks the delivery of every later one, and new work
                             keeps going round-robin to the busy worker, so the other workers sit idle (most of
                             their capacity in stages 2-3, where task cost is heavy-tailed in N).
      LargestFirstSampler    each step's groups are built largest N first.
      prefetch_factor 4      items in flight = workers x 4; new work is only queued when the main process pulls,
                             so this must cover the items workers build while the GPU runs a step (>= 2 steps of
                             stage 2-3 items at 32 workers).
    Items therefore arrive in no fixed order; `iterate_steps` restores the stream order before the trainer sees
    anything, so the trainer's batches are exactly those of an in-order loader.
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    kw = dict(prefetch_factor=prefetch_factor, persistent_workers=True, in_order=False) if num_workers > 0 else {}
    return DataLoader(prior, batch_size=None, sampler=LargestFirstSampler(prior), num_workers=num_workers,
                      pin_memory=pin_memory, worker_init_fn=_worker_init, **kw)


def iterate_steps(loader: DataLoader, on_item: Optional[Callable[[UQBatch], None]] = None
                  ) -> Iterator[Tuple[int, List[UQBatch]]]:
    """Yield (step, the step's n_groups groups in group order) for steps start_step, start_step + 1, ...

    The reorder buffer: items arrive in any order (out-of-order delivery and largest-first dispatch,
    `make_dataloader`); each is filed under its (step, group), and step s is yielded as soon as all its groups are
    in, sorted by group index. Items of later steps wait in the buffer. The yielded steps are therefore identical,
    in content and order, to an in-order loader's. The buffer holds at most the items in flight beyond the current
    step (bounded by workers x prefetch_factor, since new work is only queued when this loop pulls).
    `on_item(item)` is called on every arriving item (the benchmark uses it to record arrival order).
    """
    prior: CutGraphPrior = loader.dataset
    buffer: Dict[int, Dict[int, UQBatch]] = {}
    step = prior.start_step
    for item in loader:
        if on_item is not None:
            on_item(item)
        assert prior.start_step <= item.step < prior.max_steps and 0 <= item.group < prior.n_groups
        slot = buffer.setdefault(item.step, {})
        assert item.group not in slot and item.step >= step, f"item (step {item.step}, group {item.group}) twice"
        slot[item.group] = item
        while len(buffer.get(step, ())) == prior.n_groups:
            groups = buffer.pop(step)
            yield step, [groups[g] for g in range(prior.n_groups)]
            step += 1
    assert not buffer and step == prior.max_steps, f"stream ended with incomplete steps {sorted(buffer)}"
