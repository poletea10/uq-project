"""The split of an observed-cut task into context and queries (design Training §1.1-1.2, protocol §3 step 3).

WHAT IT DOES
    An accepted task (step 2) is a pool of P rows. The model sees exactly N = n_ctx + m of them: n_ctx context rows
    with their labels and m query rows. The queries mix four types, so that the heads see both high-divergence and
    low-divergence regimes:

      extrapolation  (default 15%)  rows beyond the support of the context in some direction
      kNN hole       (default 15%)  rows inside a region the context does not cover
      i.i.d.         (default 70%)  ordinary rows, exactly TabICLv2's random split
      duplicate      (5% of slots)  a context row's x, asked again as a query

    The pool holds the N kept rows plus exactly the P - N rows the regions discard (`TaskShape`, pipeline steps
    1-3), so n_ctx and m are always TabICLv2's sizes for the stage.

THE CONSTRUCTION (in this order; each step removes a region from what is left; no row is ever created)
    1. Extrapolation. Draw a unit direction u uniformly on the sphere, in the standardized X space (`task.X` is
       standardized on the pool), and score every row by s = x . u. The T rows with the largest s are the tail.
       m_ext of them, chosen uniformly, are extrapolation queries; the other T - m_ext are discarded.
    2. kNN hole. Draw a centre uniformly among the P - T remaining rows. The centre and its nearest remaining
       neighbours, k rows in total (Euclidean distance in standardized X), are the hole. m_hole of them, chosen
       uniformly, are hole queries; the other k - m_hole are discarded. Only one centre needs neighbours, so the
       hole is one distance computation over the remaining rows and a sort: O(P d), fast even for stage-3 pools
       (P up to ~84K); no approximate-neighbour index is needed.
    3. I.i.d. The P - T - k = n_ctx + m_iid remaining rows are an ordinary i.i.d. pool: m_iid of them, chosen
       uniformly, are i.i.d. queries, the rest is the context (exactly n_ctx rows).
    4. Duplicates. round(dup_fraction * m) query slots, chosen uniformly across the three types, are overwritten
       with the x of a random context row (design §1.2). The slot keeps no trace of its original row.
    Bookkeeping: queries = m_ext + m_hole + m_iid = m; context = n_ctx; discarded = (T - m_ext) + (k - m_hole) =
    P - N. Region sizes are row counts (not radii), so they stay meaningful in any dimension.

WHY IT MAY ONLY LOOK AT X
    Every decision here depends only on X and on random numbers independent of y. The y-law is produced by its own
    graph from X, so an X-only selection carries no information about it, and the posterior over the y-law given
    the manipulated context is unchanged: the targets keep their meaning E_{tau|D}[.] (design §1, "Why
    manipulating the split is allowed"). Selecting on y (e.g. dropping context rows with large residuals) would
    break this. `make_split` therefore never receives y: it takes X, the task shape and the duplicate share.

OUTPUT: ROWS OF THE POOL, NOT DATA
    A `Split` holds pool row indices. `model_rows` = [context rows, query rows] are the N rows the model sees, in
    that order (context first, as TabICLv2's forward expects). For a duplicate slot, `query_rows` holds the copied
    CONTEXT row, so that both its x (`task.X[query_rows]`) and the target graph's input for its replicates
    (`task.replicates(query_rows, R)`) are that context row's. Its replicates are fresh U2 draws; the context row's
    label is never used as a target.

RANDOMNESS
    All draws come from the global torch RNG: call `make_split` inside `task_rng(seed, step, task_idx,
    STREAM_SPLIT)`. Ties (equal scores or distances) are broken by stable sorts, so a split is a pure function of
    (X, shape, RNG state).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from iltm2.prior._dataset import TaskShape, _round

# Query-type codes (protocol §7.5 logs every metric per type).
IID, HOLE, EXT, DUP = 0, 1, 2, 3
QUERY_TYPES = {IID: "iid", HOLE: "hole", EXT: "extrapolation", DUP: "duplicate"}


@dataclass
class Split:
    """Pool row indices of one split (all LongTensors).

    ctx_rows (n_ctx)        context rows
    query_rows (m)          query rows; for a duplicate slot, the copied context row
    query_type (m)          IID, HOLE, EXT or DUP per query slot
    discarded (P - N)       tail and hole rows that are neither context nor query
    For tests and figures: u (d) the extrapolation direction, tail_rows (T), tail_threshold (the smallest tail
    score; nan if T = 0), hole_centre (pool row; -1 if k = 0), hole_rows (k), query_rows_before_dup (m).
    """
    ctx_rows: torch.Tensor
    query_rows: torch.Tensor
    query_type: torch.Tensor
    discarded: torch.Tensor
    u: torch.Tensor
    tail_rows: torch.Tensor
    tail_threshold: float
    hole_centre: int
    hole_rows: torch.Tensor
    query_rows_before_dup: torch.Tensor

    @property
    def model_rows(self) -> torch.Tensor:
        """The N rows the model sees: context first, then queries."""
        return torch.cat([self.ctx_rows, self.query_rows])


def _take(rows: torch.Tensor, n: int):
    """Split `rows` into n chosen uniformly at random and the rest (both in random order)."""
    perm = torch.randperm(len(rows))
    return rows[perm[:n]], rows[perm[n:]]


def _without(rows: torch.Tensor, removed: torch.Tensor) -> torch.Tensor:
    """`rows` minus `removed`, in increasing row order."""
    keep = torch.ones(int(rows.max()) + 1 if len(rows) else 0, dtype=torch.bool)
    keep[removed] = False
    return rows[keep[rows]]


def make_split(X: torch.Tensor, shape: TaskShape, dup_fraction: float) -> Split:
    """Split a pool of P rows (X: (P, d), standardized) into context, queries and discarded rows; X only."""
    P, d = X.shape
    assert P == shape.P, f"pool has {P} rows, the shape expects {shape.P}"
    rows = torch.arange(P)

    # ---- 1. extrapolation: the T rows furthest along a random direction u
    u = torch.randn(d, dtype=X.dtype)
    u = u / u.norm()
    if shape.T > 0:
        score = X @ u
        order = torch.sort(score, descending=True, stable=True).indices
        tail_rows = order[:shape.T]
        tail_threshold = float(score[tail_rows[-1]])
    else:
        tail_rows, tail_threshold = rows[:0], float("nan")
    ext_q, ext_discard = _take(tail_rows, shape.m_ext)
    remaining = _without(rows, tail_rows)

    # ---- 2. kNN hole: a random centre and its nearest remaining rows, k in total
    if shape.k > 0:
        c = int(torch.randint(len(remaining), (1,)))
        hole_centre = int(remaining[c])
        dist = (X[remaining] - X[hole_centre]).square().sum(dim=1)
        dist[c] = -1.0                                       # the centre is always in its own hole
        hole_rows = remaining[torch.sort(dist, stable=True).indices[:shape.k]]
    else:
        hole_centre, hole_rows = -1, rows[:0]
    hole_q, hole_discard = _take(hole_rows, shape.m_hole)
    remaining = _without(remaining, hole_rows)

    # ---- 3. i.i.d.: TabICLv2's random split of what is left
    assert len(remaining) == shape.n_ctx + shape.m_iid
    iid_q, ctx_rows = _take(remaining, shape.m_iid)

    # ---- 4. duplicates: some query slots, of any type, get a random context row's x
    query_rows = torch.cat([ext_q, hole_q, iid_q])
    query_type = torch.cat([torch.full((len(ext_q),), EXT), torch.full((len(hole_q),), HOLE),
                            torch.full((len(iid_q),), IID)])
    before_dup = query_rows.clone()
    n_dup = _round(dup_fraction * shape.m)
    slots = torch.randperm(shape.m)[:n_dup]
    query_rows[slots] = ctx_rows[torch.randint(len(ctx_rows), (n_dup,))]
    query_type[slots] = DUP

    return Split(ctx_rows=ctx_rows, query_rows=query_rows, query_type=query_type,
                 discarded=torch.cat([ext_discard, hole_discard]), u=u, tail_rows=tail_rows,
                 tail_threshold=tail_threshold, hole_centre=hole_centre, hole_rows=hole_rows,
                 query_rows_before_dup=before_dup)
