"""Validation of head pairs on the fixed bank and the mini-bank (phase1_protocol.md §7.4-7.5; design Training §3).

WHAT IS FIXED AND WHAT IS NOT
    The bank stores, per query, the targets (t_A, t_E, half splits) and e(q_theta) of the frozen forecast, and the
    reference lines (`iltm2.prior._bank`, scripts/01_make_val_bank.py). The PPD is frozen in every arm, so none of
    that is recomputed: validating a head pair = getting its INPUT for every bank query, running the heads, and
    scoring the predictions with `iltm2.train._metrics.head_metrics` against the stored targets.

THE INPUTS, CACHED OR RECOMPUTED (protocol §1.3, "Precomputed bank targets")
    The inputs are a deterministic function of the frozen backbone, except arm B's fork output. `BankEvaluator`
    can cache, once at job start, any of:
        "h"      h_star per query (A1)          bank: about 6.4 GB in CPU RAM (3.1M queries x 512 x 4 bytes)
        "taps"   the 12 taps per query (A2)     bank: about 77 GB, too large: recomputed; mini-bank: about 0.6 GB
        "fork"   the fork input, ALL rows (B)   bank: about 30 GB: recomputed; mini-bank: about 0.13 GB
    Intended use: A jobs cache "h" on the bank and ("h", "taps") on the mini-bank; B jobs cache "fork" on the
    mini-bank only. A pass that needs an input that is not cached runs the backbone forward task by task (one task
    at a time, as the bank was built) and serves every pair from it.

GUARDS
    - The backbone's sha256 must equal the bank's (the targets belong to one specific forecast).
    - Every recomputed e(q) must match the stored one: the forward that feeds the heads is the forward that made
      the targets. Per task, max |diff| <= 1e-6 + 1e-3 x mean |stored e(q)| of that task. The tolerance is set on
      the task's e(q) SCALE, not per query: on the GPU type that built the bank (the training GPU) the recomputed
      e(q) is bit-equal, but on another device (e.g. the CPU unit tests) float32 round-off gives absolute
      differences of a few 1e-6, and e(q) is a spread of nearby quantiles, so at a query with a small e(q) that
      round-off is a large per-query RELATIVE error (several 1e-4 seen). A wrong checkpoint or forward route moves
      e(q) by a sizeable fraction of its scale, far above 1e-3.
    - Task order: tasks are read in bank order and each one's id and query count are checked against the query
      table, so predictions line up with the targets row by row (`check_alignment` does this without forwards).

BEST-PER-HEAD CHECKPOINTS (design Training §3)
    `BestTracker` keeps, per (head pair, head), the lowest pooled validation MSE seen so far ("val/mse_<h>/all" on
    the full bank, never the mini-bank) and flags when it improves; the trainer then overwrites best/<id>_<h>.pt.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from iltm2.model._heads import HEADS, Fork, HeadPair, select_input
from iltm2.prior._bank import iter_tasks, load_manifest, load_queries
from iltm2.train._metrics import head_metrics

E_Q_ATOL, E_Q_RTOL = 1e-6, 1e-3    # e(q) guard, relative to the task's mean e(q) (docstring, GUARDS)
CACHE_KINDS = ("h", "taps", "fork")


class BankEvaluator:
    """Head-pair validation on one bank (full or mini). Not tied to a trainer: it holds the query table, the
    reference lines and optional input caches, and is given the head pairs (and B's fork) at each call.

    root      the bank directory (the mini-bank is root/mini, selected by `mini=True`)
    backbone  the job's FrozenBackbone (taps enabled if A2 pairs are evaluated without a "taps" cache; fork_k set
              if B is evaluated without a "fork" cache)
    cache     input kinds to cache at construction (module docstring)
    """

    def __init__(self, root, backbone, mini: bool = False, cache: Sequence[str] = (), device=None):
        self.root, self.mini, self.backbone = Path(root), mini, backbone
        self.device = torch.device(device) if device is not None else backbone.device
        man = load_manifest(self.root)
        assert backbone.sha256 == man["backbone"]["sha256"], (
            f"backbone sha256 {backbone.sha256[:12]}... is not the bank's {man['backbone']['sha256'][:12]}...: "
            "the bank's targets belong to another forecast")
        self.offsets = man["task_offsets"]
        self.Q = load_queries(self.root, mini=mini)
        self.refs: Dict[str, float] = man["mini"]["reference_lines"] if mini else man["reference_lines"]
        self.ref_prefix, self.prefix = ("ref_mini", "mini") if mini else ("ref", "val")
        # Contiguous query rows per task, in table order.
        ids, counts = torch.unique_consecutive(self.Q.task_id, return_counts=True)
        assert bool((ids[1:] > ids[:-1]).all()), "query table not in task order"
        self.task_ids: List[int] = ids.tolist()
        self.bounds: List[Tuple[int, int]] = []
        lo = 0
        for c in counts.tolist():
            self.bounds.append((lo, lo + c))
            lo += c
        self.cache: Dict[str, List[Tensor]] = {}
        self.max_eq_diff = 0.0          # largest |recomputed - stored| e(q) seen so far
        self.max_eq_scaled = 0.0        # the same, relative to the task's mean |stored e(q)|
        if cache:
            self.build_cache(cache)

    # ----- tasks and the backbone forward

    def _records(self):
        """(position in the table, task record), with the id and query count checked against the table."""
        i = -1
        for i, rec in enumerate(iter_tasks(self.root, mini=self.mini)):
            assert i < len(self.task_ids), "more tasks than the query table"
            tid = self.offsets[rec["bin"]] + rec["index"]
            lo, hi = self.bounds[i]
            assert tid == self.task_ids[i] and rec["m"] == hi - lo, (
                f"task {i}: record ({rec['bin']}, {rec['index']}, m={rec['m']}) does not match the query table "
                f"(id {self.task_ids[i]}, m={hi - lo})")
            yield i, rec
        assert i == len(self.task_ids) - 1, "fewer tasks than the query table"

    def check_alignment(self) -> int:
        """Walk the bank's task records against the query table, no forward. Returns the number of tasks."""
        n = 0
        for n, _ in enumerate(self._records(), start=1):
            pass
        return n

    @torch.no_grad()
    def _forward(self, i: int, rec: dict):
        """The backbone forward of task i, with its recomputed e(q) checked against the stored one."""
        out = self.backbone(rec["X"][None], rec["y_ctx"][None])
        lo, hi = self.bounds[i]
        stored = self.Q.e_q[lo:hi].double()
        err = (out.e_q[0].double().cpu() - stored).abs()
        scale = float(stored.abs().mean())
        diff = float(err.max())
        scaled = diff / max(scale, E_Q_ATOL)
        self.max_eq_diff, self.max_eq_scaled = max(self.max_eq_diff, diff), max(self.max_eq_scaled, scaled)
        if not diff <= E_Q_ATOL + E_Q_RTOL * scale:
            raise RuntimeError(f"task {self.task_ids[i]}: recomputed e(q) differs from the bank's by {diff:.2e} "
                               f"({scaled:.2e} of the task's mean e(q) {scale:.3g}): "
                               "the backbone forward is not the one that built the bank")
        return out

    @torch.no_grad()
    def build_cache(self, kinds: Iterable[str]):
        kinds = tuple(kinds)
        assert all(k in CACHE_KINDS for k in kinds), kinds
        if "taps" in kinds:
            assert self.backbone.tap_layers, "caching taps needs a backbone with tap_layers"
        if "fork" in kinds:
            assert self.backbone.fork_k is not None, "caching the fork input needs a backbone with fork_k"
        store = {k: [] for k in kinds}
        for i, rec in self._records():
            out = self._forward(i, rec)
            if "h" in kinds:
                store["h"].append(out.h_star[0].float().cpu())
            if "taps" in kinds:
                store["taps"].append(out.taps[0].float().cpu())
            if "fork" in kinds:
                store["fork"].append(out.fork_input[0].float().cpu())
        self.cache.update(store)

    # ----- predictions

    @staticmethod
    def _needs(pairs: Sequence[HeadPair]) -> set:
        return {{"A1": "h", "A2": "taps", "B": "fork"}[p.arm] for p in pairs}

    @torch.no_grad()
    def predict(self, pairs: Sequence[HeadPair], fork: Optional[Fork] = None) -> Dict[str, Dict[str, Tensor]]:
        """{head_id: {"A": Â, "E": Ê_dir}} over every query of the table, in table order (float32, CPU)."""
        needs = self._needs(pairs)
        if "fork" in needs:
            assert fork is not None, "B pairs need the fork"
        recompute = not needs <= set(self.cache)
        if recompute:
            if "taps" in needs:
                assert self.backbone.tap_layers, "A2 pairs without a taps cache need a backbone with taps"
            if "fork" in needs:
                assert self.backbone.fork_k is not None, "B pairs without a fork cache need fork_k on the backbone"
        parts: Dict[str, Dict[str, List[Tensor]]] = {p.head_id: {h: [] for h in HEADS} for p in pairs}
        dev = self.device
        tasks = self._records() if recompute else ((i, None) for i in range(len(self.task_ids)))
        for i, rec in tasks:
            if recompute:
                out = self._forward(i, rec)
                h_star, taps, fork_in, ts = out.h_star, out.taps, out.fork_input, out.train_size
            else:
                get = lambda k: self.cache[k][i].to(dev)[None] if k in needs else None  # noqa: E731
                h_star, taps, fork_in = get("h"), get("taps"), get("fork")
                ts = None if fork_in is None else fork_in.shape[1] - (self.bounds[i][1] - self.bounds[i][0])
            fork_h = fork(fork_in.to(dev), ts) if "fork" in needs else None
            for p in pairs:
                a, e = p(select_input(p.arm, h_star=h_star, taps=taps, fork_h=fork_h))
                parts[p.head_id]["A"].append(a[0].float().cpu())
                parts[p.head_id]["E"].append(e[0].float().cpu())
        return {hid: {h: torch.cat(v[h]) for h in HEADS} for hid, v in parts.items()}

    def evaluate(self, pairs: Sequence[HeadPair], fork: Optional[Fork] = None, best: Optional["BestTracker"] = None,
                 step: Optional[int] = None, slices: Optional[tuple] = None) -> Dict[str, float]:
        """Every pair's metrics, keys prefixed by the head id (protocol §7.1: e.g. 'A1_lr3e-4/val/xmse_E/hole'),
        plus the bank's reference lines (logged as constants, so they draw as flat lines on the same panels) and,
        if a BestTracker is given (full bank only), the '<id>/val/is_best_<h>' flags. The mini-bank defaults to the
        pooled slice only."""
        if slices is None and self.mini:
            slices = ("all",)
        preds = self.predict(pairs, fork)
        out: Dict[str, float] = {k: v for k, v in self.refs.items()       # per-slice lines of the kept slices
                                 if slices is None or k.count("/") == 1 or k.rsplit("/", 1)[-1] in slices}
        for p in pairs:
            m = head_metrics(preds[p.head_id], self.Q, self.refs, self.ref_prefix, self.prefix, slices)
            out.update({f"{p.head_id}/{k}": v for k, v in m.items()})
            if best is not None:
                assert not self.mini, "best checkpoints are chosen on the full bank, never the mini-bank"
                for h, improved in best.update(p.head_id, m, step).items():
                    out[f"{p.head_id}/{self.prefix}/is_best_{h}"] = float(improved)
        return out


class BestTracker:
    """Lowest pooled validation MSE per (head pair, head) so far (design Training §3). Resumable."""

    KEY = "val/mse_{h}/all"

    def __init__(self):
        self.best: Dict[str, Dict[str, Tuple[float, Optional[int]]]] = {}

    def update(self, head_id: str, metrics: Dict[str, float], step: Optional[int]) -> Dict[str, bool]:
        """metrics: the pair's `head_metrics` (unprefixed keys). Returns {"A": improved?, "E": improved?}."""
        cur = self.best.setdefault(head_id, {})
        flags = {}
        for h in HEADS:
            v = metrics[self.KEY.format(h=h)]
            improved = h not in cur or v < cur[h][0]
            if improved:
                cur[h] = (v, step)
            flags[h] = improved
        return flags

    def state_dict(self) -> dict:
        return {hid: {h: list(v) for h, v in d.items()} for hid, d in self.best.items()}

    def load_state_dict(self, d: dict):
        self.best = {hid: {h: (v[0], v[1]) for h, v in dd.items()} for hid, dd in d.items()}
