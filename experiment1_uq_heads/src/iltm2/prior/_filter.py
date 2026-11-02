"""Task filters of the observed-cut prior (phase1_design.md Training §1.4, proposal §4.3 "Task filtering").

WHY FILTERS AT ALL
    TabICLv2 was pretrained only on tasks that pass its filters (`filter_unpredictable_graphs` and
    `filter_unpredictable_datasets` are on in every `scripts/train_v2_reg_stage*.sh`; about 25% of stage-1
    regression tasks are rejected). The frozen backbone's posterior is therefore conditioned on "the task passed".
    Our tasks must pass the same filters, or the heads would be trained on tasks outside the backbone's prior.
    Every function here is a thin wrapper around tabicl's own code, so the filter behaviour stays tabicl's.

THE FILTERS, IN THE ORDER `CutGraphSCM.sample` APPLIES THEM
    | Check                | tabicl code                                       | Use in the cut                           |
    |----------------------|---------------------------------------------------|------------------------------------------|
    | non-finite values    | NaN guard in `GraphSCM.__call__`                  | reject if X, y^(0) (or later a replicate) |
    |                      |                                                   | has a NaN or inf                         |
    | constant columns     | `Prior.delete_unique_features` (prior/_dataset.py) | as-is on the X pool; reject if none left |
    | graph check          | `check_x_y_ancestors_overlap`                     | ADAPTED: on the target graph, with its   |
    |                      | (prior/graph_lib/_dataset.py)                     | X-input nodes tagged "x" (the X-graph has |
    |                      |                                                   | no y node, so it is never checked there) |
    | predictability test  | `should_filter` (prior/_dataset.py)               | as-is, on N uniformly drawn pool rows     |
    |                      |                                                   | with their g-transformed y^(0)           |
    | trivial-task test    | `should_filter(remove_trivial_datasets=True)`     | OFF, as in TabICLv2's scripts: the        |
    |                      |                                                   | near-deterministic A ≈ 0 end is learned   |

    Every rejection is task-level and happens before the split and the replicates: the whole task (graphs, pool,
    noise) is redrawn, never repaired by redrawing only the noise. The graph check is the exception in form only:
    like in tabicl, a failing graph is resampled on the spot (it has not generated any data yet).

WHY THE PREDICTABILITY TEST DOES NOT BIAS THE TARGETS
    It looks at y^(0), so it selects on y, but only at the task level, exactly as in TabICLv2: it changes which
    tasks enter the prior. Given an accepted task, the R replicates are fresh U2 draws the test never saw, so
    t_A and t_E stay unbiased for the posterior of the filtered prior, which is what the backbone amortizes.
"""
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import torch

from tabicl.prior._dataset import Prior, should_filter
from tabicl.prior.graph_lib._base import FeatureSpec
from tabicl.prior.graph_lib._config import PriorConfig
from tabicl.prior.graph_lib._dataset import check_x_y_ancestors_overlap

# Rejection reasons, in the order the checks run. Graph resamples are counted separately (not a task rejection).
REASONS = ("non_finite_x", "x_scaling_clip", "constant_columns", "non_finite_y", "unpredictable",
           "non_finite_replicate")


# ------------------------------------------------------------------------------------------- graph check


def graph_check(dag: List[List[int]], input_nodes: Sequence[int], y_node: int) -> bool:
    """True if at least one X-input node of the target graph is an ancestor of the y node.

    Reuses tabicl's `check_x_y_ancestors_overlap` unchanged. That function tags nodes by feature group and returns
    True if some node is an ancestor of both an "x" node and the "y" node (a node counts as its own ancestor). In
    the target graph the X-input nodes have no parents, so their "x" ancestor set is themselves, and the check
    reduces to "some X-input node is an ancestor of y": y depends on X by construction, the cut's analogue of
    tabicl's "x and y share an ancestor".
    """
    specs: List[Dict[str, FeatureSpec]] = [dict() for _ in dag]
    for p in input_nodes:
        specs[p][f"x{p}"] = FeatureSpec(group="x")
    specs[y_node]["y"] = FeatureSpec(group="y")
    return bool(check_x_y_ancestors_overlap(dag, specs))


def ancestors(dag: List[List[int]], node: int) -> Set[int]:
    """All ancestors of `node` in a parent-list DAG, including `node` itself."""
    out, stack = {node}, [node]
    while stack:
        for p in dag[stack.pop()]:
            if p not in out:
                out.add(p)
                stack.append(p)
    return out


# ------------------------------------------------------------------------------------------- constant columns


def drop_constant_columns(X: torch.Tensor) -> torch.Tensor:
    """Boolean mask (d,) of the columns of X (n, d) that take more than one value.

    tabicl's `Prior.delete_unique_features` removes single-valued columns and returns the reduced tensor and the
    new feature count, not the mask we need to keep track of which columns survive. The mask below uses the same
    rule (`len(torch.unique(col)) > 1`), and tabicl's function is called to check that both keep the same number
    of columns. The caller rejects the task when the mask is all False (tabicl: `if not (d > 0).all(): continue`).
    """
    d = X.shape[1]
    keep = torch.tensor([len(torch.unique(X[:, j])) > 1 for j in range(d)], dtype=torch.bool)
    _, d_tabicl = Prior.delete_unique_features(X[None], torch.tensor([d]))
    assert int(d_tabicl[0]) == int(keep.sum()), "constant-column rule disagrees with tabicl's"
    return keep


# ------------------------------------------------------------------------------------------- predictability test


def predictability_rejects(X: torch.Tensor, y: torch.Tensor, prior_config: PriorConfig, n_rows: Optional[int] = None,
                           rng: Optional[np.random.Generator] = None) -> bool:
    """True if TabICLv2's predictability test rejects the task (y, X) .

    Runs tabicl's `should_filter` unchanged: ExtraTrees (25 trees, depth 6, bootstrap) is fitted, and the task is
    rejected unless its out-of-bag MSE beats the mean-label baseline on >= 95% of 200 bootstrap resamples. The
    trivial-task test is forced off and the predictability test forced on, whatever `prior_config` says.

    TabICLv2 tests datasets of N rows; our pool has P >= N rows (the extra rows feed the hole and the tail). So,
    when `n_rows` is given, the test runs on `n_rows` rows drawn uniformly without replacement from the pool
    (`rng`, or the global numpy RNG, which `task_rng` seeds).
    """
    if n_rows is not None and n_rows < X.shape[0]:
        draw = rng.choice if rng is not None else np.random.choice
        rows = torch.from_numpy(np.sort(draw(X.shape[0], size=n_rows, replace=False)))
        X, y = X[rows], y[rows]
    cfg = replace(prior_config, filter_unpredictable_datasets=True, remove_trivial_datasets=False)
    return bool(should_filter(X, y, cfg, is_classif=False))


def all_finite(*tensors: torch.Tensor) -> bool:
    """True if no tensor holds a NaN or an inf (tabicl's guard only catches NaN; inf is rejected too)."""
    return all(bool(torch.isfinite(t).all()) for t in tensors)


# ------------------------------------------------------------------------------------------- bookkeeping


@dataclass
class RejectionLog:
    """Counts of accepted tasks, task-level rejections per reason, and target-graph resamples.

    The protocol's step-2 gate reads `rate` (task rejections / task attempts, which TabICLv2 puts at about 25% in
    stage 1) and expects 5%-30% for the cut prior. Graph resamples are not task rejections: tabicl's
    `RandomDataset.sample` also resamples failing graphs on the spot, and its ~25% figure excludes them.
    """
    accepted: int = 0
    rejected: Counter = field(default_factory=Counter)
    graph_resamples: int = 0

    def reject(self, reason: str):
        assert reason in REASONS, f"unknown rejection reason {reason!r}"
        self.rejected[reason] += 1

    @property
    def attempts(self) -> int:
        return self.accepted + sum(self.rejected.values())

    @property
    def rate(self) -> float:
        return sum(self.rejected.values()) / max(1, self.attempts)

    def __iadd__(self, other: "RejectionLog") -> "RejectionLog":
        self.accepted += other.accepted
        self.rejected += other.rejected
        self.graph_resamples += other.graph_resamples
        return self

    def summary(self) -> str:
        parts = ", ".join(f"{r} {self.rejected[r]}" for r in REASONS if self.rejected[r])
        return (f"rejection rate {self.rate:.1%} ({sum(self.rejected.values())}/{self.attempts}"
                f"{': ' + parts if parts else ''}); target-graph resamples {self.graph_resamples}")
