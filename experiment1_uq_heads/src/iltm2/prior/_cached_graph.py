"""graph_lib's random pieces with every task-level draw cached (proposal §4.3, CLAUDE.md "graph-prior gotchas").

THE PROBLEM
    A graph_lib task is defined by everything its generator draws: the DAG, the node functions, the converters and
    their fitted constants, and the laws of the root nodes. Only the base points drawn at the roots on each call
    (and, in the X-graph, the per-row categorical softmax draws) are the randomness *within* the task. As shipped,
    however, `RandomNodeFunction._transform` redraws three task-level things on EVERY call:
      - the root point law: `RandomPoints.sample` re-picks the base-point family (uniform, circle, Gaussian,
        covariance), redraws the covariance matrix and its inner family, and pushes the points through a brand-new
        `RandomFunction`;
      - the per-node feature-importance weights;
      - the `node_importance` scalar.
    So a second call of a graph function is a DIFFERENT task. That silently breaks two things the cut needs:
      - `ensure_iid=True` fits the node transforms on a first call and generates data with a second call; without
        caching the two calls are different tasks (any pool assembled from more than one call is invalid too);
      - replicates of y are "hold X, redraw U2, rerun the target graph"; without caching the rerun would also
        redraw the task, so the "replicates" would come from different task laws and t_A would be wrong.

THE FIX
    Draw the three things once, when the node is fitted, and reuse them on every later call:
      RootPointLaw         cached `RandomPoints`: family, covariance matrix and inner family drawn in `__init__`,
                           the warping `RandomFunction` fitted on the first call; `sample` draws fresh base points
      CachedNodeFunction   `RandomNodeFunction` whose `_fit` also draws the root law, the feature-importance
                           weights and `node_importance`; its `_transform` is tabicl's, line for line, with those
                           three replaced by the cached values
      CachedGraphFunction  `RandomGraphFunction` built from `CachedNodeFunction`s, with tabicl's pruning
    After fitting, a call redraws only the base points at roots: U1 in the X-graph, U2 in the target graph (its
    private roots; its X-input nodes are non-root nodes whose parent value is the observed block, so they are
    deterministic functions of X).

TESTING THE FIX: THE TAPE
    Every base-point draw goes through `RootPointLaw.sample`. With `set_tape(laws, "record")` each draw is stored;
    with "replay" the stored draws are used instead of fresh ones. If a replayed call under a different global RNG
    state reproduces the recorded call bit for bit, nothing but the base points is redrawn between calls, which is
    exactly the caching requirement (scripts/00b_test_cut_prior.py, section 1). The tape is off in normal use.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from tabicl.prior.graph_lib._base import Context, FeatureSpec, PriorComponent
from tabicl.prior.graph_lib._function import RandomFunction
from tabicl.prior.graph_lib._graph_function import RandomGraphFunction
from tabicl.prior.graph_lib._node_function import RandomNodeFunction
from tabicl.prior.graph_lib._points import (RandomCirclePoints, RandomCovariancePoints, RandomGaussianPoints,
                                            RandomUniformPoints)
from tabicl.prior.graph_lib._weights import RandomWeights


# =========================================================================================== cached graph_lib pieces


class RootPointLaw(PriorComponent):
    """Cached version of tabicl's `RandomPoints` (graph_lib/_points.py): the law of a root node's values.

    tabicl's `RandomPoints.sample(n_batch, n)` does, on every call: pick a base-point family (uniform, circle,
    Gaussian, covariance), draw base points from it (for covariance points, also a fresh random matrix and inner
    family), and push them through a brand-new `RandomFunction(n, n)`. Everything except the base points is part
    of the task, so here it is drawn once in `__init__` (family, covariance matrix and inner family) or fitted
    once on the first call (the `RandomFunction`, i.e. on the `ensure_iid` fit pass). `sample` then draws only
    fresh base points: these are the noise of the root (U1 in the X-graph, U2 in the target graph).

    Tape (tests only): with `tape_mode = "record"` every base-point draw is appended to `tape`; with "replay" the
    draws are popped from `tape` instead of drawn. Replaying a recorded pass under a different RNG state must
    reproduce it bit for bit, which proves that nothing but the base points is redrawn.
    """

    def __init__(self, context: Context, n: int):
        super().__init__(context)
        self.n = n
        family = self.sampler.choice(
            "random_base_points",  # same name and choices as RandomPoints.sample
            [RandomUniformPoints, RandomCirclePoints, RandomGaussianPoints, RandomCovariancePoints],
        )
        self.family_ = family
        if family is RandomCovariancePoints:
            # RandomCovariancePoints.sample: matrix = randn(n, n) * weights; base = inner_family @ matrix.T
            self.cov_matrix_ = torch.randn(n, n) * RandomWeights(self.context).sample(1, n)
            inner = self.sampler.choice("random_covariance_base_points", [RandomUniformPoints, RandomGaussianPoints])
            self.base_ = inner(self.context)
        else:
            self.base_ = family(self.context)
        self.warp_ = RandomFunction(self.context, n, n)
        self.tape_mode: Optional[str] = None
        self.tape: List[torch.Tensor] = []

    def draw_base(self, n_rows: int) -> torch.Tensor:
        base = self.base_.sample(n_rows, self.n)
        if self.family_ is RandomCovariancePoints:
            base = base @ self.cov_matrix_.t()
        return base

    def sample(self, n_rows: int) -> torch.Tensor:
        if self.tape_mode == "replay":
            base = self.tape.pop(0)
            assert base.shape[0] == n_rows, "replayed base points have the wrong number of rows"
        else:
            base = self.draw_base(n_rows)
            if self.tape_mode == "record":
                self.tape.append(base.clone())
        return self.warp_(base)


class CachedNodeFunction(RandomNodeFunction):
    """tabicl's `RandomNodeFunction` (graph_lib/_node_function.py) with every task-level draw cached in `_fit`.

    `_fit` first runs tabicl's own `_fit` (converters, node width = LogInt(1, 32) + converter widths, the random
    multi-input function, `Standardize`, `L2Normalize`), then draws the three things tabicl redraws on every call:
    the root point law (roots only), the feature-importance weights and the node importance. `_transform` is
    tabicl's `_transform`, line for line, with those three replaced by the cached values.

    Extras: `out_scale` multiplies the node's output (1.0 = tabicl; the target graph sets it on private roots,
    a difficulty-match knob). `latent_dims_range`, if set on a featureless root, replaces tabicl's LogInt(1, 32)
    draw of its number of dims (another difficulty-match knob for private roots; None = tabicl). With `keep_writeback = True` the node stores, per extracted column, the block the
    converter wrote back into the node (`writeback_`; used by the exactness test).
    """

    def __init__(self, context: Context, feature_specs: Dict[str, FeatureSpec]):
        super().__init__(context, feature_specs)
        self.out_scale = 1.0
        self.latent_dims_range: Optional[Tuple[int, int]] = None   # roots only: override LogInt(1, 32)
        self.keep_writeback = False
        self.writeback_: Dict[str, torch.Tensor] = {}

    def _fit(self, x: List[torch.Tensor], n_samples: int):
        super()._fit(x, n_samples)
        if self.latent_dims_range is not None and len(x) == 0:
            # A root without features has n_features_ = its latent dims only; redraw them from the override range
            # (same sampler name as tabicl's draw). Used for the target graph's private roots (difficulty knob).
            assert not self.converters_, "the latent-dims override is for featureless roots"
            lo, hi = self.latent_dims_range
            self.n_features_ = self.sampler.randint("node_n_latent_features", lo, hi, use_log=True)
        if self.config.add_gaussian_noise:
            raise NotImplementedError("node noise would also need its sigma cached; it is off in every v2 recipe")
        self.root_law_ = RootPointLaw(self.context, self.n_features_) if len(x) == 0 else None
        if self.config.use_node_feature_importance:
            w = self.random_weights_.sample(1, self.n_features_).squeeze(0)
            self.feature_weights_ = w / w.square().mean().sqrt()
        if self.config.use_node_importance:
            self.node_importance_ = self.sampler.numerical("node_importance", low=0.1, high=10.0, use_log=True)

    def _transform(self, x: List[torch.Tensor], n_samples: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # ---- tabicl's RandomNodeFunction._transform, with cached task-level draws ----
        if len(x) == 0:
            x = self.root_law_.sample(n_samples)            # tabicl: self.random_points_.sample(...)
        else:
            x = self.random_func_(x)

        if self.config.use_node_feature_importance:
            x = self.std_(x)
            x = x * self.feature_weights_[None, :]          # tabicl: weights resampled here
        if self.config.use_node_l2_norm:
            x = self.l2_norm_(x)

        out_features = dict()
        start_idx = 0
        for name, converter in self.converters_.items():
            end_idx = start_idx + converter.get_n_features()
            assert end_idx <= self.n_features_
            tens, feat = converter(x[:, start_idx:end_idx])
            x[:, start_idx:end_idx] = tens                  # the write-back: what children of this node read
            if self.keep_writeback:
                self.writeback_[name] = x[:, start_idx:end_idx].clone()
            out_features[name] = feat
            start_idx = end_idx

        if self.config.use_node_importance:
            x = x * self.node_importance_                   # tabicl: node_importance resampled here
        # ---- end of tabicl's body ----
        if self.out_scale != 1.0:
            x = x * self.out_scale
        return x, out_features


class CachedGraphFunction(RandomGraphFunction):
    """tabicl's `RandomGraphFunction` (graph_lib/_graph_function.py) built from `CachedNodeFunction`s.

    `_fit` is tabicl's, copied, with the node class swapped; it also keeps tabicl's pruning (`should_compute_`:
    only nodes that host a feature, and their ancestors, are evaluated). `_transform` is inherited unchanged.
    """

    def _fit(self, n_samples: int):
        self.nodes_ = [CachedNodeFunction(self.context, feature_specs=specs) for specs in self.node_feature_specs]
        self.should_compute_ = [False for _ in range(len(self.nodes_))]
        for node_idx in reversed(range(len(self.nodes_))):
            if len(self.node_feature_specs[node_idx]) >= 1:
                self.should_compute_[node_idx] = True
            if self.should_compute_[node_idx]:
                for parent in self.dag[node_idx]:
                    self.should_compute_[parent] = True


def set_tape(laws: Sequence[RootPointLaw], mode: Optional[str]):
    """Set the tape mode of every root law (tests only); "record" also clears the tapes."""
    for law in laws:
        law.tape_mode = mode
        if mode == "record":
            law.tape = []
