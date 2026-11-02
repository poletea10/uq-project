"""The observed-cut prior: TabICLv2's graph prior with exact replicates of y (proposal §4.3, phase1_design.md Training §1).

WHY A CUT IS NEEDED
    The heads are trained on t_A and t_E, which need R exact replicates y^(1..R) of a query's target at fixed x:
    i.i.d. draws from the task law p_tau(y | x). In TabICLv2's prior (`graph_scm`) this is not available. Random
    vectors at root nodes U are propagated through a random DAG; X reads some node dimensions, y reads another, and
    the unread dimensions are the only noise. Write X = Phi(U), y = Psi(U, V). A replicate would first need U from
    the abduction posterior p(U | Phi(U) = x), which for these dense nonlinear graphs has no tractable form.

THE CUT
    Keep the feature generator and make the observed X the only frozen-side input to the target branch:

        X = Phi(U1)   (graph_scm's X-graph),     y = Psi(X_obs, U2),     U1 ⟂ U2.

    Nothing has to be inferred from X: p(y | X = x) is the law of Psi(x, U2), so a replicate is "hold X, redraw U2,
    rerun Psi". Psi copies graph_scm as closely as possible:

    U1 ─► X-graph (graph_scm, cached nodes) ─► X columns ─► clamp / standardize / permute ─► X (what the model sees)
                         │ host node h: the columns read from h
                         ▼
            observed write-back W_h(x̃) ─┐            (x̃ = the clamped observed column, never a latent value)
            ‖ hidden dims (U2, 1–31)   ─┴─► X-input node ─┐
            private roots (U2) ─────────────────────────────┼─► target DAG (same sampler, 2–32 nodes) ─► y node ─► y

    - Target DAG: sampled exactly like the X-graph's DAG (`RandomCauchyDAG`, `min/max_n_nodes`, Cauchy offset),
      with the same node functions (`RandomNodeFunction`, here cached).
    - X-input nodes: a random subset of the X-graph's feature-hosting nodes (mirroring `subsample_feature_nodes`),
      placed on random DAG positions whose incoming edges are deleted (X is exogenous to Psi). Each holds the
      columns read from its host node, as the values the host's converters wrote back into it, plus fresh hidden
      dims. In graph_scm a node has d_i = sum_j d_ij + LogInt(1, 32) dims and converters read only the first part;
      the unread dims are the noise. Here the hidden dims play that role and belong to U2.
    - Private roots: every other parentless node draws random points, as graph_scm's roots do (also U2).
    - y: read from a random non-input node with tabicl's `NumericalConverter` (warping per `disallow_y_warping`).
    - Graph check (tabicl's `check_x_y_ancestors_overlap`, adapted): resample until an X-input node is an ancestor
      of y. Since X-input nodes carry hidden U2 dims, this also guarantees that private noise reaches y.

EXACTNESS RULE: OBSERVED VALUES ONLY
    Psi may depend on X only through what the model sees, otherwise y would depend on U1 beyond X and the abduction
    problem comes back. So an X-input node never receives the X-graph's latent node values. It receives W_h(x̃): the
    converter's write-back map evaluated on the observed column after the outlier clamp (`observed_writeback`).
      - On rows where the clamp does not bind, this is bit-identical to what the X-graph wrote back.
      - On clamped (> 4 sigma) rows, the raw write-back would carry information the clamped column has lost.
      - For the categorical `*_id` modes, the X-graph writes back the continuous latent, which the observed category
        does not determine; the observed category's center / embedding is used instead (the `*_disc` write-back).

WHAT IS AND IS NOT PRESERVED FROM graph_scm
    X keeps graph_scm's generating mechanism (DAG sampler, node functions, converters, warping, cross-feature
    correlations), but X's distribution is NOT identical to the pretraining prior's:
      1. The X-graph is sampled with no y node. In graph_scm y takes part in node assignment, pruning and the graph
         check, so removing it changes which graphs and column placements occur.
      2. The predictability filter runs on the cut y, so accepted X is conditioned on a different acceptance event.
      3. `ensure_iid=True` (proposal requirement), whereas TabICLv2's scripts ran with `ensure_iid=False`.
    Given up on the y side: latent confounding between X and y, and anti-causal features (X is never downstream of
    y). The difficulty-match gate (scripts/00f_difficulty_match.py) and the T3 transfer test measure the effect.

CACHING
    Both graphs are built from cached graph_lib pieces (`_cached_graph.py`): as shipped, a second call to a
    graph_lib graph function is a different task, so neither the X-graph's `ensure_iid` fit pass + data pass nor
    the target graph's replicates would be valid without them. The full explanation is in that module.

WHERE THINGS LIVE (the observed-cut prior is split over five modules of iltm2.prior)
    _cut_graph_scm.py   this file: configuration, the X-graph, the exactness rule, the target graph, task assembly
    _cached_graph.py    graph_lib's root law, node function and graph function with task-level draws cached
    _postprocess.py     fit/apply versions of tabicl's `outlier_removing` / `standard_scaling`, and g
    _filter.py          tabicl's task filters (graph check, constant columns, predictability, non-finite values)
    _dataset.py         task RNG, task shapes (pipeline steps 1-3), `sample_task`; later the training iterator

MAP OF THIS FILE (design Training §1 pipeline steps in brackets)
    CutGraphConfig, v2_prior_config     configuration (the checkpoint's prior flags + the cut's knobs)
    Host, XGraph                        [4] the X-graph and its feature-hosting nodes
    observed_writeback                  the exactness rule
    InputNodeFunction, TargetGraph      [5, 10] Psi(X_obs, U2): y^(0) and the replicates
    NonFiniteReplicate, CutGraphTask,   [4-7] task assembly: post-processing, filters and retries; one accepted
      CutGraphSCM                       task (its shape comes from `_dataset.sample_task_shape`)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from tabicl.prior.graph_lib._activation import L2Normalize, Standardize
from tabicl.prior.graph_lib._base import Context, DatasetProperties, FeatureSpec, PriorComponent, RandomTransformer
from tabicl.prior.graph_lib._config import PriorConfig
from tabicl.prior.graph_lib._converter import (CategoricalConverter, CategoricalNeighborDiscretizer,
                                               NumericalConverter)
from tabicl.prior.graph_lib._graph import RandomDAG
from tabicl.prior.graph_lib._properties import sample_categorical_sizes
from tabicl.prior.graph_lib._weights import RandomWeights

from iltm2.prior._cached_graph import CachedGraphFunction, CachedNodeFunction, RootPointLaw
from iltm2.prior._filter import (RejectionLog, all_finite, ancestors, drop_constant_columns, graph_check,
                                 predictability_rejects)
from iltm2.prior._postprocess import RobustClamp, Standardizer, YTransform

if TYPE_CHECKING:                                            # type hints only: _dataset imports this module
    from iltm2.prior._dataset import TaskShape


# =========================================================================================== configuration


def v2_prior_config(**overrides) -> PriorConfig:
    """The `PriorConfig` of the released TabICLv2 regression checkpoint, plus `ensure_iid=True`.

    Flags set in `scripts/train_v2_reg_stage{1,2,3}.sh` (identical across stages): `graph_noise False`,
    `filter_unpredictable_graphs True`, `filter_unpredictable_datasets True`, `allow_act_warping False`,
    `min_n_nodes 2`, `max_n_nodes 32`, `cauchy_dag_offset 0.0`. Every other flag is left at its default, which is
    also the `PriorConfig` dataclass default. `ensure_iid=True` is the cut's requirement (proposal §4.3): node
    transforms are fitted on a separate pass, so rows are conditionally i.i.d. given the task.
    """
    kw = dict(add_gaussian_noise=False, filter_unpredictable_graphs=True, filter_unpredictable_datasets=True,
              allow_act_warping=False, min_n_nodes=2, max_n_nodes=32, cauchy_dag_offset=0.0, ensure_iid=True)
    kw.update(overrides)
    return PriorConfig(**kw)


@dataclass
class CutGraphConfig:
    """Everything that defines the observed-cut prior. Defaults reproduce graph_scm wherever graph_scm has an
    analogue; the three target-graph knobs are the ones the difficulty-match gate may tune (never the filters)."""

    prior: PriorConfig = field(default_factory=v2_prior_config)
    min_features: int = 1                           # TabICLv2: --min_features 1
    max_features: int = 100                         # TabICLv2: --max_features 100
    query_mix: Tuple[float, float, float] = (0.70, 0.15, 0.15)   # [p_iid, p_hole, p_extrap], design §1.3
    region_max_frac: float = 0.2                    # upper bound of the tail size T and hole size k, as a share of N
    # Target-graph knobs (difficulty-match gate). (1, 32) is graph_scm's `node_n_latent_features` draw (1..31).
    hidden_dims_range: Tuple[int, int] = (1, 32)
    hidden_scale: float = 1.0                       # multiplies the hidden dims of X-input nodes
    private_root_scale: float = 1.0                 # multiplies the output of private roots
    max_task_attempts: int = 1000                   # safety cap on task-level retries
    max_graph_resamples: int = 10_000               # safety cap on target-graph resamples

    def __post_init__(self):
        assert abs(sum(self.query_mix) - 1.0) < 1e-9, "query_mix must sum to 1"
        assert self.prior.ensure_iid, "the cut needs ensure_iid=True (proposal §4.3)"
        assert not self.prior.add_gaussian_noise, "node noise is off in every v2 recipe and is not cached here"


# =========================================================================================== X-graph [4]


@dataclass
class Host:
    """A feature-hosting node of the X-graph: its index, the names of the columns it hosts (in converter order),
    and their converters. It becomes a candidate X-input node of the target graph."""
    node: int
    names: List[str]
    converters: List


class XGraph:
    """The X side of the cut: tabicl's `RandomDataset.sample` (graph_lib/_dataset.py) for the "x" group only.

    Copied steps: number of nodes `randint("n_nodes", min, max + 1, log)`, `RandomDAG`, feature placement on a
    random subset of nodes (`subsample_feature_nodes`), the graph function, and the `ensure_iid` fit pass. Changes:
    there is no y feature, so there is no graph check here (design §1.4: it must not run on the X-graph); the
    graph function is the cached one; and the fit pass's rows are returned, because the target graph uses them as
    its own fit pass (they are independent of the pool). Categorical cardinalities come from tabicl's
    `sample_categorical_sizes(..., max_cat_size=200)`, as in `GraphSCM.__call__`, unless given (tests).
    """

    def __init__(self, context: Context, num_features: int, cat_sizes: Optional[Sequence[int]] = None):
        cfg, sampler = context.config, context.sampler
        if cat_sizes is None:
            cat_sizes = sample_categorical_sizes(num_features, context, max_cat_size=200)
        assert len(cat_sizes) == num_features
        self.feature_specs = DatasetProperties(n_train=0, n_test=0, cat_sizes={"x": list(cat_sizes)}).feature_specs

        n_nodes = sampler.randint("n_nodes", cfg.min_n_nodes, cfg.max_n_nodes + 1, use_log=True)
        self.dag = RandomDAG(context).sample(n_nodes)
        if cfg.subsample_feature_nodes:
            n_feature_nodes = sampler.randint("n_feature_nodes", 1, n_nodes + 1)
            feature_nodes = np.random.permutation(n_nodes)[:n_feature_nodes]
        else:
            feature_nodes = np.arange(n_nodes)
        node_idx = np.random.choice(feature_nodes, replace=True, size=len(self.feature_specs))
        self.node_feature_specs = [dict() for _ in range(n_nodes)]
        for i, (name, spec) in enumerate(self.feature_specs.items()):
            self.node_feature_specs[node_idx[i]][name] = spec
        self.graph_fn = CachedGraphFunction(context, dag=self.dag, node_feature_specs=self.node_feature_specs)

        # Column order of GraphSCM: Dataset.get_concat_tensors groups "x_cat" then "x_num", and GraphSCM
        # concatenates [X_cat, X_num]. Within a group, the order of feature_specs.
        names = list(self.feature_specs)
        self.columns = ([n for n in names if self.feature_specs[n].cat_size > 1]
                        + [n for n in names if self.feature_specs[n].cat_size <= 1])
        self.col_index = {n: j for j, n in enumerate(self.columns)}

    def _run(self, n: int) -> torch.Tensor:
        tensors = self.graph_fn(n)
        return torch.cat([tensors[name].float() for name in self.columns], dim=-1)

    def sample(self, n_fit: int, n_data: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """(fit-pass columns (n_fit, d), data columns (n_data, d)): the raw columns before any post-processing."""
        assert not self.graph_fn.fitted, "an XGraph is sampled once"
        return self._run(n_fit), self._run(n_data)

    @property
    def hosts(self) -> List[Host]:
        # Nodes that host no feature and feed none are pruned by tabicl and never fitted (no `converters_`);
        # a node that hosts a feature is always computed.
        assert self.graph_fn.fitted
        return [Host(i, list(node.converters_), list(node.converters_.values()))
                for i, node in enumerate(self.graph_fn.nodes_) if len(self.node_feature_specs[i]) > 0]

    @property
    def n_nodes(self) -> int:
        return len(self.dag)

    def laws(self) -> List[RootPointLaw]:
        return [nd.root_law_ for nd, c in zip(self.graph_fn.nodes_, self.graph_fn.should_compute_)
                if c and nd.root_law_ is not None]


# =========================================================================================== exactness rule


_ID_MODES = ("neighbor_id", "softmax_id")
_TABLE_MODES = ("neighbor_id", "softmax_id", "neighbor_disc", "softmax_disc")
_INT_MODES = ("neighbor_int", "softmax_int")


def observed_writeback(converter, x_obs: torch.Tensor, prior_config: PriorConfig) -> torch.Tensor:
    """The block an X-input node holds for one column: the converter's write-back map applied to the OBSERVED value.

    x_obs: (n, 1), the column after the outlier clamp (before standardization and permutation, which are
    bijective). Returns (n, converter.get_n_features()), the width the column occupies in its host node.

    In the X-graph, a converter maps its slice of the node to (write-back, column): the column goes into the
    dataset, the write-back replaces the slice and is what the node's children read. The target graph must see a
    function of the column only (module docstring, "exactness rule"), so the write-back is recomputed from x_obs:

      NumericalConverter, use_corrected_num_converters = False (as in v2):
        "id"      returns (x, x)                -> write-back = column = x       -> x_obs
        "kumar"   returns (tfm(x), x)           -> write-back = column = tfm(x)  -> x_obs
                  The returned column x is a VIEW of the node slice, and RandomNodeFunction then writes tfm(x) into
                  that slice in place, so the column the dataset gets is tfm(x) too (tabicl's comment calls this the
                  "wrong version"; the aliasing is what actually ships). The column IS the write-back: identity.
        corrected converters (write-back x, column tfm(x)) are not used by v2 and raise NotImplementedError: they
        would need the inverse warping. "act" warping is off in every v2 recipe and raises as well.
      CategoricalConverter (column = category index; idx = x_obs rounded, clipped to a valid category):
        "*_disc"        write-back = the category's center / embedding  -> centers_[idx] / embeddings_[idx]
        "neighbor_func" write-back = f(center)                          -> post_func(centers_[idx])
        "*_int"         write-back = idx as float, broadcast            -> idx.float() over the block
        "*_id"          write-back = the continuous LATENT slice, which the category does not determine
                        -> substituted by the category's center / embedding (the "*_disc" write-back)
    On rows where the clamp does not bind, every mode except "*_id" reproduces the X-graph's write-back exactly.
    """
    if isinstance(converter, NumericalConverter):
        if converter.mode_ not in ("id", "kumar"):
            raise NotImplementedError(f"numerical converter mode {converter.mode_!r} is not used by v2")
        if converter.mode_ == "kumar" and prior_config.use_corrected_num_converters:
            raise NotImplementedError("corrected num converters write back x and output tfm(x): needs tfm^-1")
        return x_obs                                         # the column is the write-back (see docstring)
    if isinstance(converter, CategoricalConverter):
        disc = converter.disc
        table = disc.centers_ if isinstance(disc, CategoricalNeighborDiscretizer) else disc.embeddings_
        idx = x_obs[:, 0].round().clamp(0, table.shape[0] - 1).long()
        if converter.mode in _TABLE_MODES:
            return table[idx]
        if converter.mode == "neighbor_func":
            return converter.post_func(table[idx])
        if converter.mode in _INT_MODES:
            return idx.float()[:, None].expand(-1, converter.get_n_features()).clone()
        raise ValueError(f"unknown categorical mode {converter.mode!r}")
    raise TypeError(f"unknown converter {type(converter).__name__}")


# =========================================================================================== target graph [5, 10]


class InputNodeFunction(RandomTransformer):
    """An X-input node of the target graph: [observed write-back block ‖ hidden dims] x node importance.

    Mirrors what the children of a graph_scm feature-hosting node read: the converter slices hold the written-back
    values as they are (not re-standardized), the remaining LogInt(1, 32) dims are latent values that went through
    `Standardize`, the feature-importance weights and `L2Normalize`, and the whole node is multiplied by
    `node_importance`. Here the latent dims are fresh random points (a cached `RootPointLaw`): they are part of U2,
    redrawn on every call, and play the role of the host node's unread dims. Every task-level draw is cached in
    `_fit` (fitted on the `ensure_iid` fit pass).
    """

    def __init__(self, context: Context, hidden_dims_range: Tuple[int, int], hidden_scale: float):
        super().__init__(context)
        lo, hi = hidden_dims_range
        self.n_hidden = self.sampler.randint("node_n_latent_features", lo, hi, use_log=True)  # graph_scm's draw
        self.hidden_scale = hidden_scale

    def _fit(self, obs: torch.Tensor, n_samples: int):
        self.hidden_law_ = RootPointLaw(self.context, self.n_hidden)
        self.std_ = Standardize(self.context)
        self.l2_norm_ = L2Normalize(self.context)
        if self.config.use_node_feature_importance:
            w = RandomWeights(self.context).sample(1, self.n_hidden).squeeze(0)
            self.feature_weights_ = w / w.square().mean().sqrt()
        if self.config.use_node_importance:
            self.node_importance_ = self.sampler.numerical("node_importance", low=0.1, high=10.0, use_log=True)

    def _transform(self, obs: torch.Tensor, n_samples: int) -> torch.Tensor:
        h = self.hidden_law_.sample(n_samples)              # U2: fresh on every call
        if self.config.use_node_feature_importance:
            h = self.std_(h) * self.feature_weights_[None, :]
        if self.config.use_node_l2_norm:
            h = self.l2_norm_(h)
        x = torch.cat([obs, self.hidden_scale * h], dim=-1)
        if self.config.use_node_importance:
            x = x * self.node_importance_
        return x


class TargetGraph(PriorComponent):
    """Psi(X_obs, U2): the target graph of the cut (module docstring, "the cut").

    Construction (resampled until the graph check passes; each resample is counted in `log.graph_resamples`):
      1. n_nodes and the DAG exactly as the X-graph: `randint("n_nodes", min, max + 1, log)`, `RandomDAG`. The
         task's `GlobalSampler` is shared with the X-graph, as all nodes of a graph_scm task share one sampler.
      2. n_in ~ randint("cut_n_input_nodes", 1, min(#hosts, n_nodes - 1) + 1) X-input nodes, mirroring
         `subsample_feature_nodes`: n_in random hosts, placed on n_in random positions whose incoming edges are
         deleted. At least one node stays non-input, for y.
      3. y on a random non-input node (graph_scm places y on a random node too).
      4. Graph check: an X-input node must be an ancestor of y (`graph_check`).
    Then the node functions: `InputNodeFunction` at input positions, a `CachedNodeFunction` with the y spec at the
    y node, plain `CachedNodeFunction`s elsewhere; parentless non-input nodes are the private roots (U2).
    Pruning as tabicl's: only ancestors of y are evaluated.

    Use: `fit(inputs_fit, n_fit)` once (the `ensure_iid` fit pass), then `sample(inputs, n, R)`.
    """

    def __init__(self, context: Context, hosts: List[Host], cfg: CutGraphConfig, log: RejectionLog):
        super().__init__(context)
        pc = self.config
        for _ in range(cfg.max_graph_resamples):
            n_nodes = self.sampler.randint("n_nodes", pc.min_n_nodes, pc.max_n_nodes + 1, use_log=True)
            dag = RandomDAG(self.context).sample(n_nodes)
            n_in = self.sampler.randint("cut_n_input_nodes", 1, min(len(hosts), n_nodes - 1) + 1)
            host_pick = np.random.permutation(len(hosts))[:n_in]
            positions = [int(p) for p in np.random.permutation(n_nodes)[:n_in]]
            for p in positions:
                dag[p] = []                                  # X is exogenous to Psi
            non_input = [i for i in range(n_nodes) if i not in positions]
            y_node = int(np.random.choice(non_input))
            if graph_check(dag, positions, y_node):
                break
            log.graph_resamples += 1
        else:
            raise RuntimeError("target graph: graph check failed max_graph_resamples times")

        self.dag, self.y_node = dag, y_node
        self.input_hosts: Dict[int, Host] = {p: hosts[h] for p, h in zip(positions, host_pick)}
        self.nodes_: List = []
        for i in range(n_nodes):
            if i in self.input_hosts:
                node = InputNodeFunction(self.context, cfg.hidden_dims_range, cfg.hidden_scale)
            else:
                specs = {"y": FeatureSpec(group="y", cat_size=0)} if i == y_node else {}
                node = CachedNodeFunction(self.context, feature_specs=specs)
                if not dag[i]:
                    node.out_scale = cfg.private_root_scale  # a private root
            self.nodes_.append(node)
        self.needed_ = ancestors(dag, y_node)                # tabicl's pruning, for a graph whose only feature is y
        assert any(p in self.needed_ for p in self.input_hosts), "graph check passed but no X input reaches y"
        self.fitted_ = False

    # ----- inputs: the exactness rule

    def observed_inputs(self, x_clamped: torch.Tensor, col_index: Dict[str, int]) -> Dict[int, torch.Tensor]:
        """Per needed X-input position, the observed write-back block of its host's columns.

        x_clamped: (n, d_raw), the X-graph's columns after the pool's outlier clamp, in `XGraph.columns` order.
        """
        out = {}
        for p, host in self.input_hosts.items():
            if p in self.needed_:
                out[p] = torch.cat([observed_writeback(conv, x_clamped[:, col_index[name]:col_index[name] + 1],
                                                       self.config)
                                    for name, conv in zip(host.names, host.converters)], dim=-1)
        return out

    # ----- evaluation

    def _run(self, inputs: Dict[int, torch.Tensor], n: int) -> torch.Tensor:
        values: List[Optional[torch.Tensor]] = [None] * len(self.dag)
        y = None
        for i in range(len(self.dag)):                       # parents always have lower indices (RandomCauchyDAG)
            if i not in self.needed_:
                continue
            if i in self.input_hosts:
                values[i] = self.nodes_[i](inputs[i], n)
            else:
                values[i], feats = self.nodes_[i]([values[p] for p in self.dag[i]], n)
                if i == self.y_node:
                    y = feats["y"][:, 0]
        return y

    def fit(self, inputs_fit: Dict[int, torch.Tensor], n_fit: int):
        """The `ensure_iid` fit pass: every node fits its transforms on these rows (with their own U2); the output
        is discarded. Must run once, before any `sample`."""
        assert not self.fitted_
        self._run(inputs_fit, n_fit)
        self.fitted_ = True

    def sample(self, inputs: Dict[int, torch.Tensor], n: int, R: int = 1) -> torch.Tensor:
        """y (n, R): R i.i.d. draws of y at each of the n input rows, in ONE pass.

        The input rows are tiled R times (row r * n + i is replicate r of row i) and pushed through the graph with
        fresh U2 for every tiled row. All node functions act row-wise after fitting, so this equals R separate
        passes. The cut's only correctness guard (proposal §4.3): X must be bit-identical across replicates, so the
        tiled input blocks are snapshotted and must be unchanged after the pass (no node may modify X in place).
        """
        assert self.fitted_, "fit the target graph before sampling"
        tiled = {p: b.repeat(R, 1) for p, b in inputs.items()}
        snapshot = {p: t.clone() for p, t in tiled.items()}
        y = self._run(tiled, n * R)
        for p in tiled:
            assert torch.equal(tiled[p], snapshot[p]), "X changed during the replicate pass"
            assert torch.equal(tiled[p].view(R, n, -1), inputs[p].unsqueeze(0).expand(R, -1, -1)), \
                "replicates saw different X"
        return y.view(R, n).t().contiguous()

    # ----- tests / diagnostics

    def laws(self) -> List[RootPointLaw]:
        out = []
        for i in sorted(self.needed_):
            nd = self.nodes_[i]
            if isinstance(nd, InputNodeFunction):
                out.append(nd.hidden_law_)
            elif nd.root_law_ is not None:
                out.append(nd.root_law_)
        return out

    @property
    def n_private_roots_used(self) -> int:
        return sum(1 for i in self.needed_ if i not in self.input_hosts and not self.dag[i])


# =========================================================================================== tasks [7]


class NonFiniteReplicate(Exception):
    """A replicate is NaN or inf. The task must be rejected entirely (design §1.4), by the caller."""


@dataclass
class CutGraphTask:
    """One accepted observed-cut task: the pool of P rows before any split.

    X       (P, d)      what the model sees: clamped, standardized, permuted, constant columns removed
    y0      (P,)        g(y^(0)): the labels (context rows) and, for query rows, only g's fit and the filter input
    y0_raw  (P,)        y^(0) before g
    g                   the y transform, fitted on y0_raw
    x_clamped (P, d_raw) the X-graph's columns after the outlier clamp, in XGraph.columns order: the target graph's
                        input (bijective with X's columns, up to the dropped constant columns)
    x_clamp             the pool's outlier-clamp bounds (to apply the same clamp to other rows)
    """
    shape: TaskShape
    X: torch.Tensor
    y0: torch.Tensor
    y0_raw: torch.Tensor
    g: YTransform
    x_clamped: torch.Tensor
    x_clamp: RobustClamp
    x_graph: XGraph
    target: TargetGraph
    feature_perm: torch.Tensor
    kept_columns: torch.Tensor
    diagnostics: Dict[str, float]

    def replicates(self, rows, R: int) -> torch.Tensor:
        """Pipeline step 10: (len(rows), R) g-transformed i.i.d. draws of y at the given pool rows.

        Hold X, redraw U2, rerun the target graph, apply the pool's g. Draws from the global torch RNG: call it
        inside `task_rng(..., stream=STREAM_REPLICATES)`. Raises NonFiniteReplicate if any draw is NaN or inf.
        """
        rows = torch.as_tensor(rows, dtype=torch.long)
        X_before, xc_before = self.X.clone(), self.x_clamped.clone()
        inputs = self.target.observed_inputs(self.x_clamped[rows], self.x_graph.col_index)
        y_raw = self.target.sample(inputs, len(rows), R)
        assert torch.equal(self.X, X_before) and torch.equal(self.x_clamped, xc_before), "X changed"
        if not all_finite(y_raw):
            raise NonFiniteReplicate
        return self.g.apply(y_raw)


class CutGraphSCM:
    """Generates accepted observed-cut tasks (design Training §1, pipeline steps 4-7).

    One attempt (a rejection at any step discards everything and starts a new attempt, as tabicl's
    `GraphPrior.generate_dataset` loop does; each attempt has a fresh `Context`, i.e. a fresh `GlobalSampler`):
      4. X-graph: fit pass (P rows, kept as the target graph's fit rows) and data pass (the P pool rows).
         Reject non-finite X.
         Post-processing as `GraphSCM.__call__`: outlier clamp, standardization, feature permutation (constants
         fitted on the pool). Reject if the standardization would clip (it would make X non-injective).
         Constant columns: dropped; reject if none is left.
      5. Target graph: built (graph check inside), fitted on the fit rows, then one draw y^(0) per pool row.
         Reject non-finite y^(0).
      6. g fitted on the pool's y^(0) and applied to it.
      7. Predictability test on N uniformly drawn pool rows with g(y^(0)). Reject if it fails.
    """

    def __init__(self, cfg: Optional[CutGraphConfig] = None):
        self.cfg = cfg or CutGraphConfig()

    def attempt(self, shape: TaskShape, log: RejectionLog, cat_sizes: Optional[Sequence[int]] = None
                ) -> Optional[CutGraphTask]:
        cfg, P = self.cfg, shape.P
        context = Context(config=cfg.prior)

        # ---- 4. X pool
        x_graph = XGraph(context, shape.d, cat_sizes)
        raw_fit, raw = x_graph.sample(n_fit=P, n_data=P)
        if not all_finite(raw, raw_fit):
            log.reject("non_finite_x")
            return None
        clamp = RobustClamp.fit(raw)
        x_clamped = clamp.apply(raw)
        scaler = Standardizer.fit(x_clamped)
        if scaler.clips(x_clamped):
            log.reject("x_scaling_clip")
            return None
        perm = torch.randperm(shape.d)                       # GraphSCM: permute_features=True
        X = scaler.apply(x_clamped)[:, perm]
        kept = drop_constant_columns(X)
        if not kept.any():
            log.reject("constant_columns")
            return None
        X = X[:, kept]

        # ---- 5. target graph and y^(0)
        target = TargetGraph(context, x_graph.hosts, cfg, log)
        target.fit(target.observed_inputs(clamp.apply(raw_fit), x_graph.col_index), P)
        y0_raw = target.sample(target.observed_inputs(x_clamped, x_graph.col_index), P, R=1)[:, 0]
        if not all_finite(y0_raw):
            log.reject("non_finite_y")
            return None

        # ---- 6. g
        g = YTransform.fit(y0_raw)
        y0 = g.apply(y0_raw)

        # ---- 7. predictability test (TabICLv2 tests datasets of N rows)
        if predictability_rejects(X, y0, cfg.prior, n_rows=shape.N):
            log.reject("unpredictable")
            return None

        log.accepted += 1
        diagnostics = dict(n_x_nodes=x_graph.n_nodes, n_hosts=len(x_graph.hosts), n_y_nodes=len(target.dag),
                           n_inputs=len(target.input_hosts),
                           n_inputs_used=sum(p in target.needed_ for p in target.input_hosts),
                           y_ancestor_size=len(target.needed_), n_private_roots_used=target.n_private_roots_used,
                           d_raw=shape.d, d_kept=int(kept.sum()))
        return CutGraphTask(shape, X, y0, y0_raw, g, x_clamped, clamp, x_graph, target, perm, kept, diagnostics)

    def sample(self, shape: TaskShape, log: Optional[RejectionLog] = None,
               cat_sizes: Optional[Sequence[int]] = None) -> Tuple[CutGraphTask, RejectionLog]:
        """Attempts until one task is accepted. Returns the task and the log of this call's attempts."""
        log = log if log is not None else RejectionLog()
        for _ in range(self.cfg.max_task_attempts):
            task = self.attempt(shape, log, cat_sizes)
            if task is not None:
                return task, log
        raise RuntimeError(f"no task accepted in {self.cfg.max_task_attempts} attempts ({log.summary()})")
