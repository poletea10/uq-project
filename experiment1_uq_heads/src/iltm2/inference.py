"""Inference with a trained iLTMv2 model: TabICLv2's forecast plus the aleatoric (Â) and epistemic (Ê_dir) heads,
for any query set, from either arm (A, B) and any checkpoint (phase1_design.md Architecture §1, §4).

    from iltm2.inference import ILTMv2Regressor

    reg = ILTMv2Regressor.from_best("A")                  # the selected checkpoints of an arm (BEST, filled after the
                                                          # current round's selection, transfer guide §3.10)
    reg = ILTMv2Regressor.from_best("B", legacy=True)     # the first round's selected checkpoints (LEGACY_BEST)
    reg = ILTMv2Regressor("runs/A_carved_seed0/stage3/best/A_lr1e-3_p1_A.pt",
                          "runs/A_carved_seed0/stage3/best/A_lr1e-3_p1_E.pt")
    reg = ILTMv2Regressor("runs/A_carved_seed0/stage3/final.pt", pair="A_lr1e-3_p0")   # both heads of one pair
    reg = ILTMv2Regressor("runs/B_h3e-4_f3e-5_p1_carved_seed0/stage3/best/B_E.pt")     # B: fork + both heads
    reg.fit(X_train, y_train)                             # the context (raw units); nothing is trained
    reg.predict(X_test)                                   # the forecast's mean (like TabICLRegressor.predict)
    out = reg.predict_uncertainty(X_test)                 # {"quantiles", "median", "mean", "A", "E", "e_q"}

CHECKPOINTS (what `checkpoint_A` / `checkpoint_E` may point to)
    A best files         stage<s>/best/<pair>_A.pt and _E.pt: one head each, so both paths are needed (the best Â
                         and the best Ê_dir may come from different configurations, design Training §3)
    B best files         stage<s>/best/B_A.pt or B_E.pt: the whole model (fork + both heads); one path serves both
                         heads. Two different B files give each head its own fork.
    stage files          stage<s>/final.pt or latest.pt: every pair of the job; A jobs need `pair=` (e.g. A_lr1e-3_p1)
    A head's architecture (A = trunk on h_star, B = fork + trunk) is read from the checkpoint, and the backbone's
    sha256 must equal the one the checkpoint was trained on. The first round's checkpoints (under
    iltm2.paths.LEGACY_ROOT) load too: its arm A1 is arm A (same trunk, same state-dict keys) and its B is B. Its arm
    A2 (a layer mix of 12 residual taps) has no counterpart in this code; it needs the archived repository
    (transfer guide §6).

THE FORWARD ROUTE (design Architecture §1; protocol §1.3 "Forward route"): one raw TRAINING-MODE forward of the frozen
TabICLv2 regressor under no_grad, never eval mode and never TabICLRegressor. "Training mode" is only PyTorch's module
flag: nothing is trained, and dropout is 0. In eval mode tabicl switches to `_inference_forward`, a different path
(automatic batching, offloading, and mixed precision, use_amp=True by default); TabICLRegressor adds its own
preprocessing and an ensemble. Both give a different q_theta, while the heads' targets, the validation bank and every
validation number come from this exact fp32 forecast (Ê_dir is trained to ITS divergence). B's fork (tabicl blocks)
also runs in training mode, as it did in training and validation; the A heads have no mode-dependent layer. The rows are scaled as the training prior scaled them (`iltm2.prior._postprocess`), with the
constants fitted on the CONTEXT only (the residual caveat of design Training §1: at deployment only context labels
exist): per column, a robust clamp at mean +- 4 std then standardization for X, and the same for y.

UNITS: everything returned is in the raw units of y. The forecast's quantiles are mapped back affinely
(q * s_y + m_y); Â, Ê_dir and e(q) are CRPS-entropy / divergence values, positively 1-homogeneous, so they are
multiplied by s_y (design Architecture §1, "Units"), where (m_y, s_y) are the standardization constants of the
clamped context labels.

Queries never attend to each other (only to the context), so long query sets are processed in chunks of
`query_chunk` rows with the same result per query.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

from iltm2.model._backbone import DEFAULT_BACKBONE, FrozenBackbone
from iltm2.model._entropy import e_crps
from iltm2.model._heads import HEADS, Fork, Head
from iltm2.paths import LEGACY_ROOT
from iltm2.prior._postprocess import RobustClamp, Standardizer, YTransform

# The current round's selected checkpoints, (Â file, Ê_dir file) per arm, under iltm2.paths.RUNS_ROOT: filled after the selection
# of transfer guide §3.10 (step 1: learning rates per cell from validation; step 2: the final model from the tests).
BEST: Dict[str, Tuple[Path, Path]] = {}
# The first round's selected checkpoints (transfer guide §6): end of stage 3, first validation bank. Its A1 is arm A.
LEGACY_RUNS = LEGACY_ROOT / "runs"
LEGACY_BEST = {
    "A": (LEGACY_RUNS / "A_seed1/stage3/best/A1_lr1e-3_A.pt", LEGACY_RUNS / "A_seed0/stage3/best/A1_lr1e-3_E.pt"),
    # B: one configuration chosen as a whole (seed 0); its best-A and best-E files are the same fork + heads state.
    "B": (LEGACY_RUNS / "B_h3e-4_f3e-5_seed0/stage3/best/B_A.pt", LEGACY_RUNS / "B_h3e-4_f3e-5_seed0/stage3/best/B_E.pt"),
}
# Arm names in checkpoints: the first round's A1 is arm A; its A2 cannot be loaded here (module docstring).
_ARM_OF = {"A": "A", "A1": "A", "B": "B"}


def _arm(name: str, path) -> str:
    if name not in _ARM_OF:
        raise ValueError(f"{path}: arm {name!r} has no counterpart in this code (the first round's A2 needs the "
                         "archived repository /private/groups/ioannidislab/UQ_project_v1)")
    return _ARM_OF[name]


def _load(path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _sub_state(state: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


class _HeadSpec:
    """One head (h = "A" or "E") read from a checkpoint: its arm, its weights and, for B, its fork's weights."""

    def __init__(self, path, h: str, pair: Optional[str]):
        self.path, self.h = str(path), h
        ck = _load(path)
        self.sha = ck.get("backbone_sha256")
        self.fork_state, self.k = None, None
        if "head" in ck:                                           # A best file: one head
            assert ck["which"] == h, (f"{path} holds the {ck['which']} head, not {h}: pass the "
                                      f"{'_A' if h == 'A' else '_E'}.pt file")
            self.arm, self.state = _arm(ck["arm"], path), ck["head"]
        elif "pair" in ck and "fork" in ck:                        # B best file: fork + both heads
            self.arm, self.state, self.fork_state = "B", _sub_state(ck["pair"], f"{h}."), ck["fork"]
            self.k = int(ck["k"])
        elif "heads" in ck:                                        # stage file: latest.pt / final.pt
            if ck["arm"] == "B":
                self.arm, self.k = "B", int(ck["config"]["k"])
                self.state, self.fork_state = _sub_state(ck["heads"]["pair"], f"{h}."), ck["heads"]["fork"]
            else:
                assert pair is not None and pair in ck["heads"], (
                    f"{path} holds pairs {sorted(ck['heads'])}: choose one with pair=")
                self.arm, self.state = _arm(pair.split("_")[0], path), _sub_state(ck["heads"][pair], f"{h}.")
        else:
            raise ValueError(f"{path} is not an iLTMv2 Phase-1 checkpoint")

    def build(self, backbone: FrozenBackbone, device) -> Tuple[Head, Optional[Fork]]:
        head = Head()
        head.load_state_dict(self.state)
        fork = None
        if self.arm == "B":
            fork = Fork(backbone, k=self.k)
            fork.load_state_dict(self.fork_state)
            fork = fork.to(device).train()     # training mode, as in training and validation (module
                                               # docstring, THE FORWARD ROUTE)
        return head.to(device).eval(), fork


class ILTMv2Regressor:
    """sklearn-style iLTMv2 regressor (module docstring). `fit` stores and scales the context; nothing is trained.

    checkpoint_A   checkpoint of the aleatoric head (module docstring, CHECKPOINTS)
    checkpoint_E   checkpoint of the epistemic head (default: checkpoint_A, for files that hold both heads)
    pair           for an A job's stage file: which pair (e.g. "A_lr1e-3_p1")
    backbone_path  the frozen TabICLv2 regressor (default: the one the heads were trained on)
    device         "cuda" if available, else "cpu"
    query_chunk    queries per forward
    """

    def __init__(self, checkpoint_A, checkpoint_E=None, pair: Optional[str] = None,
                 backbone_path=DEFAULT_BACKBONE, device: Optional[str] = None, query_chunk: int = 4096):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.specs = {"A": _HeadSpec(checkpoint_A, "A", pair), "E": _HeadSpec(checkpoint_E or checkpoint_A, "E", pair)}
        ks = {s.k for s in self.specs.values() if s.arm == "B"}
        assert len(ks) <= 1, "the two B heads come from forks of different depths"
        self.backbone = FrozenBackbone(backbone_path, self.device, fork_k=ks.pop() if ks else None)
        for h, s in self.specs.items():
            assert s.sha is None or s.sha == self.backbone.sha256, (
                f"the {h} head ({s.path}) was trained on another backbone (sha256 {str(s.sha)[:12]}...)")
        self.heads: Dict[str, nn.Module] = {}
        self.forks: Dict[str, Optional[Fork]] = {}
        for h, s in self.specs.items():
            self.heads[h], self.forks[h] = s.build(self.backbone, self.device)
        self.query_chunk = int(query_chunk)
        self.arm = {h: s.arm for h, s in self.specs.items()}

    @classmethod
    def from_best(cls, arm: str, legacy: bool = False, **kw) -> "ILTMv2Regressor":
        """The selected checkpoints of an arm, "A" or "B": the current round's (BEST) or the first round's
        (LEGACY_BEST, legacy=True)."""
        table = LEGACY_BEST if legacy else BEST
        if arm not in table:
            raise KeyError(f"no selected {'legacy ' if legacy else ''}checkpoint for arm {arm!r}"
                           + ("" if legacy else " yet (BEST is filled after the selection; legacy=True loads the "
                                                "first round's)"))
        path_A, path_E = table[arm]
        return cls(path_A, path_E, **kw)

    # ----- sklearn-style API

    def fit(self, X, y) -> "ILTMv2Regressor":
        X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        y = torch.as_tensor(np.asarray(y, dtype=np.float32)).reshape(-1)
        assert X.ndim == 2 and X.shape[0] == y.shape[0] and X.shape[0] >= 2, "X (n, d) and y (n,), n >= 2"
        assert bool(torch.isfinite(X).all() and torch.isfinite(y).all()), "missing values are not supported"
        self.x_clamp = RobustClamp.fit(X)
        self.x_scale = Standardizer.fit(self.x_clamp.apply(X))
        self.y_tf = YTransform.fit(y)
        self.y_mean = float(self.y_tf.scale.mean.reshape(-1)[0])
        self.y_std = float(self.y_tf.scale.std.reshape(-1)[0])
        self.X_ctx = self._scale_X(X)
        self.y_ctx = self.y_tf.apply(y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X, output_type: str = "mean"):
        """output_type: "mean" or "median" of the forecast (n,), "quantiles" (n, 999), or "uncertainty" (dict)."""
        out = self.predict_uncertainty(X)
        if output_type == "uncertainty":
            return out
        assert output_type in ("mean", "median", "quantiles"), output_type
        return out[output_type]

    def predict_quantiles(self, X) -> np.ndarray:
        return self.predict_uncertainty(X)["quantiles"]

    @torch.no_grad()
    def predict_uncertainty(self, X) -> Dict[str, np.ndarray]:
        """Raw y units. quantiles (n, 999) at levels k/1000 (raw, possibly unsorted, as TabICLv2 returns them);
        median (n,), mean (n,) = the quantiles' average; A = Â (n,); E = Ê_dir (n,); e_q = e_CRPS(q_theta) (n,)."""
        assert hasattr(self, "X_ctx"), "call fit(X_train, y_train) first"
        X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        assert X.ndim == 2 and X.shape[1] == self.n_features_in_, f"X must be (n, {self.n_features_in_})"
        Xq = self._scale_X(X)
        parts = {k: [] for k in ("q", "A", "E", "e_q")}
        for lo in range(0, len(Xq), self.query_chunk):
            rows = torch.cat([self.X_ctx, Xq[lo:lo + self.query_chunk]])[None]
            out = self.backbone(rows, self.y_ctx[None])
            fork_h = {}
            for h in HEADS:
                if self.arm[h] == "A":
                    inp = out.h_star
                else:
                    key = id(self.forks[h])
                    if key not in fork_h:
                        fork_h[key] = self.forks[h](out.fork_input, out.train_size)
                    inp = fork_h[key]
                parts[h].append(self.heads[h](inp)[0].double().cpu())
            parts["q"].append(out.q[0].double().cpu())
            parts["e_q"].append(e_crps(out.q[0].double()).cpu())
        q = torch.cat(parts["q"]).numpy() * self.y_std + self.y_mean
        qs = np.sort(q, axis=1)
        return dict(quantiles=q, median=qs[:, qs.shape[1] // 2], mean=q.mean(axis=1),
                    A=torch.cat(parts["A"]).numpy() * self.y_std, E=torch.cat(parts["E"]).numpy() * self.y_std,
                    e_q=torch.cat(parts["e_q"]).numpy() * self.y_std)

    # ----- helpers

    def _scale_X(self, X: torch.Tensor) -> torch.Tensor:
        return self.x_scale.apply(self.x_clamp.apply(X)).to(self.device)

    def __repr__(self) -> str:
        return (f"ILTMv2Regressor(A: {self.arm['A']} {self.specs['A'].path}, E: {self.arm['E']} "
                f"{self.specs['E'].path}, device {self.device})")
