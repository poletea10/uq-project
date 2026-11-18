"""The trainable parts of iLTMv2 Phase 1: the two scalar heads and arm B's fork (phase1_design.md Architecture §1
and §4; protocol §3 step 7, §4.5; transfer guide §3.3, §3.6).

    softplus_inv     the inverse of softplus, for the output-bias initialization
    Trunk            one head's MLP: Linear(512 -> 1024) -> GELU -> Linear(1024 -> 1) -> softplus; the last layer is
                     zero-initialized and its bias set to softplus^-1(mean target), so at step 0 every query gets
                     the same output, the mean target: the constant baseline of the reference lines (protocol §8.1)
    Head             one scalar head (a Trunk on a 512-d input)
    HeadPair         the A head and the E head of one configuration (arm, lr, p): Â = A(input), Ê_dir = E(input)
    Fork             arm B: the trainable copy of the last k TFicl blocks and the final LN (shared by both heads)

THE TWO ARMS
    A   the heads read h_star, the final LayerNorm output the quantile decoder reads (frozen backbone)
    B   the heads read the output of a trainable fork of the last 2 blocks + LN (the originals keep feeding the
        decoder, so the PPD is unchanged)

INDEPENDENCE OF THE HEADS
    In A the two heads share no parameter (design Training §3: the best Â and the best Ê_dir are chosen separately
    and may come from different configurations). In B both heads read the one fork, by design. Each HeadPair is
    trained by its own optimizer (`make_optimizer`), so stepping one pair never moves another (protocol §4.5;
    checked in scripts/00e_test_heads.py). An A job's pairs are head LR x loss exponent p (transfer guide §4).

INITIALIZATION AND PAIRING
    The trunks' first layers are drawn under `seeded(seed, INIT_KEY, head)`: the same job seed gives the same
    initialization to every learning rate and p (the comparisons are paired), and A and B share it (both read a
    512-d vector shaped like h_star), so a B job starts exactly where the A heads of the same seed start.
    The last layer is zero (weights) and softplus^-1(mean target) (bias), set by `HeadPair.init_output_bias`.
    With a zero last layer, at step 0 the gradient reaches only that layer: everything upstream (first trunk layer,
    fork) gets an exactly zero gradient until the first optimizer step makes the last layer nonzero.

UNITS: TabICLv2's internal z-normalized y units, the units of every target (design Architecture §1).
"""
from __future__ import annotations

import contextlib
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

ARMS = ("A", "B")
HEADS = ("A", "E")
D_MODEL = 512                   # width of h_star and of the residual stream (TFicl)
HIDDEN = 1024                   # mirrors the quantile decoder's 512 -> 1024 -> 999 (TabICLv2 App. A.4)
MIN_TARGET = 1e-6               # floor for the bias initialization: softplus^-1 needs a positive value
INIT_KEY = 1                    # the trunks' seeding key, shared by A and B (the first round's A1 / B key)


def fmt_lr(lr: float) -> str:
    """3e-4 -> '3e-4', 1e-3 -> '1e-3', 2.5e-4 -> '2.5e-4' (the head ids of protocol §7.1, e.g. 'A_lr3e-4_p0')."""
    mant, exp = f"{lr:.1e}".split("e")              # 2 significant digits: 2.5e-4 stays 2.5e-4, not 3e-4
    mant = mant[:-2] if mant.endswith(".0") else mant
    return f"{mant}e{int(exp)}"


def fmt_p(p: float) -> str:
    """0.0 -> '0', 1.0 -> '1', 0.5 -> '0.5'."""
    return f"{p:g}"


def pair_id(arm: str, lr: float, p: float) -> str:
    """The default id of a pair: '<arm>_lr<LR>_p<P>' (transfer guide §4, e.g. 'A_lr1e-3_p1'). B jobs name their one
    pair 'B' instead (`HeadPair(name=...)`)."""
    return f"{arm}_lr{fmt_lr(lr)}_p{fmt_p(p)}"


@contextlib.contextmanager
def seeded(*key: int):
    """CPU torch RNG seeded from a SeedSequence of `key` inside the block, restored afterwards."""
    s = int(np.random.SeedSequence([int(k) for k in key]).generate_state(1)[0])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(s)
        yield


def softplus_inv(y) -> float:
    """x with softplus(x) = y, for y > 0 (y is floored at MIN_TARGET): x = y + log(-expm1(-y)), stable in float64
    for small y (x ~ log y) and large y (x ~ y)."""
    y = torch.as_tensor(y, dtype=torch.float64).clamp_min(MIN_TARGET)
    return float(y + torch.log(-torch.expm1(-y)))


class Trunk(nn.Module):
    """Linear(d_in -> hidden) -> GELU -> Linear(hidden -> 1) -> softplus. Input (..., d_in), output (...)."""

    def __init__(self, d_in: int = D_MODEL, hidden: int = HIDDEN):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    @torch.no_grad()
    def init_bias(self, mean_target: float):
        self.fc2.bias.fill_(softplus_inv(mean_target))

    def forward(self, h: Tensor) -> Tensor:
        return F.softplus(self.fc2(self.act(self.fc1(h)))).squeeze(-1)


class Head(nn.Module):
    """One scalar head: a Trunk on h (..., D) (h_star in A, the fork's output in B). The `trunk` attribute keeps the
    state-dict keys of the first round's A1 / B heads ("trunk.fc1.weight", ...), so their checkpoints still load."""

    def __init__(self, d: int = D_MODEL, hidden: int = HIDDEN):
        super().__init__()
        self.trunk = Trunk(d, hidden)

    def forward(self, h: Tensor) -> Tensor:
        return self.trunk(h)


class HeadPair(nn.Module):
    """The Â and Ê_dir heads of one configuration. `forward(h) -> (A_hat, E_hat)`, each (...), non-negative.

    arm     "A" (input h_star) or "B" (input the fork's output)
    lr      the stage-1 max LR of this pair (stages 2-3 divide it by 8 and 40: design Training §2)
    p       the loss exponent of this pair: w = (e(q) + c)^-p (transfer guide §3.6)
    seed    the job seed: fixes the trunks' initialization (module docstring)
    name    a fixed id instead of 'A_lr<lr>_p<p>' (B jobs use 'B', so all B runs share their wandb keys and are told
            apart by their config)
    """

    def __init__(self, arm: str, lr: float, seed: int, p: float = 0.0, d: int = D_MODEL, hidden: int = HIDDEN,
                 name: Optional[str] = None):
        super().__init__()
        assert arm in ARMS, arm
        self.arm, self.lr, self.p, self.seed, self.name = arm, float(lr), float(p), int(seed), name
        heads = {}
        for i, h in enumerate(HEADS):
            with seeded(seed, INIT_KEY, i):
                heads[h] = Head(d, hidden)
        self.A, self.E = heads["A"], heads["E"]

    @property
    def head_id(self) -> str:
        return self.name or pair_id(self.arm, self.lr, self.p)

    @property
    def weighted(self) -> bool:
        """True if this pair trains with the weighted loss (p > 0): its best checkpoint is chosen on the weighted
        validation MSE (transfer guide §3.10, step 1)."""
        return self.p > 0

    def head(self, h: str) -> Head:
        return self.A if h == "A" else self.E

    def init_output_bias(self, mean_t_A: float, mean_t_E: float):
        """Zero last layer (already) + bias softplus^-1(mean target): at step 0 every query outputs the mean target
        (floored at MIN_TARGET, since a mean t_E can be <= 0 on a small batch)."""
        self.A.trunk.init_bias(mean_t_A)
        self.E.trunk.init_bias(mean_t_E)

    def forward(self, h: Tensor) -> Tuple[Tensor, Tensor]:
        return self.A(h), self.E(h)

    def make_optimizer(self, lr: Optional[float] = None, weight_decay: float = 1e-5,
                       betas=(0.9, 0.999)) -> torch.optim.AdamW:
        """This pair's own AdamW (design Training §2, DecoupledPFN App. B.3). Never share one across pairs."""
        return torch.optim.AdamW(self.parameters(), lr=self.lr if lr is None else lr, weight_decay=weight_decay,
                                 betas=betas)


class Fork(nn.Module):
    """Arm B (design Architecture §4): trainable copies of TFicl blocks 13-k..12 and the final LN, initialized with
    the pretrained weights (`FrozenBackbone.copy_last_blocks`). Input: the full residual stream after block 12-k,
    ALL rows (`BackboneOutput.fork_input`, the query attends to the context in every block); output: the copy's
    LN output at the query rows, (B, m, D), which feeds both heads. At step 0 it equals h_star. The originals stay
    frozen and keep feeding the decoder, so the PPD is untouched. `recompute` = gradient checkpointing (needed above
    about 20K rows, design Training §2)."""

    def __init__(self, backbone, k: int = 2):
        super().__init__()
        self.k = k
        self.blocks, self.ln = backbone.copy_last_blocks(k)
        self._backbone = backbone          # plain attribute: FrozenBackbone is not an nn.Module, never registered
        self._theta0 = [p.detach().clone() for p in self.parameters()]

    def forward(self, fork_input: Tensor, train_size: int, recompute: bool = False) -> Tensor:
        return self._backbone.run_blocks(self.blocks, self.ln, fork_input, train_size, recompute=recompute)

    @torch.no_grad()
    def distance_from_init(self) -> float:
        """||theta_fork - theta_fork,0|| (protocol §7.3, B)."""
        return float(torch.sqrt(sum(((p - p0) ** 2).sum() for p, p0 in zip(self.parameters(), self._theta0))))


def build_a_job_heads(seed: int, head_lrs: Sequence[float] = (3e-4, 1e-3, 3e-3),
                      ps: Sequence[float] = (0.0, 1.0)) -> List[HeadPair]:
    """The head pairs of one A job (transfer guide §4): every head LR x loss exponent p, on one seed. Ids are
    A_lr<LR>_p<P>; a repeated (LR, p) gets a version suffix (A_lr1e-3_p1, A_lr1e-3_p1_v1, ...)."""
    pairs, seen = [], {}
    for p in ps:
        for lr in head_lrs:
            base = pair_id("A", lr, p)
            k = seen.get(base, 0)
            seen[base] = k + 1
            # A repeated (head LR, p), or two LRs that format alike: later copies are versioned, base_v1, base_v2, ...
            # (the trainer, the best files and the logs are keyed by the id, so every id must be unique)
            pairs.append(HeadPair("A", lr, seed, p=p, name=base if k == 0 else f"{base}_v{k}"))
    ids = [x.head_id for x in pairs]
    assert len(set(ids)) == len(ids), f"head ids are not unique: {ids}"
    return pairs
