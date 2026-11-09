"""The trainable parts of iLTMv2 Phase 1: the two scalar heads, A2's layer mix and B's fork
(phase1_design.md Architecture §1 and §4; protocol §3 step 7, §4.5).

    softplus_inv     the inverse of softplus, for the output-bias initialization
    Trunk            one head's MLP: Linear(512 -> 1024) -> GELU -> Linear(1024 -> 1) -> softplus; the last layer is
                     zero-initialized and its bias set to softplus^-1(mean target), so at step 0 every query gets
                     the same output, the mean target: the constant baseline of the reference lines (protocol §8.1)
    LayerMix         arm A2: sum_l alpha_l LN_l(r_l), alpha = softmax(s), one trainable LN per tapped layer
    Head             one scalar head = [LayerMix (A2 only)] + Trunk
    HeadPair         the A head and the E head of one configuration (arm, lr): Â = A(input), Ê_dir = E(input)
    Fork             arm B: the trainable copy of the last k TFicl blocks and the final LN (shared by both heads)

INDEPENDENCE OF THE HEADS
    In A1 and A2 the two heads share no parameter (design Training §3: the best Â and the best Ê_dir are chosen
    separately and may come from different configurations). So in A2 each head has its OWN LayerMix: a shared mix
    would let the noisier t_E gradients move the input of the Â trunk. In B both heads read the one fork, by design.
    Each HeadPair is trained by its own optimizer (`make_optimizer`), so stepping one pair never moves another
    (protocol §4.5; checked in scripts/00e_test_heads.py).

INITIALIZATION AND PAIRING
    The trunks' first layers are drawn under `seeded(seed, arm group, head)`: the same job seed gives the same
    initialization to every learning rate of an arm (the LR comparison is paired), and A1 and B share it (both read
    a 512-d vector shaped like h_star), so a B job starts exactly where the A1 heads of the same seed start.
    The last layer is zero (weights) and softplus^-1(mean target) (bias), set by `HeadPair.init_output_bias`.
    With a zero last layer, at step 0 the gradient reaches only that layer: everything upstream (first trunk layer,
    LayerMix, fork) gets an exactly zero gradient until the first optimizer step makes the last layer nonzero.

UNITS: TabICLv2's internal z-normalized y units, the units of every target (design Architecture §1).
"""
from __future__ import annotations

import contextlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

ARMS = ("A1", "A2", "B")
HEADS = ("A", "E")
D_MODEL = 512                   # width of h_star and of the residual stream (TFicl)
HIDDEN = 1024                   # mirrors the quantile decoder's 512 -> 1024 -> 999 (TabICLv2 App. A.4)
MIN_TARGET = 1e-6               # floor for the bias initialization: softplus^-1 needs a positive value
# Arms whose trunks get the same initialization for a given seed (A1 and B both read a 512-d h_star-like vector).
_INIT_GROUP = {"A1": 1, "B": 1, "A2": 2}


def fmt_lr(lr: float) -> str:
    """3e-4 -> '3e-4', 1e-3 -> '1e-3' (the head ids of protocol §7.1, e.g. 'A1_lr3e-4')."""
    mant, exp = f"{lr:.0e}".split("e")
    return f"{mant}e{int(exp)}"


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


class LayerMix(nn.Module):
    """Arm A2 (design Architecture §4): sum_l alpha_l LN_l(r_l) with alpha = softmax(s). Input (..., L, D) taps (raw
    pre-norm residual stream after each tapped block), output (..., D). alpha starts uniform (s = 0) and each LN at
    unit gain / zero bias. The alphas are a training diagnostic only, not a localisation of the information."""

    def __init__(self, n_layers: int, d: int = D_MODEL):
        super().__init__()
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(n_layers))
        self.logits = nn.Parameter(torch.zeros(n_layers))

    def alphas(self) -> Tensor:
        return torch.softmax(self.logits, dim=0)

    def forward(self, taps: Tensor) -> Tensor:
        assert taps.shape[-2] == len(self.norms), f"expected {len(self.norms)} taps, got {taps.shape[-2]}"
        normed = torch.stack([ln(taps[..., l, :]) for l, ln in enumerate(self.norms)], dim=-2)
        return (self.alphas()[:, None] * normed).sum(dim=-2)


class Head(nn.Module):
    """One scalar head: [LayerMix (A2)] + Trunk. Input: h (..., D) for A1 and B, taps (..., L, D) for A2."""

    def __init__(self, arm: str, d: int = D_MODEL, hidden: int = HIDDEN, n_layers: int = 12):
        super().__init__()
        self.mix = LayerMix(n_layers, d) if arm == "A2" else None
        self.trunk = Trunk(d, hidden)

    def forward(self, inp: Tensor) -> Tensor:
        return self.trunk(self.mix(inp) if self.mix is not None else inp)


class HeadPair(nn.Module):
    """The Â and Ê_dir heads of one configuration. `forward(inp) -> (A_hat, E_hat)`, each (...), non-negative.

    arm     "A1" (input h_star), "A2" (input taps), "B" (input the fork's output)
    lr      the stage-1 max LR of this pair (stages 2-3 divide it by 8 and 40: design Training §2)
    seed    the job seed: fixes the trunks' initialization (module docstring)
    """

    def __init__(self, arm: str, lr: float, seed: int, d: int = D_MODEL, hidden: int = HIDDEN, n_layers: int = 12):
        super().__init__()
        assert arm in ARMS, arm
        self.arm, self.lr, self.seed = arm, float(lr), int(seed)
        heads = {}
        for i, h in enumerate(HEADS):
            with seeded(seed, _INIT_GROUP[arm], i):
                heads[h] = Head(arm, d, hidden, n_layers)
        self.A, self.E = heads["A"], heads["E"]

    @property
    def head_id(self) -> str:
        return f"{self.arm}_lr{fmt_lr(self.lr)}"

    def head(self, h: str) -> Head:
        return self.A if h == "A" else self.E

    def init_output_bias(self, mean_t_A: float, mean_t_E: float):
        """Zero last layer (already) + bias softplus^-1(mean target): at step 0 every query outputs the mean target
        (floored at MIN_TARGET, since a mean t_E can be <= 0 on a small batch)."""
        self.A.trunk.init_bias(mean_t_A)
        self.E.trunk.init_bias(mean_t_E)

    def forward(self, inp: Tensor) -> Tuple[Tensor, Tensor]:
        return self.A(inp), self.E(inp)

    def alphas(self) -> Dict[str, Tensor]:
        """A2's softmax layer weights per head (empty for A1 and B)."""
        return {h: self.head(h).mix.alphas().detach() for h in HEADS if self.head(h).mix is not None}

    def make_optimizer(self, lr: Optional[float] = None, weight_decay: float = 1e-5,
                       betas=(0.9, 0.999)) -> torch.optim.AdamW:
        """This pair's own AdamW (design Training §2, DecoupledPFN App. B.3). Never share one across pairs."""
        return torch.optim.AdamW(self.parameters(), lr=self.lr if lr is None else lr, weight_decay=weight_decay,
                                 betas=betas)


def select_input(arm: str, h_star: Optional[Tensor] = None, taps: Optional[Tensor] = None,
                 fork_h: Optional[Tensor] = None) -> Tensor:
    """The tensor a HeadPair of `arm` reads: h_star (A1), the taps (A2), the fork's output (B)."""
    inp = {"A1": h_star, "A2": taps, "B": fork_h}[arm]
    assert inp is not None, f"arm {arm} needs {'h_star' if arm == 'A1' else 'taps' if arm == 'A2' else 'the fork'}"
    return inp


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


def build_a_job_heads(seed: int, head_lrs=(1e-4, 3e-4, 1e-3), n_layers: int = 12) -> List[HeadPair]:
    """The head pairs of one A job (design Training §5): every A1 and A2 configuration, on one seed."""
    return [HeadPair(arm, lr, seed, n_layers=n_layers) for arm in ("A1", "A2") for lr in head_lrs]
