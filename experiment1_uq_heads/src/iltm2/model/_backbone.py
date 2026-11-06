"""The frozen TabICLv2 regressor and what the heads read from it (phase1_design.md Architecture §1, §4; protocol §3
step 5, §4.4).

THE FORWARD ROUTE (design Architecture §1)
    One raw training-mode forward of tabicl's own `TabICL` module, `model(X, y_ctx)` with d = None, the computation
    TabICLv2 pretraining runs (`train/_run.py: run_micro_batch`). Never `model.eval()`: in eval mode
    `ICLearning.forward` switches to `_inference_forward`, a different code path; dropout is 0, so training mode is
    deterministic. Never `TabICLRegressor`: its preprocessing and ensemble give a different q_theta. Everything runs
    under `torch.no_grad()`: no backbone activation is stored. This wrapper never re-implements any part of the
    forward: the quantiles come out of tabicl's code unchanged, and the extra tensors are read by forward hooks
    (protocol §1.1, "hooks instead of edits"), which only copy, never modify.

    Attention kernel: FlashAttention-3 (fp16 attention) is switched off (tabicl enables it whenever
    `flash_attn_interface` is importable; it is not installed in uq-iltm-bio and is Hopper-only). TabICLv2 was
    pretrained in float32 (`--dtype float32`), so every forward here is fp32 SDPA, with tabicl's training rule of
    excluding the cuDNN SDPA backend. One route for training, the bank and every Phase 0 evaluation.

WHAT IS READ (all at the query rows, positions train_size: of the N rows, unless stated)
    In `ICLearning._icl_predictions`: R[:, :n_ctx] += y_encoder(y_ctx); src = tf_icl(R, train_size) (12 blocks,
    `tf_icl.blocks[l](q=..., train_size=..., rope=tf_icl.rope)`); src = ln(src); out = decoder(src).
      q        (B, m, K)     the decoder's 999 quantiles: the frozen PPD q_theta (raw, unsorted)
      h_star   (B, m, D)     the output of the final LayerNorm `ln`, what the decoder reads (A1's input; D = 512)
      taps     (B, m, L, D)  the residual stream r_l = output of TFicl block l, l in `tap_layers` (1-based; A2's
                             input, all 12 by default). Pre-norm: raw stream, A2 applies its own LN per layer.
      fork_input (B, N, D)   r_{12-k} at ALL rows, context included (arm B's fork re-attends to the context)
    `run_blocks` + `copy_last_blocks` give arm B its trainable copy of the last k blocks and `ln`; at step 0 the
    copy maps fork_input to h_star (design §4, "Arm B"; tested in 00d). The Fork module itself is step 7
    (`_heads.py`).
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from tabicl._model.attention import flash_attn3_toggle
from tabicl._model.tabicl import TabICL

from iltm2.model._entropy import e_crps

# The released TabICLv2 regressor (also used by 00f and 02). protocol §2 plans a copy + sha256 under
# /private/groups/ioannidislab/UQ_decoupling_exp/backbone/; until it exists, this is the checkpoint.
DEFAULT_BACKBONE = "/private/groups/ioannidislab/ajaras/tabicl-dev/tabicl-regressor-v2-20260212.ckpt"


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def state_dict_sha256(module: nn.Module) -> str:
    """Hash of every tensor of a state dict (name, dtype, shape, bytes): any change to a weight changes it."""
    h = hashlib.sha256()
    for name, t in module.state_dict().items():
        t = t.detach().cpu().contiguous()
        h.update(f"{name}|{t.dtype}|{tuple(t.shape)}".encode())
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@contextlib.contextmanager
def attention_ctx(device_type: str):
    """FA3 off (fp32 SDPA, as in TabICLv2's stage-1 pretraining) and, on CUDA, tabicl's training-time SDPA backend
    list without cuDNN (`Trainer._sdpa_ctx`). Used by the backbone forward and by arm B's fork."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(flash_attn3_toggle(False))
        if device_type == "cuda" and hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            from torch.nn.attention import SDPBackend, sdpa_kernel
            stack.enter_context(sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                                             SDPBackend.MATH]))
        yield


def load_tabicl(path, device) -> TabICL:
    """`TabICL(**ckpt["config"])` with the checkpoint's weights, as the sklearn wrapper builds it; training mode."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = TabICL(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    assert model.max_classes == 0, "not a regression checkpoint"
    return model.to(device).train()


@dataclass
class BackboneOutput:
    """One forward of the frozen backbone on a micro-batch group (module docstring, "what is read")."""
    q: Tensor                       # (B, m, K) float32
    e_q: Tensor                     # (B, m) float64, e_CRPS(q_theta) under the single CDF convention (_entropy)
    h_star: Tensor                  # (B, m, D)
    taps: Optional[Tensor]          # (B, m, L, D) or None
    tap_layers: Tuple[int, ...]     # the 1-based block indices of `taps`
    fork_input: Optional[Tensor]    # (B, N, D) or None
    train_size: int


class FrozenBackbone:
    """TabICLv2 frozen, with forward hooks for h_star, the residual taps and the fork input.

    tap_layers  1-based TFicl blocks whose query-row output is returned in `taps` (None: no taps; "all": 1..12)
    fork_k      if set, the full residual stream after block n_blocks - fork_k is returned as `fork_input`
    Not an nn.Module on purpose: it owns no trainable parameter, so it can never end up in an optimizer.
    """

    def __init__(self, path: str = DEFAULT_BACKBONE, device="cpu", tap_layers="all", fork_k: Optional[int] = None):
        self.path, self.device = str(path), torch.device(device)
        self.sha256 = file_sha256(path)
        self.model = load_tabicl(path, self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        icl = self.model.icl_predictor
        assert icl.norm_first, "h_star is defined as the final LN output (pre-norm TFicl)"
        self.blocks: nn.ModuleList = icl.tf_icl.blocks
        self.ln: nn.LayerNorm = icl.ln
        self.decoder: nn.Module = icl.decoder
        self.rope = icl.tf_icl.rope
        self.n_blocks, self.d_model = len(self.blocks), self.ln.normalized_shape[0]
        self.tap_layers: Tuple[int, ...] = (tuple(range(1, self.n_blocks + 1)) if tap_layers == "all"
                                            else tuple(tap_layers or ()))
        assert all(1 <= l <= self.n_blocks for l in self.tap_layers)
        if fork_k is not None:
            assert 1 <= fork_k < self.n_blocks, "the fork copies the last k blocks, 1 <= k < 12"
        self.fork_k = fork_k
        self.fork_layer = None if fork_k is None else self.n_blocks - fork_k   # block whose output feeds the fork

        self._train_size: Optional[int] = None
        self._store: dict = {}
        self._hooks = [self.ln.register_forward_hook(self._ln_hook)]
        for l in sorted(set(self.tap_layers) | ({self.fork_layer} if self.fork_layer else set())):
            self._hooks.append(self.blocks[l - 1].register_forward_hook(partial(self._block_hook, l)))

    # ----- hooks: copy, never modify

    def _block_hook(self, layer: int, module, args, out):
        ts = self._train_size
        if ts is None:                                       # a forward not issued by this wrapper: ignore
            return
        if layer in self.tap_layers:
            self._store[("tap", layer)] = out[:, ts:].clone()
        if layer == self.fork_layer:
            self._store["fork"] = out.clone()

    def _ln_hook(self, module, args, out):
        if self._train_size is not None:
            self._store["h"] = out[:, self._train_size:].clone()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    # ----- the forward route

    @torch.no_grad()
    def __call__(self, X: Tensor, y_ctx: Tensor) -> BackboneOutput:
        """X (B, N, d) context rows first; y_ctx (B, n_ctx). Inputs are moved to the backbone's device."""
        assert self.model.training, "the forward route is the training-mode forward; never call model.eval()"
        X, y_ctx = X.to(self.device, torch.float32), y_ctx.to(self.device, torch.float32)
        self._train_size, self._store = y_ctx.shape[1], {}
        try:
            with attention_ctx(self.device.type):
                q = self.model(X, y_ctx)                     # tabicl's own forward; d = None as in v2 training
            store = self._store
        finally:
            self._train_size, self._store = None, {}
        assert q.shape[1] == X.shape[1] - y_ctx.shape[1]
        taps = (torch.stack([store[("tap", l)] for l in self.tap_layers], dim=2) if self.tap_layers else None)
        return BackboneOutput(q=q, e_q=e_crps(q.double()), h_star=store["h"], taps=taps,
                              tap_layers=self.tap_layers, fork_input=store.get("fork"), train_size=y_ctx.shape[1])

    # ----- arm B: a trainable copy of the last k blocks and the final LN

    def copy_last_blocks(self, k: int) -> Tuple[nn.ModuleList, nn.LayerNorm]:
        """Deep copies (fresh parameters, requires_grad=True, training mode) of TFicl blocks 13-k..12 and `ln`,
        initialized with the pretrained weights. The originals stay frozen and keep feeding the decoder."""
        blocks = copy.deepcopy(self.blocks[self.n_blocks - k:])
        ln = copy.deepcopy(self.ln)
        for m in (blocks, ln):
            m.train()
            for p in m.parameters():
                p.requires_grad_(True)
        return blocks, ln

    def run_blocks(self, blocks: Sequence[nn.Module], ln: nn.Module, stream: Tensor, train_size: int,
                   recompute: bool = False) -> Tensor:
        """ln(blocks(stream)) at the query rows, exactly as `Encoder.forward` + `ICLearning` run them (same
        train_size masking, same rope), with optional gradient checkpointing (tabicl's `recompute`). Gradients
        flow when called outside no_grad: this is arm B's forward."""
        out = stream
        with attention_ctx(stream.device.type):
            for block in blocks:
                if recompute:
                    out = checkpoint(partial(block, train_size=train_size, rope=self.rope), out, use_reentrant=False)
                else:
                    out = block(q=out, train_size=train_size, rope=self.rope)
            return ln(out)[:, train_size:]

    def frozen_parameters(self) -> List[nn.Parameter]:
        return list(self.model.parameters())

    def __repr__(self) -> str:
        return (f"FrozenBackbone({self.path}, sha256 {self.sha256[:12]}..., device {self.device}, "
                f"{self.n_blocks} TFicl blocks of width {self.d_model}, taps {self.tap_layers}, fork_k {self.fork_k})")
