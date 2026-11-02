"""Post-processing of the observed-cut prior as fit/apply pairs (proposal §4.3 "fit/apply", design Training §1 step 6).

WHY FIT/APPLY
    TabICLv2's prior post-processes every dataset with two functions from `prior/_reg2cls.py`, applied to X and to
    y in `GraphSCM.__call__`:
      outlier_removing(input, threshold=4)   a clamp: per column, bounds at mean +- 4 std, then mean and std
                                             recomputed inside those bounds, final bounds at their mean +- 4 std
      standard_scaling(input)                (x - mean) / std, clipped to +-100
    Both compute their constants from the very rows they transform. That is fine for one dataset, but the cut needs
    the SAME constants on other rows:
      - y's transform g is fitted once on the pool's y^(0) and must be applied unchanged to every replicate. A refit
        per replicate would make the transform itself random across draws, and the replicates would no longer be
        i.i.d. draws from one (clamped) task law. Clamped replicates ARE i.i.d. draws from the clamped law the
        backbone was pretrained on, so t_A stays unbiased for it; the targets themselves are never clipped.
      - X's clamp bounds, fitted on the pool, are applied to the X-graph's fit-pass rows (the target graph's
        `ensure_iid` fit pass) and give the clamped observed values x̃ the target graph reads.
    The classes below reproduce tabicl's statements line by line in `fit` and store the constants; `apply` uses
    them. So `apply(fit(v), v)` equals tabicl's function on v bit for bit (tested in scripts/00b_test_cut_prior.py,
    section 5), and the same constants can be applied to any other rows.

    RobustClamp    fit/apply version of `outlier_removing`
    Standardizer   fit/apply version of `standard_scaling` (+ `clips`, the check that X stays injective)
    YTransform     g = RobustClamp then Standardizer on y, fitted once on the pool's y^(0)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from tabicl.prior._reg2cls import torch_nanstd


# =========================================================================================== post-processing [6]


@dataclass
class RobustClamp:
    """Fit/apply version of tabicl's `outlier_removing` (prior/_reg2cls.py), the "outlier removal" of X and y.

    `outlier_removing` is a clamp: per column, bounds at mean +- 4 std, then the mean and std are recomputed on the
    values inside those bounds and the final bounds are their mean +- 4 std. `fit` reproduces those statements
    line by line and stores the final bounds; `apply` is the clamp. So `apply(fit(X), X)` equals
    `outlier_removing(X)` bit for bit, and the same bounds can be applied to other rows (replicates, fit rows).
    """
    lower: torch.Tensor
    upper: torch.Tensor

    @classmethod
    def fit(cls, input: torch.Tensor, threshold: float = 4.0) -> "RobustClamp":
        ddof = 1 if input.shape[0] > 1 else 0
        mean = torch.nanmean(input, dim=0)
        std = torch_nanstd(input, dim=0, ddof=ddof).clip(min=1e-6)
        cut_off = std * threshold
        lower, upper = mean - cut_off, mean + cut_off
        mask = (lower <= input) & (input <= upper) & ~torch.isnan(input)
        masked_input = torch.where(mask, input, torch.nan)
        masked_mean = torch.nanmean(masked_input, dim=0)
        masked_std = torch_nanstd(masked_input, dim=0, ddof=ddof).clip(min=1e-6)
        masked_mean = torch.where(torch.isnan(masked_mean), mean, masked_mean)
        masked_std = torch.where(torch.isnan(masked_std), torch.zeros_like(std), masked_std)
        cut_off = masked_std * threshold
        lower, upper = masked_mean - cut_off, masked_mean + cut_off
        return cls(torch.nan_to_num(lower, nan=-torch.inf), torch.nan_to_num(upper, nan=torch.inf))

    def apply(self, input: torch.Tensor) -> torch.Tensor:
        return input.clamp(min=self.lower, max=self.upper)


@dataclass
class Standardizer:
    """Fit/apply version of tabicl's `standard_scaling` (prior/_reg2cls.py): (x - mean) / std, clipped to +-100."""
    mean: torch.Tensor
    std: torch.Tensor
    clip_value: float = 100.0

    @classmethod
    def fit(cls, input: torch.Tensor) -> "Standardizer":
        mean = torch.nanmean(input, dim=0)
        std = torch_nanstd(input, dim=0, ddof=1 if input.shape[0] > 1 else 0).clip(min=1e-6)
        return cls(mean, std)

    def apply(self, input: torch.Tensor) -> torch.Tensor:
        return torch.clip((input - self.mean) / self.std, min=-self.clip_value, max=self.clip_value)

    def clips(self, input: torch.Tensor) -> bool:
        """True if `apply` would clip some value (the clip is the only non-injective step)."""
        return bool(((input - self.mean) / self.std).abs().gt(self.clip_value).any())


@dataclass
class YTransform:
    """g: TabICLv2's y post-processing (`outlier_removing` then `standard_scaling`, `GraphSCM.__call__`), split into
    fit and apply (design Training §1 step 6; proposal §4.3 "fit/apply").

    Fitted ONCE on the pool's y^(0), before any split, as TabICLv2 fits it on all rows of a dataset; then applied
    unchanged to y^(0) and to every replicate. Never refit per replicate (that would make the transform random
    across draws). Clamped replicates are i.i.d. draws from the clamped task law the backbone was pretrained on,
    so t_A stays unbiased for it; the targets themselves are never clipped.
    """
    clamp: RobustClamp
    scale: Standardizer

    @classmethod
    def fit(cls, y: torch.Tensor) -> "YTransform":
        y = y.reshape(-1, 1).float()
        clamp = RobustClamp.fit(y)
        return cls(clamp, Standardizer.fit(clamp.apply(y)))

    def apply(self, y: torch.Tensor) -> torch.Tensor:
        """Any shape; every entry is transformed with the same constants."""
        return self.scale.apply(self.clamp.apply(y.reshape(-1, 1).float())).reshape(y.shape)
