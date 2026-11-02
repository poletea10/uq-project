"""Gate for step 1 of the order of work (phase1_protocol.md §3, unit tests of §4.1): entropy kernels and targets.

WHAT IS TESTED
    iltm2.model._entropy   quantile_levels, cdf, e_crps, crps: the torch kernels that turn TabICLv2's 999 raw
                           quantiles into a CDF (one convention, design §2) and integrate its CRPS entropy
                           e(q_theta) and its CRPS S(q_theta, y) exactly. Used for the head targets, the loss
                           weight w = (e(q_theta) + c)^-p, the bank's reference lines and the A/e(q) ratio.
    iltm2.train._targets   t_aleatoric, t_epistemic, targets, half_split_targets: the replicate targets
                           t_A, t_E of design §2 and their half-split versions used by xMSE (protocol §7.3).

HOW
    The estimators themselves were validated in Phase 0 (phase0_protocol.md §10, gotcha 1), so they are not
    re-audited for bias here (one exception: section 4 checks the one property xMSE relies on). What is new is
    the torch implementation, and the reference is Phase 0's numpy code, imported from experiment0_baseline/src:
      - entropy.quantile_entropy: Phase 0's exact segment-wise e_CRPS. Both sides are exact closed forms, so
        they must agree to round-off.
      - forecasts.QuantileForecast.crps: Phase 0's CRPS on a fine numerical grid (Simpson's rule, split at y).
        It is an independent route to the same integral, so agreement "to about 1e-5" (protocol §4.1) shows the
        torch closed form integrates the right CDF. The grid's error scales with its spacing, i.e. with the
        span of the forecast and y, so that tolerance is applied per unit of span.
      - truth.replicate_aleatoric / truth.replicate_epistemic: Phase 0's t_A and t_E.
    Test forecasts are random quantile sets of two kinds: "smooth" (Gaussian quantiles, random location and
    scale) and "rough" (random increments with heavy variation and ~5% exactly tied quantiles, i.e. zero-width
    segments, which are jumps in the CDF). y values fall inside the quantile range, below it, above it, and
    exactly on knots, so every branch of crps is hit.

OUTPUT
    One "OK  <check>" or "FAIL <check>" line per check, with the worst error and its tolerance. Exit code 1 if
    any check failed.

    sbatch experiment1_uq_heads/scripts/00a_test_entropy.sh
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import ndtri

from iltm2.model._entropy import cdf, crps, e_crps, quantile_levels
from iltm2.train._targets import half_split_targets, t_aleatoric, t_epistemic, targets

# Phase 0's numpy reference code (flat modules, not a package).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiment0_baseline" / "src"))
import config as p0_config  # noqa: E402
import entropy as p0_entropy  # noqa: E402
import truth as p0_truth  # noqa: E402
from forecasts import QuantileForecast  # noqa: E402

K = 999                    # TabICLv2's number of quantiles
N_Q = 200                  # forecasts per test set
R = 8                      # replicates per forecast in the CRPS / t_E comparisons
GRID_TOL = 1e-5            # protocol §4.1: "agree with Phase 0's numpy kernel to about 1e-5", per unit of span
EXACT_RTOL = 1e-10         # two exact float64 closed forms: round-off only
FLOAT32_RTOL = 1e-4        # float32 vs float64 torch: what training in float32 costs

FAILS = []
PROTO = p0_config.protocol()          # Phase 0 grid settings (t_points, t_halfwidth_sd, tail_tol)
ALPHAS = np.linspace(0.0, 1.0, K + 2)[1:-1]
rng = np.random.default_rng(20261101)


def check(name, ok, detail=""):
    print(f"{'OK  ' if ok else 'FAIL'} {name}  {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def T(a):
    return torch.from_numpy(np.asarray(a, dtype=np.float64))


def rel_err(x, ref):
    """max |x - ref| / max(1, |ref|): relative for large values, absolute for small ones."""
    x, ref = np.asarray(x, float), np.asarray(ref, float)
    return float(np.max(np.abs(x - ref) / np.maximum(1.0, np.abs(ref))))


# ------------------------------------------------------------------------------------------- test data


def smooth_quantiles(n):
    """Gaussian forecasts: q_k = loc + scale * Phi^-1(alpha_k), scale log-uniform on [0.05, 2]."""
    loc = rng.uniform(-3.0, 3.0, n)
    scale = np.exp(rng.uniform(np.log(0.05), np.log(2.0), n))
    return loc[:, None] + scale[:, None] * ndtri(ALPHAS)[None, :]


def rough_quantiles(n):
    """Irregular forecasts: squared-exponential increments (widths vary by orders of magnitude, so the CDF's
    slope jumps from segment to segment), ~5% of increments set to exactly 0 (tied quantiles = jumps in the
    CDF), rescaled to a total width log-uniform on [0.1, 6]."""
    inc = rng.exponential(size=(n, K - 1)) ** 2
    inc[rng.random((n, K - 1)) < 0.05] = 0.0
    qs = np.concatenate([np.zeros((n, 1)), np.cumsum(inc, axis=1)], axis=1)
    width = np.exp(rng.uniform(np.log(0.1), np.log(6.0), n))
    return rng.uniform(-3.0, 3.0, n)[:, None] + qs / qs[:, -1:] * width[:, None]


def sample_y(qs, r):
    """r values per forecast: 70% uniform inside [q_0, q_{K-1}], 15% below q_0 and 15% above q_{K-1} by up to
    one forecast width."""
    lo, hi = qs[:, :1], qs[:, -1:]
    w = hi - lo
    u = rng.random((len(qs), r))
    inside = lo + w * rng.random((len(qs), r))
    below = lo - w * rng.random((len(qs), r))
    above = hi + w * rng.random((len(qs), r))
    return np.where(u < 0.70, inside, np.where(u < 0.85, below, above))


def span(qs, y):
    """Width of the region a grid must cover for (forecast, y): the grid tolerance is GRID_TOL per unit of it."""
    return np.maximum(qs[:, -1:], y) - np.minimum(qs[:, :1], y)


SETS = {"smooth": smooth_quantiles(N_Q), "rough": rough_quantiles(N_Q)}


# ------------------------------------------------------------------------------------------- 1. the CDF convention
# quantile_levels must be the levels TabICLv2 was trained on, and cdf must be the CDF Phase 0 grades with
# (piecewise-linear through the sorted quantiles, 0 below the first, 1 above the last). If these differ, every
# e(q_theta) and S(q_theta, y) below is computed on a different forecast than Phase 0's, and the heads' targets
# pick up a constant offset against the grading (design §2, "One CDF convention throughout").


def test_cdf_convention():
    print("\n# 1. CDF convention (quantile levels and F(t) vs Phase 0's QuantileForecast)")
    # torch.linspace (what tabicl uses) and np.linspace (Phase 0) may differ in the last bit.
    lev = quantile_levels(K).numpy()
    err = float(np.max(np.abs(lev - ALPHAS)))
    check("quantile levels are k/(K+1), k=1..K (tabicl default, Phase 0 adapter)",
          len(lev) == K and err < 1e-15, f"[{lev[0]:.4f} .. {lev[-1]:.4f}], max diff {err:.1e}")
    for name, qs in SETS.items():
        # t covers the forecast and one width on each side, so F = 0 and F = 1 regions are included.
        t = sample_y(qs, 64)
        got = cdf(T(qs), T(t)).numpy()
        ref = QuantileForecast(qs, ALPHAS).cdf(t)
        err = float(np.max(np.abs(got - ref)))
        check(f"cdf matches Phase 0 [{name}]", err < 1e-12, f"max |dF| = {err:.1e} (tol 1e-12)")


# ------------------------------------------------------------------------------------------- 2. e_crps and crps
# e_crps(q) = ∫F(1-F) is e(q_theta): the forecast's total uncertainty, the loss weight and the A/e(q) reference.
# crps(q, y) is S(q_theta, y): averaged over replicates it gives the first term of t_E. Each is checked against
# Phase 0, then against properties every correct CRPS must have (degenerate forecast, sorting, homogeneity),
# then in float32, the dtype the backbone produces.


def test_e_crps():
    print("\n# 2a. e_crps vs Phase 0's exact quantile_entropy")
    for name, qs in SETS.items():
        got = e_crps(T(qs)).numpy()
        ref = p0_entropy.quantile_entropy(qs, ALPHAS)
        err = rel_err(got, ref)
        check(f"e_crps matches Phase 0 [{name}]", err < EXACT_RTOL, f"rel err {err:.1e} (tol {EXACT_RTOL:g})")

    # TabICLv2's raw quantiles can come out unsorted; the convention sorts them first (as Phase 0 does).
    qs = SETS["rough"]
    shuffled = np.take_along_axis(qs, np.argsort(rng.random(qs.shape), axis=1), axis=1)
    ok = torch.equal(e_crps(T(shuffled)), e_crps(T(qs)))
    check("e_crps sorts the quantiles (shuffled input gives the identical value)", ok)


def test_crps_vs_phase0():
    print("\n# 2b. crps vs Phase 0's grid CRPS (independent numerical integration)")
    for name, qs in SETS.items():
        y = sample_y(qs, R)
        got = crps(T(qs), T(y)).numpy()
        ref = QuantileForecast(qs, ALPHAS).crps(y, PROTO)
        sp = span(qs, y)
        ratio = np.abs(got - ref) / (GRID_TOL * np.maximum(1.0, sp))
        for where, mask in (("y inside", (y >= qs[:, :1]) & (y <= qs[:, -1:])),
                            ("y below q_0", y < qs[:, :1]), ("y above q_K-1", y > qs[:, -1:])):
            if mask.any():
                worst = float(np.max(np.abs(got - ref)[mask]))
                check(f"crps matches Phase 0 [{name}, {where}, n={mask.sum()}]", ratio[mask].max() <= 1.0,
                      f"max |d| = {worst:.1e}, worst |d|/tol = {ratio[mask].max():.2f} "
                      f"(tol {GRID_TOL:g} x max(1, span))")

    # y exactly on knots: the branch boundaries of crps (first knot, last knot, an interior knot).
    qs = SETS["smooth"][:50]
    y = np.stack([qs[:, 0], qs[:, -1], qs[:, K // 2]], axis=1)
    got = crps(T(qs), T(y)).numpy()
    ref = QuantileForecast(qs, ALPHAS).crps(y, PROTO)
    ratio = np.abs(got - ref) / (GRID_TOL * np.maximum(1.0, span(qs, y)))
    check("crps matches Phase 0 with y exactly on q_0, q_K-1 and an interior knot", ratio.max() <= 1.0,
          f"worst |d|/tol = {ratio.max():.2f}")


def test_crps_properties():
    print("\n# 2c. crps / e_crps properties that hold for any correct implementation")
    # A forecast whose 999 quantiles all equal c is a point mass at c: e = 0 and CRPS(delta_c, y) = |y - c|.
    # Every segment has zero width, so this exercises only the outside branches and the jumps at the ends.
    c = rng.uniform(-3, 3, 30)
    qs = np.repeat(c[:, None], K, axis=1)
    y = c[:, None] + rng.normal(0, 2, (30, R))
    err_c = float(np.max(np.abs(crps(T(qs), T(y)).numpy() - np.abs(y - c[:, None]))))
    err_e = float(np.max(np.abs(e_crps(T(qs)).numpy())))
    check("point-mass forecast: crps = |y - c| and e_crps = 0", err_c < 1e-12 and err_e < 1e-12,
          f"max |crps - |y-c|| = {err_c:.1e}, max |e| = {err_e:.1e}")

    # Positive 1-homogeneity and translation invariance: S(a q + b, a y + b) = a S(q, y) and e(a q + b) = a e(q)
    # for a > 0. This is what lets the heads, trained in TabICLv2's z-normalized y units, be multiplied back to
    # raw units by the context std (design, Architecture §1, "Units").
    qs, y = SETS["rough"], sample_y(SETS["rough"], R)
    a, b = 3.7, -1.3
    err_s = rel_err(crps(T(a * qs + b), T(a * y + b)).numpy(), a * crps(T(qs), T(y)).numpy())
    err_e = rel_err(e_crps(T(a * qs + b)).numpy(), a * e_crps(T(qs)).numpy())
    check("crps and e_crps are positively 1-homogeneous and translation invariant",
          max(err_s, err_e) < 1e-9, f"rel err crps {err_s:.1e}, e {err_e:.1e}")

    # Leading batch dimensions: training calls the kernels on (tasks, queries, K) tensors. Equal up to round-off
    # (BLAS may block a batched product differently from a flat one).
    qs3 = T(SETS["smooth"]).reshape(10, 20, K)
    y3 = T(sample_y(SETS["smooth"], R)).reshape(10, 20, R)
    err = max(rel_err(crps(qs3, y3).reshape(N_Q, R), crps(qs3.reshape(N_Q, K), y3.reshape(N_Q, R))),
              rel_err(e_crps(qs3).reshape(N_Q), e_crps(qs3.reshape(N_Q, K))))
    check("extra leading batch dims give the same values as the flattened call", err < 1e-13,
          f"rel err {err:.1e}")


def test_float32():
    print("\n# 2d. float32 (the backbone's dtype) vs float64")
    for name, qs in SETS.items():
        y = sample_y(qs, R)
        s64, s32 = crps(T(qs), T(y)).numpy(), crps(T(qs).float(), T(y).float()).double().numpy()
        e64, e32 = e_crps(T(qs)).numpy(), e_crps(T(qs).float()).double().numpy()
        err_s, err_e = rel_err(s32, s64), rel_err(e32, e64)
        check(f"float32 crps and e_crps within {FLOAT32_RTOL:g} of float64 [{name}]",
              max(err_s, err_e) < FLOAT32_RTOL, f"rel err crps {err_s:.1e}, e {err_e:.1e}")


# ------------------------------------------------------------------------------------------- 3. targets
# t_A estimates the task's aleatoric entropy A (head Â's target); t_E estimates the divergence of the frozen
# forecast from the task law (head Ê_dir's target). Both must reproduce Phase 0's numpy estimators, which the
# Phase 0 audit showed to be unbiased.


def test_t_aleatoric():
    print("\n# 3a. t_A: sorted O(R log R) formula vs the pairwise definition and Phase 0")
    for r in (2, 3, 8, 16, 128):
        # Rounded values create exact ties among replicates (the sorted formula must handle them).
        Y = np.round(rng.normal(0, 1, (N_Q, r)), 1 if r > 8 else 6)
        got = t_aleatoric(T(Y)).numpy()
        iu = np.triu_indices(r, k=1)
        brute = np.abs(Y[:, iu[0]] - Y[:, iu[1]]).sum(axis=1) / (r * (r - 1))
        ref = p0_truth.replicate_aleatoric(Y)
        err = max(rel_err(got, brute), rel_err(got, ref))
        check(f"t_A = 1/(R(R-1)) Σ_r<s |y_r - y_s| and matches Phase 0 [R={r}]", err < 1e-12,
              f"rel err {err:.1e}")
    Y = rng.normal(0, 1, (N_Q, 2))
    err = rel_err(t_aleatoric(T(Y)).numpy(), 0.5 * np.abs(Y[:, 0] - Y[:, 1]))
    check("t_A at R=2 is ½|y1 - y2|", err < 1e-14, f"rel err {err:.1e}")
    # Equal up to round-off, not bit-identical: numpy's column fancy-indexing returns a column-major array, torch
    # keeps that layout through the sort, and the final matrix-vector product then sums in a different order.
    # Training tensors can arrive in any layout, so round-off agreement is the property that matters.
    Y = rng.normal(0, 1, (N_Q, 16))
    perm = Y[:, rng.permutation(16)]
    err = rel_err(t_aleatoric(T(Y)), t_aleatoric(T(perm)))
    check("t_A does not depend on replicate order", err < 1e-13, f"rel err {err:.1e}")


def test_t_epistemic():
    print("\n# 3b. t_E: mean CRPS over the replicates minus t_A, vs Phase 0")
    for name, qs in SETS.items():
        y = sample_y(qs, R)
        got = t_epistemic(T(qs), T(y)).numpy()
        ref = p0_truth.replicate_epistemic(QuantileForecast(qs, ALPHAS), y, PROTO)
        # t_E inherits crps's grid tolerance (t_A is exact on both sides).
        ratio = np.abs(got - ref) / (GRID_TOL * np.maximum(1.0, span(qs, y).max(axis=1)))
        check(f"t_E matches Phase 0's replicate_epistemic [{name}]", ratio.max() <= 1.0,
              f"max |d| = {np.max(np.abs(got - ref)):.1e}, worst |d|/tol = {ratio.max():.2f}")

    # Never clipped at zero (design §2): a narrow forecast at m with replicates m - d, m + d has
    # mean CRPS ≈ d - e(q) < d = t_A, so t_E ≈ -e(q) must come back negative and equal to the raw difference.
    m, d = 0.0, 1.0
    qs = m + 0.05 * ndtri(ALPHAS)[None, :]
    Y = np.array([[m - d, m + d]])
    tE = t_epistemic(T(qs), T(Y))
    raw = crps(T(qs), T(Y)).mean(-1) - t_aleatoric(T(Y))
    check("t_E is not clipped: a negative single target is returned as is",
          bool(tE.item() < 0) and torch.equal(tE, raw), f"t_E = {tE.item():.4f}")

    # Targets are computed against sg[q_theta]: no gradient may flow from t_E into the quantiles.
    q = T(SETS["smooth"][:5]).requires_grad_(True)
    tE = t_epistemic(q, T(sample_y(SETS["smooth"][:5], R)))
    check("t_E carries no gradient into q (stop-gradient)", not tE.requires_grad)

    # targets() is the pair the training loop calls: it must equal the two functions called separately.
    qs, y = T(SETS["rough"]), T(sample_y(SETS["rough"], 16))
    tA, tE = targets(qs, y)
    check("targets() = (t_aleatoric, t_epistemic)",
          torch.equal(tA, t_aleatoric(y)) and torch.equal(tE, t_epistemic(qs, y)))


# ------------------------------------------------------------------------------------------- 4. half-split targets
# xMSE (protocol §7.3) and the bank's noise floor (§8.1) multiply the errors against two targets built from
# disjoint halves of the replicates. They are only noise-free in expectation if the halves share no replicate
# and each half is an ordinary target of its own replicates.


def test_half_split():
    print("\n# 4. half-split targets (xMSE, noise floor)")
    qs, Y = T(SETS["rough"]), T(sample_y(SETS["rough"], 16))
    (tA_a, tE_a), (tA_b, tE_b) = half_split_targets(qs, Y)
    (rA_a, rE_a), (rA_b, rE_b) = targets(qs, Y[:, :8]), targets(qs, Y[:, 8:])
    check("each half is computed with the same functions as the full target (replicates 1-8, 9-16)",
          all(torch.equal(x, r) for x, r in ((tA_a, rA_a), (tE_a, rE_a), (tA_b, rA_b), (tE_b, rE_b))))

    # Disjointness: changing only the second half's replicates must leave half (a) bit-identical.
    Y2 = Y.clone()
    Y2[:, 8:] += torch.from_numpy(rng.normal(0, 1, (N_Q, 8)))
    (tA_a2, tE_a2), (tA_b2, _) = half_split_targets(qs, Y2)
    check("the halves use disjoint replicates (perturbing half b leaves half a unchanged)",
          torch.equal(tA_a, tA_a2) and torch.equal(tE_a, tE_a2) and not torch.equal(tA_b, tA_b2))

    for r in (7, 2):
        try:
            half_split_targets(qs[:, :], T(rng.normal(0, 1, (N_Q, r))))
            ok = False
        except AssertionError:
            ok = True
        check(f"half-split refuses R={r} (needs an even R >= 4)", ok)

    # The property xMSE exists for: with independent halves, E[(c - t^a)(c - t^b)] = (c - A)^2 exactly, while
    # E[(c - t_full)^2] = (c - A)^2 + Var(t_full). Checked on Gaussian replicates (sigma = 1, A = 1/sqrt(pi)) for a
    # fixed "head output" c, by Monte Carlo over many queries.
    n, c = 400_000, 0.3
    A = 1.0 / np.sqrt(np.pi)
    Yg = T(rng.normal(0, 1, (n, 16)))
    ta, tb = t_aleatoric(Yg[:, :8]).numpy(), t_aleatoric(Yg[:, 8:]).numpy()
    prod = (c - ta) * (c - tb)
    z = (prod.mean() - (c - A) ** 2) / (prod.std(ddof=1) / np.sqrt(n))
    raw = ((c - t_aleatoric(Yg).numpy()) ** 2).mean()
    check("xMSE from the halves is unbiased for (c - A)^2 (Gaussian replicates, |z| < 4)", abs(z) < 4,
          f"xMSE {prod.mean():.5f} vs (c-A)^2 {(c - A) ** 2:.5f} (z = {z:+.2f}); raw MSE {raw:.5f} "
          f"includes the target variance")


def main():
    torch.manual_seed(0)
    test_cdf_convention()
    test_e_crps()
    test_crps_vs_phase0()
    test_crps_properties()
    test_float32()
    test_t_aleatoric()
    test_t_epistemic()
    test_half_split()
    print(f"\n{'ALL OK' if not FAILS else f'{len(FAILS)} FAILED: ' + '; '.join(FAILS)}", flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
