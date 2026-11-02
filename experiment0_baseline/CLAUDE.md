# experiment0_baseline — Phase 0 of iLTMv2

**Read `phase0_protocol.md` (same folder) before writing any code.** It is the spec. This file only holds what must stay in view on every edit: the goal, the invariants, where things live, and the current status. If this file and the protocol disagree, the protocol wins — flag the disagreement rather than resolving it silently.

## Goal, in one line

Show where aleatoric/epistemic decompositions put their uncertainty (toy set), grade them against exact ground truth (cut-graph and transfer tests), then check on active learning whether the epistemic split holds useful information.

## The four tests

| # | Test | Models | Output |
|---|---|---|---|
| T1 | Toy set: one curated 1-D function, toy rungs 0–2 (deterministic / homoscedastic / heteroscedastic), sparse strip $[1,4]$ on $[-5,5]$ | Heteroscedastic GP, Dec-ICL, iLTMv2 | One 3×4 context figure per (model, toy rung) |
| T2 | Cut-graph test (in distribution, observed-cut prior), with context-hole queries | Dec-ICL, iLTMv2 | Reliability plots + table |
| T3 | Transfer test (unmodified `graph_scm`, low-cardinality categorical features, exact-match replicates), with held-out-cell queries | Dec-ICL, iLTMv2 | Reliability plots + table |
| T4 | Active learning (DecoupledPFN protocol) | Random, TabICLv2, iLTMv2, Tuned-ICL, Dec-ICL | Table + learning curves |

Scope: **regression only**; $x\in\mathbb R$ on T1, $x\in\mathbb R^d$ on T2–T4. Baselines: **heteroscedastic GP** (T1 only) and **Dec-ICL** (Bergna et al., 2026). TabICLv2 is a total-only reference (backbone check + T4), not a baseline. iLTMv2 rows stay empty until Phase 1.

## Invariants (break none of these)

- $\hat A$ is the average of **member** entropies; never convert a pre-aggregated variance split. A negative $\hat E$ from `project` means this rule was broken.
- The epistemic truth is **each model's own** excess risk $E(q_m)$, so every adapter must return its forecast `cdf`.
- **Never regress an estimate on realized truth.** Truth on estimate only.
- Validate the truth code (`00b`) and the adapter conversions (`00c`) before trusting any model output.
- On T3, replicate rows are **never** context rows.
- Freeze the toy function (curated on $f$ only, never on model outputs) and the row set before the first run.

## Recyclable code from the previous version

Previous version: **`/private/home/ajaras/iLTM-bio/UQ_project/SCRAPPED/v2/experiment0_baseline`**. Its design was partly incorrect, so reuse code only where it agrees with the current protocol. Worth taking from it:

- **Reliability-plot evaluator** (its `src/grading.py` and the reliability functions of its `src/figures.py`: truth-on-estimate OLS, bootstrap, binned plots coloured by $n$) for T2/T3 and for the optional seed-averaged toy appendix figure.

The heteroscedastic GP adapter (DTC sparse GP, protocol §5.2), the entropy kernels, the forecast classes, `truth.py` and the context-figure layout are already recycled into `src/`.

## Where things live (T1 and backbone check)

- `src/toy.py` — curated $f$, $\sigma_k(x)$, context sampler, `ToyTask`; `src/evaluate.py` — `run_toy` (fit, truth, quantiles, one parquet per (method, rung, $n$) in `outputs/raw/toy/<method>/`).
- `src/adapters/` — `hgp.py`, `tabiclv2.py`, `base.py` (`UQOutput`, `gaussian_members`).
- Run order: `00_curate_toy` → `00b_validate_estimators`, `00c_validate_adapters` → `01_toy --method hgp` → `00a_backbone_check` (criterion 1 reads the GP's outputs). Figures land in `outputs/figures/toy/`, tables in `outputs/tables/`. `01_toy --paper-figure` draws the 3×3 paper figure from saved outputs only.
- `run_toy` reuses any existing parquet: after changing a toy or method constant, move the old outputs to a `v<k>/` subfolder inside `outputs/archive` or pass `--overwrite`.

## Current task

T1 code written (protocol §11 items 1–5). Current toy set: `gap_density` 0.075, heteroscedastic GP with a learned log-noise mean $m_g$ (§5.2); the previous run (density 0.025, zero-mean $g$) is in the `archive/v1/` output folder. Next: rerun `00c`, `00_curate_toy` (re-checks the frozen seed, redraws the toy figure), `01_toy --method hgp`, then `00a_backbone_check`; the backbone check is the decision point between the hand-built $f$ and the fallback draw from our own prior (§2.7). Then item 6, the Dec-ICL adapter.

## Waiting on

- Dec-ICL and Tuned-ICL checkpoints (authors contacted). Until then, write the Dec-ICL adapter against the interface in protocol §5.3 and unit-test it on synthetic outputs.
- The Phase 1 observed-cut prior, needed for T2.
