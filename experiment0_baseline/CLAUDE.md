# experiment0_baseline — Phase 0 of iLTMv2

**Read `phase0_protocol.md` (same folder) before writing any code.** It is the spec. This file only holds what must stay in view on every edit: the goal, the invariants, where things live, and the current status. If this file and the protocol disagree, the protocol wins — flag the disagreement rather than resolving it silently.

## Goal, in one line

Show where aleatoric/epistemic decompositions put their uncertainty (toy set, error heatmap), then check on real data whether the estimates rank the right errors (PRR-A, PRR-B) and select useful data (active learning, context retrieval).

## The six tests (protocol §2–§3, §7–§10)

| # | Test | Ground truth | Models | Output |
|---|---|---|---|---|
| T1 | Toy set: one curated 1-D function, toy rungs 0–2, `n_grid` {64, 256, 1024, 4096} | Closed form | Heteroscedastic GP, Dec-ICL, iLTMv2 | 3×4 context figures per (model, rung) + one 3×3 paper figure |
| T2 | Error heatmap over local data density × local noise (probe strip) | Closed form | Heteroscedastic GP, Dec-ICL, iLTMv2 | One heatmap figure |
| T3 | PRR-A: do Â+Ê rank the forecast's realized CRPS better than its own spread? | Realized CRPS | iLTMv2 (Dec-ICL if available) | Curves + table |
| T4 | PRR-B: does Ê rank the realized CRPS gain from more data? | Realized CRPS gain | iLTMv2 (Dec-ICL if available) | Curves + table |
| T5 | Active learning (DecoupledPFN protocol) | Test-set performance | Random, TabICLv2, iLTMv2, Tuned-ICL, Dec-ICL | Table + learning curves |
| T6 | Context retrieval: Ê-guided vs k-NN | Realized CRPS | iLTMv2 | Curves + table |

Scope: **regression only**. Baselines: **heteroscedastic GP** (T1, T2) and **Dec-ICL** (Bergna et al., 2026). TabICLv2 is a total-only reference (backbone check + T5), not a baseline. T3–T6 share the datasets and stored splits of protocol §6. The iLTMv2 candidates (Phase 1's carved winners) run on T2–T5 to choose the final model; the final model runs on T1 and T6.

## Invariants (break none of these)

- $\hat A$ is the average of **member** entropies; never convert a pre-aggregated variance split. A negative $\hat E$ from `project` means this rule was broken.
- The epistemic truth is **each model's own** excess risk $E(q_m)$, so every adapter must return its forecast `cdf`; iLTMv2's `e_total` is the forecast's own entropy, never Â + Ê.
- **Mean first, ratio second** (protocol §12, gotcha 6): a single realized excess or loss carries the scatter of protocol §4.5; never grade Ê against one realized value, never report R² or rank correlation against realized truth as a grade.
- On T3–T6, **test points never enter the context** (protocol §6).
- Validate the truth code (`00b`) and the adapter conversions (`00c`) before trusting any model output.
- Freeze the toy function (curated on $f$ only, never on model outputs), the row set and the T2 grid before the first run. The one recorded deviation is the toy set's `n_grid` (protocol §1).

## Where things live (T1 and backbone check, implemented)

- `src/toy.py` — curated $f$, $\sigma_k(x)$, context sampler, `ToyTask`; `src/evaluate.py` — `run_toy` (fit, truth, quantiles, one parquet per (method, rung, $n$) in `outputs/raw/toy/<method>/`).
- `src/adapters/` — `hgp.py`, `tabiclv2.py`, `iltmv2.py` (Phase 1's `iltm2.inference.ILTMv2Regressor`; methods `iltmv2` = arm A, the paper figure's column, and `iltmv2_b` = arm B, `configs/methods.yaml`), `base.py` (`UQOutput`, `gaussian_members`). Until Phase 1's selection exists, both methods load the first round's selected checkpoints (`legacy: true`).
- Run order: `00_curate_toy` → `00b_validate_estimators`, `00c_validate_adapters` → `01_toy --method hgp` → `00a_backbone_check` (criterion 1 reads the GP's outputs). Figures land in `outputs/figures/toy/`, tables in `outputs/tables/`. `01_toy --paper-figure` draws the 3×3 paper figure from saved outputs only.
- `run_toy` reuses any existing parquet: after changing a toy or method constant, move the old outputs to a `v<k>/` subfolder inside `outputs/archive` or pass `--overwrite`.
- The target modules of T2–T6 (`heatmap.py`, `datasets.py`, `prr.py`, `retrieval.py`, scripts `02`–`06`, configs `heatmap.yaml`, `al.yaml`, `retrieval.yaml`) are listed in protocol §1; none exists yet.

## Current task

Protocol §13, items 6–11 (the user's current work): the toy-set updates (`n_grid` {64, 256, 1024, 4096} with the nested-context assert, the row-3 estimator legend, per-model row-2 limits on toy rung 0), the Dec-ICL adapter, T2, the datasets and stored splits with the T5 harness, T3/T4, and T6. Then the baselines (heteroscedastic GP, TabICLv2 rows, Dec-ICL if its checkpoint arrives) on T1, T2 and T5 (transfer guide §5 Step 4).

Code still behind the protocol (update with items 6–11): `configs/protocol.yaml` still has `n_grid` 16–1024 and the `R_rep` / `n_bins` constants; docstrings cite old protocol sections (RNG streams are now §11, gotchas §12; `truth.py` cites §4.2 and "T2, T3" replicates, which no longer exist as Phase 0 tests).

## Waiting on

- Dec-ICL and Tuned-ICL checkpoints (authors contacted). Until then, write the Dec-ICL adapter against the interface in protocol §5.3 and unit-test it on synthetic outputs.
- Phase 1's selected checkpoints (after the A job and, later, the B jobs) for the iLTMv2 rows of T2–T6.
