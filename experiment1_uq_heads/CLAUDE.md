# experiment1_uq_heads — Phase 1 of iLTMv2

Train two scalar heads (Â aleatoric, Ê epistemic) on top of the frozen TabICLv2 regressor, with exact replicate targets from the observed-cut prior. Two specs live in this folder; read the relevant one before writing code. Where they disagree, `phase1_design.md` wins on *what* is trained and `phase1_protocol.md` wins on *how* it is built and run. `../iLTMv2_research_proposal.md` outranks both: on a contradiction with it, stop and ask (root `CLAUDE.md`).

## `phase1_design.md` — what is trained

- **Architecture and loss:** §1 heads (softplus scalars on $h_*$, link to DEUP's stationarizing features), §2 targets (raw squared error and replicates), §3 loss, §4 connecting heads to the backbone: the ladder of arms (A1 MLP trunks on $h_*$ = default, A2 multi-layer tap, B trainable fork of the last $k$ blocks).
- **Training procedure and job layout:** §0 what is kept from TabICLv2's training, §1 task construction (fixed sizes, pipeline, query-type mixture, in-context duplicates, task filtering), §2 curriculum/batch/optimization, §3 validation (the bank, logged metrics, best-per-head checkpoints, no stopping rule), §4 sweeps/seeds/selection, §5 job layout and the shared-forward saving, §6 defaults at a glance.

## `phase1_protocol.md` — how it is built and run

§1 code organisation (import tabicl, never edit it; hooks instead of patches; target file layout), §2 storage (heavy outputs on `/private/groups/ioannidislab/UQ_decoupling_exp/`), §3 order of work — nine steps, each with a gate, §4 unit tests per step (as `scripts/00{a..e}_test_*.py` OK/FAIL gate scripts, no pytest), §5 split figures, §6 smoke runs before long jobs, §7 wandb logging, §8 reading the curves (reference lines, rules of thumb, red flags), §9 launch plan.

## tabicl version

`external/tabicl` pinned at commit `0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3`, clean working tree (no local patches). Record any Phase 1 patch to tabicl here (protocol §1.1).

## Layout notes

- `iltm2` is editable-installed in `uq-iltm-bio` (`pip install -e experiment1_uq_heads --no-deps`).
- `outputs` is a symlink to the group-disk root `/private/groups/ioannidislab/UQ_decoupling_exp/` (protocol §2): logs land in `outputs/slurm_logs/`.

## Current task

Step 1 (entropy and targets): done, `00a_test_entropy` passes.

Step 2 (observed-cut prior and filter): done; prior configuration fixed. Code in `src/iltm2/prior/` (`_cut_graph_scm.py` the cut itself, `_cached_graph.py` cached graph_lib pieces, `_postprocess.py` fit/apply post-processing, `_filter.py`, `_dataset.py` task RNG + shapes + `sample_task`). The design, what changed from `graph_scm` and why, and the difficulty-match results are in design Training §1, "The observed-cut prior". Defaults: no noise-ancestor check, `private_root_scale` 0.1, `private_root_dims_range` (1, 16). Gates: `scripts/00b_test_cut_prior.sh` (unit tests, CPU) and `scripts/00f_difficulty_match.sh` (prior-level gate; stage 1 accepted with the q25 deviation documented in the design). To do: add stage-2/3 results of job 45910541 to the design once it finishes. Next: step 3 (`_split.py`).

Gotchas found while building it:
- In v2's `NumericalConverter` (`use_corrected_num_converters=False`) the returned column is a view of the node slice that is then overwritten with the write-back, so the observed numerical column *is* the warped write-back.
- Nodes pruned by tabicl are never fitted (no `converters_`); only feature-hosting nodes are guaranteed fitted.
- Bit-identical tasks across processes need the same torch intra-op thread count (1 in DataLoader workers).
- An X-graph latent can be NaN while its observed column is finite (e.g. a category index from argmin): its center then reaches the target graph. Non-finite target-graph inputs reject the task (`non_finite_x`). Never guard with `torch.equal` on tensors that may hold NaN (NaN != NaN); use the bitwise `_same_bits`.
- Replicates must be R separate passes over the same rows, never one pass over tiled rows: float32 round-off depends on a row's position in the batch, and tree splits / argmins amplify it into spurious replicate spread (up to 0.2 std(y) on deterministic tasks).
- A non-finite replicate is only seen after acceptance: `RejectionLog.revoke("non_finite_replicate", ...)` turns the task into a rejection, and the caller draws a new one (done in `00f`; step 4's dataset must do the same).
- About 60% of accepted tasks are exactly deterministic (no private root reaches y, t_A ≡ 0); the `00b` tests that need noise pick a noisy task explicitly (`noisy_task`).

## Waiting on

