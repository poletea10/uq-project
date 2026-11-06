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

Step 2 (observed-cut prior and filter): done; prior configuration fixed. Code in `src/iltm2/prior/` (`_cut_graph_scm.py` the cut itself, `_cached_graph.py` cached graph_lib pieces, `_postprocess.py` fit/apply post-processing, `_filter.py`, `_dataset.py` task RNG + shapes + `sample_task`). The design, what changed from `graph_scm` and why, and the difficulty-match results are in design Training §1, "The observed-cut prior". Defaults: no noise-ancestor check, `private_root_scale` 0.1, `private_root_dims_range` (1, 16). Gates: `scripts/00b_test_cut_prior.sh` (unit tests, CPU) and `scripts/00f_difficulty_match.sh` (prior-level gate; stage 1 accepted with the q25 deviation documented in the design). To do: add the stage-2/3 difficulty-match results to the design once that run finishes.

Step 3 (split): `_split.py`, gate `00c_test_split`, figures `03_plot_split_tasks`.

Step 4 (dataset and throughput): done (gate outcome below). `_dataset.py` now has `sample_uq_task` (steps 1-10: task + split + R replicates; a non-finite replicate revokes the task and redraws it whole on streams + 100k, same shape), `UQBatch` (one micro-batch group sharing (N, n_ctx); X zero-padded to the group's max d), `CutGraphPrior` (map-style Dataset, item = one group, step = i // n_groups; groups of 4 in stage 1 and 1 in stages 2-3, as tabicl's `batch_size_per_gp`; group shape from `STREAM_SHAPE`; stage in the RNG key via `stage_step_key`), `make_dataloader` (tabicl's settings) and `iterate_steps`. Gate: `scripts/02_benchmark_prior.sh` (GPU node, 34 CPUs): workers-0 vs W-worker bit-identity, steady tasks/s per stage for W in {16, 32}, a per-component profile, and the frozen fp32 backbone forward per step as the train-time floor; passes if prior s/step <= 0.8 x backbone s/step. Output in `outputs/audits/throughput/`.
First run (in-order loader, 0.5 gate): FAIL in all stages. Stage 1 is CPU-bound (even perfect parallelism at 32 workers is slower than the backbone forward); stages 2-3 lost most of the workers' capacity to in-order delivery behind heavy-tailed large-N tasks; the 34 CPUs are ~17 physical cores with 2 threads each (phoenix GPU nodes: 2 x 64 cores x 2 threads; SLURM allocates threads). Stage-1 rejection rate there was above step 2's 30% gate. Fix applied (user's decision; no change to cores, batch size or design): out-of-order delivery (`in_order=False`), `LargestFirstSampler`, `prefetch_factor` 4, per-step reorder buffer in `iterate_steps`; gate relaxed to prior <= 0.8 x backbone. Rerun with the fix: bit-identity OK in every stage; stages 2-3 pass at 32 workers (out-of-order delivery removed the in-order loss in stage 2; stage 3 passes with room to spare); stage 1 still FAILS: it is CPU-bound (even perfect parallelism over the 34 threads is slower than the backbone forward), so stage 1 runs at the prior's pace, not the GPU's. Stage-1 rejection rate is again above 30%. User's decisions: stage 1 accepted as prior-bound (no offline stream, no core change); batch halved to 32 tasks per step (`BATCH_SIZE` in `_dataset.py`; step budget unchanged; design Training §2, proposal, protocol step 4 updated); stage-1 rejection rate above 30% accepted as a documented deviation (design Training §1). Step 4 is closed.

LR schedule (user's decision): B's fork LR scales across stages by the same /8, /40 ratios as the head LR (design Training §2 and §6, proposal). No per-stage LR branching: the sweep stays one stage-1 value per LR.

Validation cadence (user's decision): full bank every 5,000 steps in stage 1 + end of each stage (halves validation time; best-per-head checkpoints at 5,000-step resolution in stage 1); mini-bank stays every 250 steps (design Training §3 and §6, protocol §7.4–7.5 and §8.2, proposal). The trainer (steps 7–8) must implement exactly this.

Step 5 (backbone wrapper): code written, gate not yet run. `src/iltm2/model/_backbone.py`: `FrozenBackbone` loads TabICLv2 (`TabICL(**ckpt["config"])`, training mode, frozen; not an nn.Module), runs tabicl's own `model(X, y_ctx)` under no_grad, and reads `h_star` (output of `icl_predictor.ln`), the 12 residual taps (outputs of `icl_predictor.tf_icl.blocks[l]`, query rows) and arm B's fork input (full stream after block 12-k) with forward hooks only. `copy_last_blocks(k)` / `run_blocks` give arm B's trainable copy. Gate: `scripts/00d_test_backbone.sh` (CPU, short).

Gotchas found while building it:
- tabicl turns FlashAttention-3 (fp16 attention) on whenever `flash_attn_interface` imports (`_use_flash_attn3 = True` at import; v2 pretraining used it in stages 2-3 only). It is not installed in uq-iltm-bio (and is Hopper-only), but `attention_ctx` forces it off anyway, so every forward is fp32 SDPA (TabICLv2's `--dtype float32`) with cuDNN SDPA excluded as in tabicl's trainer.
- 00d passed on CPU; the padded-group check was then rewritten to force two tasks with different d (rerun pending).

Step 6 (validation bank): **done**, gate passed (`01b_check_val_bank`: ALL OK on the full bank). Stored on the group disk: the bank (2,000 tasks, 42 shards, `queries.pt`, `manifest.json` with the reference lines, `build_config.json`) in `/private/groups/ioannidislab/UQ_decoupling_exp/val_bank/`, the 64-task mini-bank in its `mini/` subfolder (`tasks.pt`, `queries.pt`; mini reference lines in the manifest under `mini`), and the 14-task smoke bank (code test only, not for training) in `/private/groups/ioannidislab/UQ_decoupling_exp/val_bank_smoke/`. Check outputs: `outputs/audits/val_bank_checks/<bank>/`. Bank results (t_E by query type, e(q) proxy skill) are in design Training §3, "What the bank showed". `src/iltm2/prior/_bank.py` (bins, `generate_bank_task`, `forecast_targets`, `BankQueries` per-query table, sharded storage), `src/iltm2/train/_metrics.py` (slices, MSE / xMSE / skill / ratio, reference lines; the trainer must reuse it so `val/...` and `ref/...` keys match), `src/iltm2/prior/_graph_scm_ref.py` (graph_scm reference tasks, moved out of 00f). Builder `scripts/01_make_val_bank.sh` (GPU A5500, resumable shards; smoke: `--n-tasks 8 4 2 --out .../val_bank_smoke`), checks `scripts/01b_check_val_bank.sh` (integrity, reproducibility, t_E ordering + t_A dup = iid, cut vs graph_scm CRPS ratio per N bin within 1.25). Bank seed `VAL_BANK_SEED = 20261103`: the T2 test bank must use another. "n bin" = the task's context size n_ctx (written into protocol §7.5).
- The checkpoint still loads from `/private/groups/ioannidislab/ajaras/tabicl-dev/tabicl-regressor-v2-20260212.ckpt` (`DEFAULT_BACKBONE`); the protocol §2 copy under `UQ_decoupling_exp/backbone/` does not exist yet.

Gotchas found while building it:
- In v2's `NumericalConverter` (`use_corrected_num_converters=False`) the returned column is a view of the node slice that is then overwritten with the write-back, so the observed numerical column *is* the warped write-back.
- Nodes pruned by tabicl are never fitted (no `converters_`); only feature-hosting nodes are guaranteed fitted.
- Bit-identical tasks across processes need the same torch intra-op thread count (1 in DataLoader workers).
- An X-graph latent can be NaN while its observed column is finite (e.g. a category index from argmin): its center then reaches the target graph. Non-finite target-graph inputs reject the task (`non_finite_x`). Never guard with `torch.equal` on tensors that may hold NaN (NaN != NaN); use the bitwise `_same_bits`.
- Replicates must be R separate passes over the same rows, never one pass over tiled rows: float32 round-off depends on a row's position in the batch, and tree splits / argmins amplify it into spurious replicate spread (up to 0.2 std(y) on deterministic tasks).
- A non-finite replicate is only seen after acceptance: `RejectionLog.revoke("non_finite_replicate", ...)` turns the task into a rejection, and the caller draws a new one (done in `00f`; step 4's dataset must do the same).
- About 60% of accepted tasks are exactly deterministic (no private root reaches y, t_A ≡ 0); the `00b` tests that need noise pick a noisy task explicitly (`noisy_task`).

## Waiting on

- The user's rerun of `sbatch experiment1_uq_heads/scripts/00d_test_backbone.sh` (padded-group check).

