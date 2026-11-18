# experiment1_uq_heads — Phase 1 of iLTMv2

Train two scalar heads (Â aleatoric, Ê epistemic) on top of the frozen TabICLv2 regressor, with exact replicate targets from the observed-cut prior. Two specs live in this folder; read the relevant one before writing code. Where they disagree, `phase1_design.md` wins on *what* is trained and `phase1_protocol.md` wins on *how* it is built and run. `../iLTMv2_research_proposal.md` outranks both: on a contradiction with it, stop and ask (root `CLAUDE.md`).

**Second training round.** The first round is finished and archived (root `CLAUDE.md`, "Current state"); `../iLTMv2_transfer_guide.md` is the only document that compares the two. Both specs describe the current design (rewritten in guide §5 Step 4); the first round's checkpoints are in the legacy block at the end of `phase1_protocol.md`. In this file, the sections up to "Waiting on" are the current round, and everything under "v1 record" is archived history.

## `phase1_design.md` — what is trained

- **Architecture and loss:** §1 heads (softplus scalars on $h_*$, link to DEUP's stationarizing features), §2 targets (raw squared error and replicates), §3 loss (p in {0, 1}, c = 0.1), §4 connecting heads to the backbone: the two arms, **A** (MLP trunks on $h_*$) and **B** (trainable fork of the last 2 blocks + LN).
- **Training procedure and job layout:** §0 what is kept from TabICLv2's training, §1 task construction (observed-cut prior, fixed sizes, pipeline, query-type mixture with thinned regions, duplicates, filtering, §1.5 the stage-1 subsampled pass, §1.6 the carved and iid priors), §2 curriculum/batch/optimization, §3 validation (the bank and what it showed, the ceiling of the epistemic skill, logged metrics, best-per-head checkpoints), §4 sweeps/seeds/selection, §5 job layout and the shared-forward saving, §6 defaults at a glance.

## `phase1_protocol.md` — how it is built and run

§1 code organisation (import tabicl, never edit it; hooks instead of patches; file layout), §2 storage (heavy outputs on the group disk), §3 order of work: nine steps, each with a gate, and the accepted deviations, §4 unit tests per step (as `scripts/00{a..e}_test_*.py` OK/FAIL gate scripts, no pytest), §5 split figures, §6 smoke runs before long jobs, §7 wandb logging (key names, both weightings), §8 reading the curves (reference lines, rules of thumb, red flags), §9 launch plan, selection, export, §10 the legacy checkpoints.

## tabicl version

`external/tabicl` pinned at commit `0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3`, clean working tree (no local patches). Record any Phase 1 patch to tabicl here (protocol §1.1).

## Layout notes

- `iltm2` is editable-installed in `uq-iltm-bio` (`pip install -e experiment1_uq_heads --no-deps`).
- `outputs` is a symlink to the v2 group-disk root `/private/groups/ioannidislab/UQ_decoupling_exp_new/` (protocol §2). Every path lives in `src/iltm2/paths.py` (`ROOT`, `RUNS_ROOT`, `LOG_DIR`, `LEGACY_ROOT`); every `scripts/*.sh` logs to `ROOT/slurm_logs/` (created).
- v1 outputs (runs, checkpoints, val banks, audits, wandb, SLURM logs) stay in `LEGACY_ROOT` = `/private/groups/ioannidislab/UQ_decoupling_exp/`. Read them, never write to them. Only `inference.LEGACY_BEST` points there.
- Validation bank: `ROOT/val_bank_v2/` (seed `VAL_BANK_SEED = 20261116`); smoke bank (code tests only): `ROOT/val_bank_v2_smoke/`.

## Current task

v2 Steps 0 and 1 of the transfer guide's workplan (§5): **done** (code written, syntax-checked; `00c` run as a first sanity check, see below). What the code now does:
- **Prior** (`prior/_dataset.py`, `prior/_split.py`, `CutGraphConfig.thin_*`): each carved region is empty with probability 0.5, else thinned (rho log-uniform [0.05, 0.2] of its non-query rows returned to the context, at least one row, so the tail and the hole are each empty or thinned 50/50, independently); sizes drawn before generation, `TaskShape.P` includes the returned rows. Six query types (codes: iid 0, hole 1, extrapolation 2, duplicate 3, hole_thinned 4, extrapolation_thinned 5). `prior_config("iid")` = query_mix (1, 0, 0) and no duplicates (P = N). Stage 1: every group carries a subsampled pass in `UQBatch.sub` (one n' log-uniform in [64, n_ctx] per group, streams `STREAM_SUB_SIZE` / `STREAM_SUB_ROWS`); `UQBatch.passes` lists the forwards.
- **Bank** (`prior/_bank.py`, `01_make_val_bank.py`): small = N uniform 256-1,024 at exactly 80% context (`int(0.8 N)`), medium / large as before (stage-3 rule 79-81%); thinning records per task; `manifest["loss_c"]` = c, **fixed at 0.1** (`_bank.LOSS_C`; user's decision after the bank's e(q) quantiles: the median, 0.013, falls in the gap of a bimodal e(q) and would let the deterministic tasks dominate the p = 1 loss; `--loss-c` rewrites the manifest and refs only); `manifest["e_q_quantiles"]`; reference lines for both weightings (`ref/w...` keys).
- **Model** (`model/_heads.py`, `model/_backbone.py`): arms A and B only; `LayerMix` and the residual-tap hooks removed; `HeadPair(arm, lr, seed, p)`; A pair ids `A_lr<LR>_p<P>`; same init for every LR and p, shared by A and B (key unchanged from v1, so v1 A1 / B checkpoints still load as A / B).
- **Training** (`train/`): steps 30,000 / 8,000 / 1,200; full bank every 5,000 / 4,000 / 600 stage steps (ten validations); head LR {3e-4, 1e-3, 3e-3}, fork LR {1e-5, 3e-5, 1e-4}, `--p` (A: list, default 0 1; B: one value), `--prior {carved,iid}`, `--loss_c` (default from the bank manifest). Loss per pair with its own p; stage 1 averages over both passes. A caches h_star on the full bank and mini-bank once per launch. Weighted metrics `wmse` / `wxmse` / `wskill` next to the unweighted ones; slices = all + 6 query types + n bins `n_le256, n_256-512, n_512-1K, n_1K-4K, n_4K-16K, n_16K-60K`; the mini-bank reports its n bins; training xMSE is logged per n bin. Best checkpoints per pair under its own weighting. Run names `A_<prior>_seed<S>`, `B_h<H>_f<F>_p<P>_<prior>_seed<S>`. Each stage also writes `last_{train,mini,val}.json`.
- **Launchers**: `train_A.sh <carved|iid> <seed> [flags]`, `train_B.sh <head_lr> <fork_lr> <p> <carved|iid> <seed> [flags]`; `04_smoke_A.sh` takes c from the real bank for every stage.
- **Inference** (`inference.py`): `BEST` (empty until the v2 selection), `LEGACY_BEST` (v1 A1-as-A and B); `from_best(arm, legacy=...)`; v1 A2 files are refused. experiment0's `iltmv2` / `iltmv2_b` use `legacy: true` until then (`iltmv2_a1` removed).
- **Gates updated for v2**: `00c` (thinning, labels, iid prior, subsampled pass), `00d` (taps removed; new section 6 = query-independence test of guide Step 2.3), `00e` (A/B, LR x p pairs, p = 0 weight exactly 1, both weightings, per-weighting best tracking), `01b` (combined region shares, thinning integrity, ordering duplicate < iid < thinned < empty, section W on c), `02` (two forwards per stage-1 group), `03` (returned rows drawn, open markers for thinned queries, extra figures until 2 thinned tails and 2 thinned holes), `04b` (passes, new slices and weightings in the logs, new cadence in the projection), `05` (new names, p / prior columns, weighted metrics; alphas export removed). `00a`, `00b`, `00f` are unchanged (the code they cover did not change).

Choices made in Step 1 that the guide does not fix (listed for the doc rewrite of Step 4):
- In the subsampled pass, a thinned region whose returned rows were all dropped is labelled empty, and a duplicate whose source was dropped takes its source row's type (iid, or its region's type).
- p = 1 pairs start at the weighted mean target (the best constant under their loss); weighted metrics are count-normalized (mean of w x, the form of the loss).
- The iid prior draws the same task distribution as the carved one, not the same tasks: removing the carving changes P and the shape's RNG draws.

Code policy (guide §5): reuse the v1 code where it fits; rewrite where keeping it would be convoluted. No A2 code paths or flags that only v1 used. Rerun a gate only when the code it covers has changed.

### Step 2 gate results

- `00c`, `03` (figures checked by the user), `00d`, `00e`: pass.
- `01b`: everything passes except the PPD-sanity ratio of the medium bin, 1.255 against the 1.25 limit (CI [1.09, 1.44]), accepted by the user: within sampling noise, and the same pattern as the first bank (cut harder than graph_scm at small / medium N, easier at large N).
- **`02` throughput: two accepted deviations (user's decision).** At the training setting of 32 workers:
  - **Stage 2 is mildly prior-bound:** prior 5.73 s/step against the backbone's 5.17 s/step (ratio 1.11, gate 0.8). The backbone cost per task matches the first round's; the prior falls behind because a 32-task step has only 32 single-task items, and the heavy-tailed task cost (a task can take tens of seconds after several rejections) holds the step back. Cost: stage 2 runs at the prior's pace, about 12.7 h instead of about 11.5 h for an A job. No change to workers, batch or design.
  - **Stage 3 misses only the 0.8 margin:** ratio 0.90 (still GPU-bound, below 1), measured on 4 steps. The prior does not slow it down; projected 17.2 h for an A job.
  - Stage 1 now passes (ratio 0.56): the subsampled pass doubles the GPU work per step, so stage 1 is no longer prior-bound. The measured 1.03 s/step is optimistic (below the 1.60 s/step of perfect parallelism, because the short run drains the prefetched items); about 1.6 s/step, ratio about 0.86, is the realistic sustained rate.
  - Projected A job: about 45 h of training (15.4 + 12.7 + 17.2) plus validation, within `train_A.sh`'s 3-day limit. B jobs add a fork backward, so they are more GPU-bound.

- `04` smoke run: passed (after the overfit batch was chosen by its A targets: a noisy task can have t_A = 0 at every query).

### Step 3: the A job is running

`A_carved_seed0` (6 pairs, wandb project `TabUD`). Reading its curves:
- `train/xmse_*` and `mini/` / `val/` `mse`, `xmse`, `skill` are UNWEIGHTED for every pair, so p = 0 and p = 1 pairs compare fairly on them; `wmse` / `wxmse` / `wskill` are the weighted versions, also logged for every pair. Only `train/loss*` uses each pair's own p (not comparable across p).
- Stage-1 training xMSE pools both passes: the subsampled pass (contexts down to 64 rows) has larger errors, so the pooled training curves sit above the first round's (single pass, contexts >= ~307). Compare per n bin (`train/xmse_*/<n bin>`), not the pooled value, against the first round.

## Future work

- **Arm B (18 jobs) is deferred** until the Phase 0 tests are written and their baselines run (user's decision; guide §5 Step 5). The code and `train_B.sh` are ready; no B gate beyond the startup checks in each job.

## Waiting on

- The A job's stages 1 -> 2 -> 3 (about 2 days); then guide §5 Step 6.1 for arm A (LR per (p, prior) cell from validation).
- Meanwhile: guide §5 Step 4 (the documents are rewritten; Phase 0 test code is the user's current work).

---

# v1 record (archived)

What follows is the history of the v1 round: the code built step by step, and the v1 results. The code and its gotchas are still the base that v2 reuses, so the gotchas stay valid unless the v2 change touches that piece. v1 artifacts and their paths are in `/private/groups/ioannidislab/UQ_decoupling_exp/`, and the archived repository is `/private/groups/ioannidislab/UQ_project_v1` at commit `c1ed503d0a269051724ee61485862b96472eee45`.

## v1 build log (protocol §3 steps)

Step 1 (entropy and targets): done, `00a_test_entropy` passes.

Step 2 (observed-cut prior and filter): done; prior configuration fixed. Code in `src/iltm2/prior/` (`_cut_graph_scm.py` the cut itself, `_cached_graph.py` cached graph_lib pieces, `_postprocess.py` fit/apply post-processing, `_filter.py`, `_dataset.py` task RNG + shapes + `sample_task`). The design, what changed from `graph_scm` and why, and the difficulty-match results are in design Training §1, "The observed-cut prior". Defaults: no noise-ancestor check, `private_root_scale` 0.1, `private_root_dims_range` (1, 16). Gates: `scripts/00b_test_cut_prior.sh` (unit tests, CPU) and `scripts/00f_difficulty_match.sh` (prior-level gate; stage 1 accepted with the q25 deviation documented in the design). Left open in v1: adding the stage-2/3 difficulty-match results to the design.

Step 3 (split): `_split.py`, gate `00c_test_split`, figures `03_plot_split_tasks`.

Step 4 (dataset and throughput): done.
- **Code in `_dataset.py`:**
  - `sample_uq_task`: steps 1-10, i.e. task + split + R replicates. A non-finite replicate revokes the task, which is redrawn whole on streams + 100k with the same shape.
  - `UQBatch`: one micro-batch group sharing (N, n_ctx), with X zero-padded to the group's max d.
  - `CutGraphPrior`: map-style Dataset. One item = one group, step = i // n_groups. Groups of 4 in stage 1 and 1 in stages 2-3, as tabicl's `batch_size_per_gp`. The group shape comes from `STREAM_SHAPE`, and the stage is in the RNG key via `stage_step_key`.
  - `make_dataloader` (tabicl's settings) and `iterate_steps`.
- **Gate:** `scripts/02_benchmark_prior.sh` (GPU node, 34 CPUs). It checks:
  - workers-0 vs W-worker bit-identity;
  - steady tasks/s per stage for W in {16, 32};
  - a per-component profile;
  - the frozen fp32 backbone forward per step, as the train-time floor.

  It passes if prior s/step <= 0.8 x backbone s/step. Output in `outputs/audits/throughput/` (v1: under the v1 root).
- **First run** (in-order loader, 0.5 gate): FAIL in all stages.
  - Stage 1 is CPU-bound: even perfect parallelism at 32 workers is slower than the backbone forward.
  - Stages 2-3 lost most of the workers' capacity to in-order delivery behind heavy-tailed large-N tasks.
  - The 34 CPUs are ~17 physical cores with 2 threads each (phoenix GPU nodes: 2 x 64 cores x 2 threads; SLURM allocates threads).
  - The stage-1 rejection rate was above step 2's 30% gate.
- **Fix applied** (user's decision; no change to cores, batch size or design): out-of-order delivery (`in_order=False`), `LargestFirstSampler`, `prefetch_factor` 4, and a per-step reorder buffer in `iterate_steps`. The gate was relaxed to prior <= 0.8 x backbone.
- **Rerun with the fix:**
  - bit-identity OK in every stage;
  - stages 2-3 pass at 32 workers (stage 3 with room to spare);
  - stage 1 still FAILS: it is CPU-bound, so stage 1 runs at the prior's pace, not the GPU's;
  - the stage-1 rejection rate is again above 30%.
- **User's decisions:**
  - stage 1 accepted as prior-bound (no offline stream, no core change);
  - batch halved to 32 tasks per step (`BATCH_SIZE` in `_dataset.py`; step budget unchanged);
  - stage-1 rejection rate above 30% accepted as a documented deviation (design Training §1).

LR schedule (user's decision): B's fork LR scales across stages by the same /8, /40 ratios as the head LR (design Training §2 and §6, proposal). No per-stage LR branching: the sweep stays one stage-1 value per LR.

Validation cadence (v1; user's decision): full bank every 5,000 steps in stage 1 + end of each stage; mini-bank every 250 steps (design Training §3 and §6, protocol §7.4–7.5 and §8.2). v2 adds full-bank validations in the middle of stages 2 and 3 (guide §3.2).

Step 5 (backbone wrapper): done, `00d_test_backbone` ALL OK. This includes the rewritten padded-group check, which forces two tasks with different d. `src/iltm2/model/_backbone.py`: `FrozenBackbone` loads TabICLv2 (`TabICL(**ckpt["config"])`, training mode, frozen; not an nn.Module) and runs tabicl's own `model(X, y_ctx)` under no_grad. It reads three things with forward hooks only: `h_star` (output of `icl_predictor.ln`), the 12 residual taps (outputs of `icl_predictor.tf_icl.blocks[l]`, query rows; A2 only, removed in v2), and arm B's fork input (full stream after block 12-k). `copy_last_blocks(k)` / `run_blocks` give arm B's trainable copy.
- tabicl turns FlashAttention-3 (fp16 attention) on whenever `flash_attn_interface` imports (`_use_flash_attn3 = True` at import; v2 pretraining used it in stages 2-3 only). It is not installed in uq-iltm-bio (and is Hopper-only), but `attention_ctx` forces it off anyway, so every forward is fp32 SDPA (TabICLv2's `--dtype float32`) with cuDNN SDPA excluded as in tabicl's trainer.
- The checkpoint loads from `/private/groups/ioannidislab/ajaras/tabicl-dev/tabicl-regressor-v2-20260212.ckpt` (`DEFAULT_BACKBONE`). The protocol §2 copy under `<root>/backbone/` was never made.

Step 6 (validation bank): done, gate passed (`01b_check_val_bank`: ALL OK on the full bank). v2 replaces this bank with `val_bank_v2` (guide §3.4).
- **Stored in the v1 root `/private/groups/ioannidislab/UQ_decoupling_exp/`:**
  - the bank in `val_bank/`: 2,000 tasks, 42 shards, `queries.pt`, `manifest.json` with the reference lines, `build_config.json`;
  - the 64-task mini-bank in `val_bank/mini/` (`tasks.pt`, `queries.pt`; its reference lines are in the manifest under `mini`);
  - the 14-task smoke bank (code test only, not for training) in `val_bank_smoke/`;
  - check outputs in `audits/val_bank_checks/<bank>/`.
- **Bank results** (t_E by query type, e(q) proxy skill) are in design Training §3, "What the bank showed".
- **Code:**
  - `src/iltm2/prior/_bank.py`: bins, `generate_bank_task`, `forecast_targets`, the `BankQueries` per-query table, sharded storage;
  - `src/iltm2/train/_metrics.py`: slices, MSE / xMSE / skill / ratio, reference lines. The trainer must reuse it so the `val/...` and `ref/...` keys match;
  - `src/iltm2/prior/_graph_scm_ref.py`: graph_scm reference tasks, moved out of 00f.
- **Scripts:**
  - builder `scripts/01_make_val_bank.sh` (GPU A5500, resumable shards; smoke: `--n-tasks 8 4 2 --out .../val_bank_smoke`);
  - checks `scripts/01b_check_val_bank.sh`: integrity, reproducibility, t_E ordering + t_A dup = iid, cut vs graph_scm CRPS ratio per N bin within 1.25.
- **Conventions:**
  - bank seed `VAL_BANK_SEED = 20261103`. Any test bank must use another seed;
  - "n bin" = the task's context size n_ctx (protocol §7.5).

Gotchas found while building steps 2–6 (still valid):
- In v2's `NumericalConverter` (`use_corrected_num_converters=False`) the returned column is a view of the node slice that is then overwritten with the write-back, so the observed numerical column *is* the warped write-back.
- Nodes pruned by tabicl are never fitted (no `converters_`); only feature-hosting nodes are guaranteed fitted.
- Bit-identical tasks across processes need the same torch intra-op thread count (1 in DataLoader workers).
- An X-graph latent can be NaN while its observed column is finite (e.g. a category index from argmin): its center then reaches the target graph. Non-finite target-graph inputs reject the task (`non_finite_x`). Never guard with `torch.equal` on tensors that may hold NaN (NaN != NaN); use the bitwise `_same_bits`.
- Replicates must be R separate passes over the same rows, never one pass over tiled rows: float32 round-off depends on a row's position in the batch, and tree splits / argmins amplify it into spurious replicate spread (up to 0.2 std(y) on deterministic tasks).
- A non-finite replicate is only seen after acceptance: `RejectionLog.revoke("non_finite_replicate", ...)` turns the task into a rejection, and the caller draws a new one (done in `00f` and in step 4's dataset).
- About 60% of accepted tasks are exactly deterministic (no private root reaches y, t_A ≡ 0); the `00b` tests that need noise pick a noisy task explicitly (`noisy_task`).

Step 7 (heads, loss, metrics, validation): done, `00e_test_heads` ALL OK.
- `src/iltm2/model/_heads.py`:
  - `Trunk`: 512 -> 1024 -> 1, softplus; zero last layer, bias softplus^-1(mean target);
  - `LayerMix`: A2, removed in v2;
  - `Head`, and `HeadPair`: one configuration with Â and Ê_dir heads, its own AdamW via `make_optimizer`, id like `A1_lr3e-4`;
  - `Fork`: B, a copy of the last k blocks + LN, with `distance_from_init`;
  - `build_a_job_heads`, `select_input`.

  Heads share no parameter. Init is seeded by (seed, arm group, head): identical across LRs, shared by A and B.
- `src/iltm2/train/_loss.py`: `batch_targets` (full + half-split targets of a group), `loss_weight` (exactly 1 for p = 0), `uq_loss`, `group_weight` (groups weighted by query rows).
- `src/iltm2/train/_metrics.py` (added): `head_metrics` (val/ and mini/ keys, skill against the bank's const xMSE), `train_metrics`, `EMA`.
- `src/iltm2/train/_validate.py`:
  - `BankEvaluator`: bank or mini-bank; caches h / taps / fork input or recomputes per task; checks backbone sha, recomputed e(q) vs stored, and task order;
  - `BestTracker`: lowest pooled val MSE per pair and head, full bank only.
- Gate: `scripts/00e_test_heads.sh` (CPU, short; uses the smoke bank's mini-bank and walks the real bank's records).
- User's decisions: gradient clipping per head (`--clip_per head`, default), output-bias init from the first batch (`--init_bias warmup`, default).
- Guard lesson: the bank's e(q) recomputed on another device differs by float32 round-off (a few 1e-6 absolute, up to ~3e-4 relative at queries with tiny e(q)), so `BankEvaluator` checks e(q) against each task's mean e(q) (1e-3), not per query; on the A5500 it is bit-equal.

Step 8 (A trainer): done, smoke run passed (`04_smoke_A`: ALL OK). The overfit gate is final train MSE <= 5% of the step-0 MSE; the batch's noise floor is only reported. Peak GPU memory and the projected A-job wall time were well inside the A5500 and the 4-day limit. One full-bank validation of the v1 bank took roughly 25 min.
- **Code:**
  - `src/iltm2/train/_config.py`: tabicl flag names; v1 per-stage defaults 50000/4000/1000 steps (v2: 30000/8000/1200); LR divisor 1/8/40.
  - `_base.py`, `BaseTrainer`:
    - stage start: skip if `final.pt`; resume from `latest.pt`; else start from the previous stage's `final.pt` with fresh optimizers; else a fresh stage 1;
    - per-step loop: mini-bank every 250 steps, full bank every 5000 + stage end;
    - saves best-per-head files + a run-level `best_index.json`, and `latest.pt` every 500 steps;
    - writes `trace.jsonl` and `summary.json` per stage;
    - reuses tabicl's `Timer` and `get_scheduler`.
  - `frozen.py`, `FrozenTrainer`: one backbone pass per group feeds all head pairs, each with its own gradient accumulated and weighted by query rows; `--overfit_steps` smoke mode.
- **Jobs (user's decision):** ONE SLURM job per run runs stages 1 -> 2 -> 3 in sequence (4 days on gpu). Each stage writes `final.pt` once (never overwritten), and the next stage starts from it. A resubmitted job skips finished stages and resumes the current one.
- **Smoke (user's decision):** ONE GPU job, `scripts/04_smoke_A.sh`, followed by `scripts/04b_check_smoke.py`:
  - overfit;
  - stage 1: 200 steps, stopped at 100 and relaunched as the resume test;
  - stage 2: 50 steps;
  - stage 3: 20 steps, with full validation on the real bank for timing.

  Bank-vs-stream consistency is read in the first A job (train xMSE EMA vs mini-bank xMSE).

Step 9 (B trainer): done. `src/iltm2/train/fork.py` (`ForkTrainer` on `BaseTrainer`):
- **Model:** frozen backbone with fork_k = 2 (no taps). `Fork` (copy of blocks 11-12 + LN) feeds one `HeadPair` named "B", so all B runs share the wandb keys `B/...`.
- **Optimization:**
  - two AdamW optimizers, each with its own tabicl schedule. The heads end at 1e-7 and the fork at 1e-8 (`--fork_cosine_lr_end`). tabicl's `get_scheduler` takes lr_init from `optimizer.defaults["lr"]`, so the fork needs its own optimizer + scheduler;
  - gradient checkpointing in the fork above 20K rows;
  - fork + heads clipped jointly (`--clip_per joint`; `separate` optional).
- **Checkpoints:** the best-A / best-E checkpoints store the whole model (fork + both heads). The mini-bank caches the fork input.
- **No B smoke job (user's decision).** At a fresh stage 1, each job runs `preflight` (one fork forward/backward at N = 60,000, peak memory) and `check_first_batch` (fork output = h_star within 1e-5), printing OK lines. A failure stops the job.

Inference: `src/iltm2/inference.py`, `ILTMv2Regressor(checkpoint_A, checkpoint_E=None, pair=None)`.
- **Methods:** fit / predict / predict_uncertainty, which return quantiles, median, mean, A, E and e_q in raw y units.
- **Checkpoints:** works with any arm (best files, or stage files + pair). `ILTMv2Regressor.from_best("A1"|"A2"|"B")` loads the v1 selected checkpoints (`BEST`, under `RUNS` = the v1 root).
- **Scaling and route:** context-only scaling of X and y as in the prior (clamp at mean +- 4 std, then standardize). It takes the raw training-mode route, the fork included (protocol §1.3 "Forward route").
- **Phase 0 adapter:** `experiment0_baseline/src/adapters/iltmv2.py`, with methods `iltmv2` = A2, `iltmv2_a1` = A1 and `iltmv2_b` = B. It needs updating for v2 (no A2).

Export for analysis: `scripts/05_export_runs.sh` writes val, train, reference, runs and alphas CSVs plus a README data dictionary (protocol §9.3). v1 output: `/private/groups/ioannidislab/UQ_decoupling_exp/audits/export/`.

## v1 results and selection (legacy checkpoints, guide §1 and §6)

The v1 sweep:
- A1/A2: 3 seeds × head LR {1e-4, 3e-4, 1e-3}, all in shared A jobs;
- B: seed 0, head LR × fork LR {3e-6, 1e-5, 3e-5};
- 50,000 / 4,000 / 1,000 steps per stage, $p=0$, one context per task.

All runs are under `/private/groups/ioannidislab/UQ_decoupling_exp/runs/`. The full analysis is in guide §1.

A1/A2 selection (protocol §9.1): per head, by seed-mean val MSE, then best seed. LR 1e-3 won everywhere, at the end of stage 3:
- A1: Â `A_seed1/stage3/best/A1_lr1e-3_A.pt` (val MSE 0.004041, skill 0.891), Ê `A_seed0/stage3/best/A1_lr1e-3_E.pt` (0.02117, skill 0.420; hole 0.447, extrapolation 0.474).
- A2: Â `A_seed1/stage3/best/A2_lr1e-3_A.pt` (0.003812, skill 0.898), Ê `A_seed1/stage3/best/A2_lr1e-3_E.pt` (0.02108, skill 0.423; hole 0.443, extrapolation 0.476).
- LR 1e-3 was the grid's top edge, so the optimum may lie above it. v2's grid moves up to {3e-4, 1e-3, 3e-3} for this reason.

B selection (protocol §9.2, seed 0): head LR 3e-4, fork LR 3e-5, `B_h3e-4_f3e-5_seed0/stage3/best/B_A.pt` / `B_E.pt` (same end-of-stage-3 state). Val MSE Â 0.003678 (skill 0.902), Ê 0.02058 (skill 0.436; hole 0.472, extrapolation 0.487). Best Ê of all 9 and best Â of the finished runs. Fork LR 3e-5 was the grid's top edge.
