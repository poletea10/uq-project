# Phase 1 — Model training: execution protocol

This protocol is the step-by-step guide for implementing and running Phase 1. It covers how the code is organised, where outputs are stored, the order of work with a gate after each step, the checks run before long jobs, and how runs are logged and read. The design itself (heads, targets, loss, arms, curriculum, sweeps) lives in `phase1_design.md` and is not repeated here. Where the two disagree, `phase1_design.md` wins on *what* is trained and this protocol wins on *how* it is built and run.

| Section | Content |
|---|---|
| §1 | Code organisation: where the code lives and what each file does |
| §2 | Storage: code, checkpoints, bank, logs |
| §3 | Order of work, with a gate per step |
| §4 | Unit tests |
| §5 | Split figures (after step 3) |
| §6 | Smoke runs before the long jobs |
| §7 | Logging (wandb) |
| §8 | Reading the curves: reference lines, rules of thumb, red flags |
| §9 | Launch plan |

---

## 1. Code organisation

### 1.1 Where the code lives

Phase 1 is a **separate experiment folder that imports TabICLv2**, never a set of files added inside `external/tabicl`.

- **Reuse comes from the editable install.** Everything needed is importable: `GraphSCM` and `graph_lib`, `should_filter`, `delete_unique_features`, `check_x_y_ancestors_overlap`, the model classes, `get_scheduler`, `build_parser`, `PriorConfig`, `seed_worker`, `Timer`.
- **The PPD must stay bit-identical to TabICLv2** (design, Arm A1). Phase 0's TabICLv2 rows import the same package. Any edit to tabicl's source silently changes both.
- **Hooks instead of edits.** Read internal activations with `register_forward_hook` or a subclass in our own code. If a patch is truly unavoidable, make it one small commit on a branch of `external/tabicl` and record it in `CLAUDE.md`.

The folder follows the project convention (one top-level folder per experiment with `CLAUDE.md`, `scripts/`, `src/`). It is installed as its own editable package, `iltm2`, so `experiment0_baseline` can import `iltm2.prior` to build the T2 test bank (`phase0_protocol.md` §3.1).

### 1.2 Layout

The layout mirrors tabicl's `prior/`, `_model/`, `train/` split and keeps tabicl's CLI argument names, so flags from `scripts/train_v2_reg_stage*.sh` carry over.

```
experiment1_uq_heads/
├── CLAUDE.md                  # points to phase1_design.md + this protocol; pinned tabicl commit
├── pyproject.toml             # package `iltm2`; pip install -e .
├── src/iltm2/
│   ├── prior/
│   │   ├── _cut_graph_scm.py  # CutGraphSCM: X-graph (graph_scm mechanism) + graph_scm-style target graph Ψ(X_obs, U2)
│   │   ├── _cached_graph.py   # graph_lib root law / node / graph functions with task-level draws cached
│   │   ├── _postprocess.py    # fit/apply versions of outlier_removing / standard_scaling; the y transform g
│   │   ├── _split.py          # extrapolation tail, kNN hole (distances to one centre), iid split, duplicates — X only
│   │   ├── _filter.py         # wraps check_x_y_ancestors_overlap (on the y-graph), delete_unique_features, should_filter
│   │   ├── _dataset.py        # task RNG, task shapes (pipeline steps 1–3), sample_task; CutGraphPrior: infinite
│   │   │                      #   iterator → UQBatch (mirrors tabicl GraphPrior.get_batch; step 4)
│   │   ├── _graph_scm_ref.py  # unmodified graph_scm tasks (the reference prior of 00f and the bank checks)
│   │   └── _bank.py           # build / save / load the validation bank (sharded, like tabicl's _genload.py)
│   ├── model/
│   │   ├── _entropy.py        # quantiles → CDF (one convention), e_crps(q), crps(q, y), in torch
│   │   ├── _backbone.py       # load + freeze TabICLv2; forward → (h_star, resid_taps, quantiles), via hooks
│   │   └── _heads.py          # Trunk (512→1024→1, softplus), HeadPair, LayerMix (A2), Fork (B)
│   └── train/
│       ├── _targets.py        # t_A, t_E, and the half-split targets used by xMSE (§7.3)
│       ├── _loss.py           # weighted MSE per head (p = 0 default)
│       ├── _metrics.py        # MSE, xMSE, EMA, reference lines, Â/e(q) ratio (§7–8)
│       ├── _validate.py       # bank and mini-bank evaluation, per head × query type × n bin
│       ├── _config.py         # tabicl build_parser() + our arguments
│       ├── _base.py           # shared trainer skeleton: loop, Timer, checkpoints, wandb
│       ├── frozen.py          # FrozenTrainer: one backbone pass → all A1 + A2 heads (one A job)
│       └── fork.py            # ForkTrainer: one arm-B configuration per job
└── scripts/
    ├── 00a_test_entropy.py    # unit tests §4.1  (each 00* script: OK/FAIL per check, .sh wrapper alongside)
    ├── 00b_test_cut_prior.py  # unit tests §4.2
    ├── 00c_test_split.py      # unit tests §4.3
    ├── 00d_test_backbone.py   # unit tests §4.4
    ├── 00e_test_heads.py      # unit tests §4.5
    ├── 00f_difficulty_match.py # prior-level gate of step 2: cut vs graph_scm difficulty (§3, step 2; GPU)
    ├── 01_make_val_bank.py    # bank + mini-bank + reference lines (§3, step 6)
    ├── 01b_check_val_bank.py  # the bank checks of §3 step 6 (+ integrity, reproducibility), after 01
    ├── 02_benchmark_prior.py  # throughput per stage (§3, step 4)
    ├── 03_plot_split_tasks.py # split figures (§5)
    ├── 04_smoke_A.sh          # the A trainer's smoke run, one GPU job (§6); ends with 04b_check_smoke.py
    ├── train_A.sh             # <seed>: one A job, stages 1 → 2 → 3 in sequence (§7.2, §9)
    └── train_B.sh             # <head_lr> <fork_lr> <seed>: one B job, same pattern (step 9)
```

### 1.3 Conventions

- **Trainer skeleton.** Do not subclass tabicl's `Trainer`: it is built around training the full model under DDP, AMP and Muon. Copy its loop skeleton into `_base.py` and keep its conventions: `Timer`, `prior_time` / `train_time` logging, `save_temp_every`, and the `RESUME_ARGS` restart pattern of `scripts/train_v2_reg_stage2.sh`.
- **Single GPU, no DDP.** The backbone runs under `torch.no_grad()`, so the only memory pressure is arm B at large N, which gradient checkpointing handles (design, Training §2).
- **`UQBatch`** is a small dataclass: `X, y_ctx, y_reps (m × R), query_type (m,), d, seq_len, train_size`. Tabicl's micro-batch rule still applies (`validate_micro_batch`): one `(seq_len, train_size)` per micro-batch, so the task shape is sampled per micro-batch group.
- **Deterministic task stream.** Each task's RNG depends only on `(seed, step, task_idx)`, never on the number of DataLoader workers. This makes A job *s* and the B jobs on seed *s* see the same tasks (design, Training §5) and makes resuming exact. Fix `PYTHONHASHSEED` in every job (proposal §4.3).
- **Forward route.** One raw training-mode forward, no `TabICLRegressor` ensemble, everywhere (design, Architecture §1, "Forward route").
- **Optimizer.** AdamW, weight decay 1e-5, gradient clipping 5.0, β = (0.9, 0.999), following DecoupledPFN App. B.3. Muon is not inherited from TabICLv2's scripts (design, Training §2). If tabicl flags are reused, pass `--muon False`.
- **Precomputed bank targets.** The PPD is frozen in every arm, so the bank's targets ($\hat t_A$, $\hat t_E$, their half-split versions) and $e(q_\theta)$ never change. They are stored once with the bank. Validation is then a backbone forward plus the heads. In A jobs, cache the bank's $h_*$ once at job start; validation then costs only the head MLPs. A2's per-layer taps are too large to cache and are recomputed.

---

## 2. Storage

Code lives in git under `~/iLTM-bio/UQ_project/experiment1_uq_heads`. Everything heavy lives on the group disk:

```
/private/groups/ioannidislab/UQ_decoupling_exp/
├── backbone/      TabICLv2 regressor checkpoint + sha256
├── val_bank/      2,000 tasks, sharded .pt + manifest.json (seed, git hashes, config, reference lines)
│   └── mini/      the 64-task mini-bank (§7.4)
├── runs/<run>/
│   ├── best_index.json        # where each head's current best checkpoint lives (stage, global step, val MSE)
│   └── stage{1,2,3}/
│       ├── latest.pt          # heads / fork + optimizers + schedulers + EMAs + best tracker + step (resume)
│       ├── final.pt           # end of the stage: written once, NEVER overwritten; the next stage starts from it
│       ├── best/<head_id>_{A,E}.pt   # best head so far (pooled full-bank val MSE), reached in this stage
│       ├── trace.jsonl        # per step: LR per pair, batch fingerprint, launch id, global step
│       └── summary.json       # timings, peak GPU memory, projected stage wall time
├── wandb/         WANDB_DIR
├── slurm_logs/
└── audits/        split figures (§5), throughput tables, smoke-run outputs
```

- **Checkpoints store only trainable parameters**, plus the backbone path and sha256, never a copy of the backbone. They are a few MB each.
- **The bank is the large object**: a few GB, mostly the 400 large tasks of up to 60K rows.
- **The training stream is not pre-generated.** Stage 1 alone is about 3.2M tasks, hundreds of GB. The deterministic on-the-fly stream (§1.3) gives the same pairing for free.

---

## 3. Order of work

Each step has a gate that must pass before the next one starts.

| # | Build | Gate |
|---|---|---|
| 1 | `_entropy.py`, `_targets.py` | Unit tests of §4.1 pass |
| 2 | `_cut_graph_scm.py`, `_cached_graph.py`, `_postprocess.py`, `_filter.py`, `_dataset.py` (task shapes) | Replicate tests of §4.2 pass (`00b`); rejection rate between 5% and 30% (stage 1's training stream slightly above it: accepted, design Training §1); difficulty-match gate passes (`00f_difficulty_match`) |
| 3 | `_split.py` | Split tests of §4.3 pass; split figures of §5 look right |
| 4 | `_dataset.py` + DataLoader, `02_benchmark_prior.py` | Throughput: prior time ≤ 0.8 × step time in every stage |
| 5 | `_backbone.py` | Backbone tests of §4.4 pass |
| 6 | `01_make_val_bank.py` | Bank checks below pass; reference lines stored in the manifest |
| 7 | `_heads.py`, `_loss.py`, `_metrics.py`, `_validate.py` | Head tests of §4.5 pass |
| 8 | `_config.py`, `_base.py`, `frozen.py`, `train_A.sh` | The smoke run of §6 passes (`04_smoke_A.sh`) → launch A jobs (§9) |
| 9 | `fork.py` + `train_B.sh` | Fork equals $h_*$ at step 0; smoke runs on B → launch B jobs (§9) |

The validation bank (step 6) comes **before** any trainer, because the trainer logs against it from step 0 and the reference lines are what make a loss value readable (§8).

### Step 1 — Entropy and targets

The estimators themselves were already validated in Phase 0 (`phase0_protocol.md` §10, gotcha 1), so they are not re-audited here. What is new is the torch implementation, so step 1 only checks that it reproduces Phase 0's numpy code (§4.1), under the single CDF convention of design §2: piecewise-linear through the 999 quantiles, 0 below the first and 1 above the last (`phase0_protocol.md` §5.4).

### Step 2 — Observed-cut prior and filter

Implements design Training §1 (pipeline steps 1–7, 10) and §1.4.

**Rejection-rate gate: 5%–30%.** TabICLv2 rejects about 25% of stage-1 regression tasks (design, Training §1.4). The observed cut removes the latent confounding between X and y and makes the observed X the only frozen-side input of the y-graph, so tasks are expected to be *more* predictable and the rate is expected to fall. Any rate in 5%–30% passes.

- **Above 30%:** the cut's private noise makes tasks noisier than the original prior. Tune the y-graph's noise scale, not the filter, because the filter defines the backbone's prior (design §1.4).
- **Below 5%:** check that the predictability test is actually wired in and runs on the $g$-transformed $y^{(0)}$.

Log the rate per stage and the empirical distribution of $\hat t_A$, including its zero-mass fraction (proposal §4.3, "Known caveat").

**Difficulty-match gate.** `00f_difficulty_match.py` (GPU) compares matched cut and unmodified `graph_scm` tasks (stages 1–2; stage 3 rejection rate only) on ExtraTrees held-out $R^2$ (difficulty profile) and on TabICLv2's advantage over ExtraTrees at matched difficulty (in-domain check), with the 5%–30% rejection gate; the pass rule is in design §1.4. It also logs both priors' rejection rates, TabICLv2 CRPS, the y-ancestor subgraph sizes and the $\hat t_A$ distribution. On a failure, tune the target graph's private-root noise (`--private-root-scale`, `--private-root-dims`), never the filters, and rerun. The calibrated defaults and their stage-1 results are in design Training §1, "The observed-cut prior". Record the final configuration and tolerance in design §1.4.

### Step 3 — Split

Implements design Training §1.1–1.2. Once `_split.py` passes its tests, run `03_plot_split_tasks.py` (§5) and inspect the figures before moving on.

### Step 4 — Dataset and throughput

`02_benchmark_prior.py` measures tasks per second for each stage's shapes with the planned number of DataLoader workers. A step consumes 32 tasks (`BATCH_SIZE`, design Training §2), so the target is a prior time per step of at most 0.8 × the step's train time (the backbone forward, a lower bound on an A-job step). The workers build tasks while the GPU trains, so a step costs the slower of the two; the remaining 20% absorbs step-to-step variation, with at least 2 steps of items in flight. If it is not met, raise `n_jobs` first, then lower the region bounds in stage 3 (design §1.1, "Cost of the enlarged pool").

**Gate outcome (accepted).** Stages 2–3 pass. Stage 1 does not: it is CPU-bound, since even perfect parallelism over the job's 34 CPU threads (about 17 physical cores) builds a step's tasks more slowly than the backbone consumes them. This was accepted with the core allocation unchanged: stage 1 runs at the prior's pace, and the batch was halved to 32 tasks (design Training §2) to shorten every stage. No offline pre-generation of the stream.

**Loader.** Task cost is heavy-tailed in N (in stage 3, from under a second to tens of seconds per task). An in-order DataLoader lets one slow task block the delivery of all later ones and idles the other workers. `make_dataloader` therefore delivers items out of order (`in_order=False`, new work only to workers with free capacity), builds each step's groups largest N first, and keeps `prefetch_factor` 4; `iterate_steps` is a per-step reorder buffer that yields each step's groups only when all are in, in group order. Every item is a pure function of (seed, stage, step, group), so the trainer's batches are bit-identical to an in-order loader's; the benchmark checks this.

### Step 5 — Backbone wrapper

Implements design Architecture §1 "Forward route", with hooks for $h_*$ and A2's residual taps. Before writing the A2 hooks, look up the attribute names of the TFicl blocks in `_model/encoders.py`.

### Step 6 — Validation bank

`01_make_val_bank.py` builds the 2,000-task bank (design, Training §3), stratified 50/30/20 by N, with R_val = 128, its own seed disjoint from all test banks. It also builds the 64-task mini-bank (§7.4) and computes the reference lines (§8.1). It stores per query: `query_type`, $n$ bin, $e(q_\theta)$, $\hat t_A$, $\hat t_E$, and the two half-split targets from replicates 1–64 and 65–128.

**Bank checks, before any training:**

- **Target ordering by query type.** Mean $\hat t_E$ should be ordered duplicate < iid < hole ≈ extrapolation. If the targets themselves do not show this, the split or the prior is broken.
- **The frozen PPD is sane on the cut prior.** Mean CRPS of TabICLv2's forecast on iid bank queries should be in the same range as on unmodified `graph_scm` tasks of the same shapes. A much worse value means the cut moves the prior further than assumed.

### Steps 7–9

Heads, loss, metrics and validation (step 7), then the two trainers (steps 8–9), each followed by the smoke runs of §6 before launching.

---

## 4. Unit tests

Project convention (root `CLAUDE.md`): no pytest. Each subsection below is one plain validation-gate script, `scripts/00{a..e}_test_*.py`, with a matching `.sh` SLURM wrapper (`short` partition, CPU, seconds each). Each script runs its checks, prints one `OK <check>` or `FAIL <check>: <detail>` line per check, and exits non-zero if any check failed. Submit all five before every launch and read the OK/FAIL lines in their logs.

| Script | Section |
|---|---|
| `00a_test_entropy.py` | §4.1 |
| `00b_test_cut_prior.py` | §4.2 |
| `00c_test_split.py` | §4.3 |
| `00d_test_backbone.py` | §4.4 |
| `00e_test_heads.py` | §4.5 |

### 4.1 Entropy and targets

- `e_crps(q)` and `crps(q, y)` in torch agree with Phase 0's numpy kernel to about 1e-5 on random sorted quantile sets and random `y`, including `y` outside the quantile range.
- `t_A` (sorted $O(R\log R)$ formula) and `t_E` reproduce Phase 0's numpy functions on random replicate sets.
- The half-split targets of `_targets.py` use disjoint replicates, and each half is computed with the same functions as the full target.

### 4.2 Observed-cut prior

- **X is bit-identical across replicates** (proposal §4.3, the construction's only correctness guard).
- Task-level objects are cached (`RandomNodeFunction._fit`): two replicate passes differ only through $U_2$.
- The y transform $g$ is fitted once on the pool. All replicates are transformed with the pool's bounds, mean and std, never refitted.
- Same `(seed, step, task_idx)` gives a bit-identical task, whatever the number of workers.

### 4.3 Split

- Counts are exact: $n_{\text{ctx}}$, $m_{\text{iid}}$, $m_{\text{hole}}$, $m_{\text{ext}}$, and $P-N$ discarded rows.
- Duplicates copy $x$ only: their $x$ equals a context row's $x$, and their replicates are fresh draws, never the context label.
- **The split never looks at y:** calling it with two different y vectors on the same X gives identical indices.

### 4.4 Backbone

- No backbone parameter has `requires_grad`. The backbone's state-dict hash is unchanged after an optimizer step.
- The wrapper's quantiles are `torch.equal` to tabicl's own training-mode forward on the same inputs.
- $h_*$ equals the final LN applied to the last TFicl block's output at the query rows. A2 receives 12 taps of the expected shape.
- B: the fork's output is `allclose` to $h_*$ at step 0, and the decoder input is untouched.

### 4.5 Heads and loss

- Initialization: last-layer weights are zero and its bias is softplus⁻¹ of the mean target (design, Architecture §1). At step 0 every query gets the same output.
- **Head independence in A jobs.** Stepping one head pair leaves the other five unchanged. This catches a shared-optimizer bug.
- With p = 0 the weight is exactly 1.
- Gradients reach the heads (and in B the fork) and nothing else.

---

## 5. Split figures (after step 3)

`03_plot_split_tasks.py` draws ten figures, one per random task, to check by eye that the cut-graph tasks are heteroscedastic and that the split does what it should. The layout follows TabICLv2's Figure F.11: a 2-D scatter with the third quantity encoded by colour.

**Tasks.**

- Ten tasks from the observed-cut prior with exactly **2 features** and the stage-1 shape (N = 1,024), from a fixed figure seed disjoint from all others. Ten rather than five because about 60% of tasks are exactly deterministic (no private root reaches y); there is **no selection on noise**, so the figures show the real mix, and each title says whether the task is noisy or deterministic.
- Keep only tasks whose two features each have at least 10 unique values, as TabICLv2 does for its prior plots (Figure F.1 caption). Otherwise categorical columns stack points on top of each other.
- The full pipeline runs: pool, $g$, filter, split, duplicates.
- For the figure only, **every pool row** gets $R_{\text{fig}} = 128$ replicates, so the aleatoric entropy can be shown everywhere, not just at queries.

**One figure per task, three panels side by side, same axes ($x_1$, $x_2$):**

| Panel | Points | Colour |
|---|---|---|
| 1 — Target | All $P$ pool rows | $g(y^{(0)})$, sequential colormap with a colourbar |
| 2 — Aleatoric spread | All $P$ pool rows | $\hat t_A$ from the 128 replicates, min–max scaled per task to [0, 1] (0 = lowest, 1 = highest spread in that task); a deterministic task ($\hat t_A\equiv0$) shows a flat panel saying so |
| 3 — Split | All $P$ pool rows | **blue**: context; **red**: queries; **purple**: duplicate queries; **light grey**: discarded hole/tail rows |

Panel 3 details:

- **Draw order:** grey, blue, red, purple, so duplicates stay visible on top of the context point they copy.
- **Marker shape per query type:** circle for iid, square for hole, triangle for extrapolation. This makes the carved regions readable without extra colours.
- **Discarded rows are shown in grey** because they belong to neither group. Leaving them out would make the hole and tail look like empty space.
- **Hole centre:** the pool row the kNN hole is built around, drawn as a large star.
- **Extrapolation direction:** the random direction $u$, drawn as an arrow from the pool's centroid, and the tail boundary, a dashed line perpendicular to $u$ at the smallest tail score; the tail lies beyond it on the arrow's side. With $d = 2$ both are exact, not projections.

**What to look for:**

- **Panel 2 is not uniform** in most noisy tasks, i.e. the noise is heteroscedastic. A task whose panel 2 is flat is homoscedastic, which is allowed but should not be the norm. With R = 128 the scaled values carry about 6% noise, so small speckle is expected.
- **Panel 2 does not track the split.** $A$ is a property of $x$ in the task, fixed before the split, so the hole and tail are not systematically high-$A$ regions.
- **Panel 3:** the hole is one compact region around the star with no blue points. The tail is the region beyond the dashed line, in the arrow's direction. Purple points sit on blue points. Red iid points are scattered uniformly among the blue.

Save to `audits/split_figures/task{01..10}.png`. These figures also go into the thesis as the illustration of the observed-cut prior and the query mixture.

---

## 6. Smoke runs before the long jobs

**One GPU job** (`scripts/04_smoke_A.sh`, about 1–2 h), because every GPU job waits in the queue: runs 1–3 below run in sequence inside it and it stops at the first failure, so an `afterok` dependency can hold the A jobs on it. It must pass before the A jobs train; for B the same job is repeated with the fork trainer (step 9). Run 4 is read in the first A job, not in a separate job.

1. **Overfit one fixed batch.** The first stage-1 micro-batch group with a noisy task, backbone outputs computed once, every pair trained 3,000 full-batch steps. Each head's training MSE must fall to at most 5% of its step-0 value, the constant-baseline MSE every head starts from: the heads can memorize their targets from the embeddings, so the plumbing from batch to backbone to heads to loss works. The batch's replicate-noise floor $\overline{\hat t^{\,2}-\hat t^{(a)}\hat t^{(b)}}$ is reported, not gated: deterministic tasks have noise-free $\hat t_E$, so the $\hat t_E$ floor sits far below $\hat t_E$'s spread, and a floor-based bar would ask the E head for a far larger reduction than the A head.
2. **Short stages at real sizes:** stage 1 for 200 steps, stage 2 for 50, stage 3 for 20 (N up to 60K; for B, with gradient checkpointing), each starting from the previous stage's `final.pt`. Record peak GPU memory, seconds per step, `prior_time` vs `train_time`, and the time of one full-bank validation (stage 3 validates on the real bank; stages 1–2 on the small smoke bank, to keep the job short), then extrapolate the wall time of a full job. The step-4 benchmark measured the prior and the bare backbone forward; this run adds what it could not: the heads' forward and backward for all pairs, the 12 taps in memory, validation and checkpointing. Logs to the `iLTMv2` wandb project as run `smoke_A_<jobid>` in group `smoke` (tag `smoke`), which also checks the logging of §7; the real A jobs log to the same project in group `A`.
3. **Resume**, inside run 2 at no extra cost: stage 1 is stopped after 100 steps (`--stop_after 100`, which saves `latest.pt` as a killed job would) and relaunched. It must continue at step 100 with every pair's LR on tabicl's schedule, the batches of steps 100 and 101 bit-identical to the same steps rebuilt from (seed, stage, step), and the same wandb run (§7.1).
   `scripts/04b_check_smoke.py` checks runs 2–3 at the end of the job, and also that each stage's `final.pt` is unchanged after the later stages (never overwritten) and that the stages chain (each starts from the previous `final.pt`, global steps continuous).
4. **Bank-versus-stream consistency**, read in the first A job (seed 0) after the first few thousand stage-1 steps: no separate job. Compare the EMA of the training xMSE (§7.3) with the mini-bank xMSE (§7.4: 64 tasks of N = 1,024, the stage-1 shape, logged every 250 steps), per head and pair.
   - **Why both sides are comparable.** Both are noise-free in expectation, so no floor correction is needed. Training losses are computed **before** the optimizer step, so each training value is an evaluation on tasks the head has not yet trained on.
   - **What the test detects.** Each training task is seen once, so there is no stream to overfit. A gap means the bank and the stream are generated differently: a different shape distribution, filter, query mixture or transform.
   - **Gate:** the two curves agree within their noise.
   - **For the results:** their agreement shows that the heads learned a function of the prior that generalizes to fresh tasks, not properties of specific tasks. Report it alongside the T2 grades.

---

## 7. Logging (wandb)

### 7.1 One wandb run per job across all three stages

Each job (A job *s*, or one B configuration) is **one wandb run spanning stages 1, 2 and 3**, so each curve is continuous across the curriculum.

- **Fixed run id.** `--wandb_id` is the run name (e.g. `A_seed0`, `B_h3e-4_f1e-5_seed0`), and `wandb.init(id=..., resume="allow")` reattaches every launch of every stage to the same run. tabicl's parser already has `--wandb_id`.
- **Global step.** Logged steps are offset by the earlier stages: stage 1 runs 0–50,000, stage 2 runs 50,000–54,000, stage 3 runs 54,000–55,000. Log `stage` as a metric, so stage boundaries are visible.
- **Config.** `vars(args)` plus the git hashes of `iltm2` and tabicl and the backbone's sha256.
- **Key namespacing in A jobs.** One process trains 6 head pairs, so every metric is prefixed by the head id, e.g. `A1_lr3e-4/val/xmse_E/hole`.

### 7.2 Script

**One SLURM job per A seed runs the three stages in sequence** (`scripts/train_A.sh <seed>`), each stage as its own `python -m iltm2.train.frozen --stage s` process. Three jobs chained with `afterok` would wait in the queue three times, and on this cluster a job held by a dependency gains no queue priority while it waits; one job waits once. The job asks for 4 days of the `gpu` partition (7-day limit): a full A job is projected at about 2.5 days (the smoke run prints the measured projection).

- **Per-stage settings come from the trainer, not the script** (`iltm2.train._config`): steps 50,000 / 4,000 / 1,000, LR divisor 1 / 8 / 40, and the stage's shapes from `iltm2.prior._dataset.STAGES`. The other defaults are design Training §6 (AdamW, weight decay 1e-5, clipping 5.0, `cosine_with_restarts` with 1% warmup to `cosine_lr_end` 1e-7 via tabicl's own `get_scheduler` (B's fork: 1e-8, `--fork_cosine_lr_end`), clipping per head (`--clip_per head`), output biases from the first batch (`--init_bias warmup`), `--muon False`). Extra flags after the seed are passed to every stage.
- **How a stage starts** (`iltm2.train._base`): if its `final.pt` exists it is done and skipped; if its `latest.pt` exists it resumes with the full optimizer state; otherwise stage s > 1 loads the heads (and the best tracker) from stage s−1's `final.pt` with fresh optimizers and schedulers, tabicl's `--only_load_model` restart (`scripts/train_v2_reg_stage2.sh`). So after a timeout or a crash the same `sbatch` command continues where the job stopped.
- **The global step** of a stage starts at the previous stage's final global step (§7.1).
- **wandb key:** kept in a netrc of its own, `~/.wandb_uq/netrc`, created once with `mkdir -p ~/.wandb_uq && NETRC=~/.wandb_uq/netrc wandb login`. The job scripts export `NETRC` to it (wandb reads `$NETRC` before `~/.netrc`), so the shared `~/.netrc`, which holds another account's key on this login, is never read or overwritten; `WANDB_NETRC=<path>` selects another file. Never pass the key as an argument (visible to other users through `scontrol show job`) or write it into a committed file.

```bash
sbatch -J A_seed0 experiment1_uq_heads/scripts/train_A.sh 0
STAGES="2 3" sbatch -J A_seed0 experiment1_uq_heads/scripts/train_A.sh 0      # a subset of the stages
```

`train_B.sh` follows the same pattern, with the fork LR and one head LR, and runs `iltm2.train.fork` (step 9).

### 7.3 Training metrics (every step, per head)

**Raw MSE and xMSE.** Both are logged:

- `train/mse_A`, `train/mse_E`: the unweighted MSE of each head against its full 16-replicate target, the quantity the loss minimizes.
- `train/xmse_A`, `train/xmse_E`: the **cross-product MSE**. Split each query's 16 replicates into two disjoint halves of 8, build one target from each half, $\hat t^{(a)}$ and $\hat t^{(b)}$, and take

$$
\text{xMSE}=\overline{(\hat A-\hat t_A^{(a)})\,(\hat A-\hat t_A^{(b)})}.
$$

  For $\hat E_{\text{dir}}$, each half computes its own $\frac18\sum_r S(q_\theta,y^{(r)})-\hat t_A^{(\text{half})}$.

**Why xMSE.** The raw training MSE is the head's true error plus a large constant: the variance of a 16-replicate target, which for $\hat t_E$ exceeds $E$ itself. The real decrease is a small dip on top of that offset. The two halves are independent draws from the same task law and each is unbiased, so their noise is uncorrelated and the product's expectation is exactly the head's error against the realized task,
$\mathbb E[(\hat A-A_\tau)^2]$, with the replicate-noise term gone. **xMSE is the curve that shows the downward trend.** It costs nothing extra (the loss still uses the full 16-replicate target), but single values can be negative and only its average is meaningful.

**EMA.** `train/xmse_A_ema`, `train/xmse_E_ema`: an exponential moving average of xMSE with smoothing 0.99 (a window of about 100 steps), bias-corrected at the start of each stage. It removes the per-step fluctuations from task-to-task variation, so the trend can be read between validations.

**Other training metrics:**

- loss (weighted), learning rate, gradient norm per head;
- `prior_time`, `train_time`, rejection rate;
- means of $\hat t_A$, $\hat t_E$, $\hat A$, $\hat E_{\text{dir}}$, $e(q_\theta)$, and the fraction of $\hat t_E<0$;
- A2: the softmax weights $\alpha_\ell$ per layer (a diagnostic only, not a localisation of the information; design Architecture §4, "Arm A2");
- B: $\lVert\theta_{\text{fork}}-\theta_{\text{fork},0}\rVert$.

### 7.4 Mini-bank (every 250 steps)

64 small tasks (N = 1,024) from the validation bank, with their R_val = 128 targets. They are the same tasks at every evaluation, so the comparison between checkpoints is paired and the curve is far smoother than the training one. Logged: `mini/mse_*` and `mini/xmse_*` per head. Cost: one backbone forward on 64 small tasks (cached for A jobs) plus the heads.

### 7.5 Validation (every 5,000 steps in stage 1, and at the end of each stage)

Per head, per query type (pooled, iid, hole, extrapolation, duplicate) and per n bin (≤512, 512–1K, 1K–4K, 4K–16K, 16K–60K; n is the task's context size $n_{\text{ctx}}$), as in design Training §3. Slices, metric definitions and key names are shared with the bank's reference lines (`iltm2.train._metrics`): `val/<metric>_<head>/<slice>` next to `ref/<line>_<head>/<slice>`:

- `val/mse_*` against the R_val = 128 target, and `val/xmse_*` from the two halves of 64;
- `val/skill_*` (§8.1);
- **`val/ratio_A_eq/<type>`**: the per-type mean of $\hat A/e(q_\theta)$, both clipped below at $\epsilon=10^{-6}$ so that queries where both are 0 do not divide by zero. Log beside it the same ratio for the target, `val/ratio_tA_eq/<type>`, the mean of $\hat t_A/e(q_\theta)$. It is an unbiased per-type reference, since $e(q_\theta)$ is fixed per query. (Optional: the same pair for $\hat E_{\text{dir}}$.)
- the three reference lines of §8.1, logged as constants at every validation step, so they draw as flat lines on the same panels;
- "is best" flags for the best-$\hat A$ and best-$\hat E_{\text{dir}}$ checkpoints.

---

## 8. Reading the curves

### 8.1 Reference lines

A raw value like `val/mse_E = 0.0031` cannot be read on its own: it depends on the target's scale and on how noisy the targets are. Three numbers, computed once on the bank with no training, put every curve on a scale. They answer three questions: how low can it go, where did it start, and does it beat what the backbone already offers for free. They are stored in the bank manifest, per head and per query type, and drawn as flat lines on the validation panels.

| Line | What it is | What it tells you |
|---|---|---|
| **Noise floor** | The part of the raw validation MSE that is replicate noise: $\overline{\hat t^{\,2}-\hat t^{(a)}\hat t^{(b)}}$ over the bank's queries (exact in expectation, by the same independence argument as xMSE) | How much of the raw MSE no head can ever remove. It belongs on the raw-MSE panel only: on the xMSE panel the noise is already gone and the floor is 0 |
| **Constant baseline** | MSE (and xMSE) of predicting the bank-wide mean target for every query, evaluated per query type | Where every head **starts**: the zero-init + bias-at-mean-target initialization outputs exactly this. Moving below it is the first sign of learning |
| **e(q) proxy** | MSE (and xMSE) of the best predictor that only sees $e(q_\theta)$: 50 equal-count bins of $e(q_\theta)$, each predicting its bin's mean target, fitted on half the bank and scored on the other half | What the backbone's total uncertainty gives for free. The thesis claims the heads extract something $e(q_\theta)$ does not carry (proposal §1.1); the backbone check already showed $e(q_\theta)$ missing the divergence in sparse regions (design, Architecture §1). Beating this line is the claim |

The floor is a lower bound, not a reachable value. The irreducible spread of the realized truth over tasks consistent with the context sits on top of it and needs no oracle to be accounted for, because it is identical for every head and run (design, Architecture §4, "Comparing rungs needs no ground truth").

**Skill** turns the xMSE curve into one number per head and query type:

$$
\text{skill}=1-\frac{\text{xMSE}_{\text{head}}}{\text{xMSE}_{\text{const}}}.
$$

0 is the starting point, 1 is perfect, and the e(q) proxy has its own skill to compare against.

**Why per query type.** The split matters most where $e(q_\theta)$ rises for epistemic reasons: holes and extrapolation. A head can beat the lines on the pooled bank (dominated by iid queries) and still fail exactly there. Per-type lines show which regime is solved.

### 8.2 Rules of thumb

These thresholds are heuristics, not derived results. The grade that counts is T2's slope and intercept (`phase0_protocol.md` §6).

- **First 2,500 steps (mini-bank curve, §7.4):** both heads below the constant baseline. The first full validation comes at step 5,000; it should confirm this on the whole bank.
- **End of stage 1:** both heads clearly below the e(q) proxy, pooled and on hole and extrapolation queries.
- **Skill:** around 0.7 or more for $\hat A$ suggests it works. For $\hat E_{\text{dir}}$, 0.3–0.6 is plausible. $\hat E_{\text{dir}}$ skill below about 0.1 is not a bug: it is the ladder's signal that $h_*$ lacks the information, which is what A2 and B test (design, Architecture §4).
- **Bias:** mean $\hat E_{\text{dir}}$ against mean $\hat t_E$ on the bank within a few percent.
- **Per query type:** $\hat E_{\text{dir}}$ ordered duplicate < iid < hole ≈ extrapolation, following the targets' own ordering (§3, step 6).
- **Ratio $\hat A/e(q_\theta)$:** follows the target's ratio per type. In particular it **drops** on hole and extrapolation queries relative to iid, where $e(q_\theta)$ rises for epistemic reasons but the true $A$ stays at the local noise level.
- **Train xMSE EMA vs val xMSE (small bin):** agree within noise (§6, run 4).
- **Stages 2–3:** the large-n bins improve while the ≤ 1K bins do not degrade.

### 8.3 Red flags

- **Head collapsed to a constant.** Val xMSE stays on the constant baseline, and the standard deviation of the head's output across bank queries is near 0. Causes, in order of likelihood: targets too noisy for the signal (the MSE-optimal response to pure noise is to stay at the mean), learning rate too small to leave the initialization, or $h_*$ lacking the information. The A1 → A2 → B results tell these apart.
- **$\hat A$ just reads total spread.** No gain over the e(q) proxy **and** a flat $\hat A/e(q_\theta)$ ratio across query types. A high correlation between $\hat A$ and $e(q_\theta)$ alone is *not* a red flag: on iid queries most of the total is aleatoric, so a correct $\hat A$ should correlate strongly with $e(q_\theta)$ there.
- **Seen-bit not learned.** $\hat E_{\text{dir}}$ on duplicates ≈ on iid queries.
- **Bank and stream disagree.** Train xMSE EMA and val xMSE on the small bin differ by more than their noise (§6, run 4).
- **Broken stream.** Rejection rate outside 5%–30%, or a gap in mean $\hat t_A$ between duplicate and iid queries (§3, step 6).

---

## 9. Launch plan

Following design Training §4–5. Every job runs stages 1 → 2 → 3 in sequence inside one SLURM job, into one wandb run (§7.1, §7.2). The smoke run (§6) goes first; the A jobs can be submitted at the same time, held on it by a dependency, so they start as soon as it passes and never if it fails:

```bash
SMOKE=$(sbatch --parsable experiment1_uq_heads/scripts/04_smoke_A.sh)
for S in 0 1 2; do
  sbatch -J A_seed$S --dependency=afterok:$SMOKE --kill-on-invalid-dep=yes experiment1_uq_heads/scripts/train_A.sh $S
done
```

| Round | Jobs | Command |
|---|---|---|
| 1 | A, seeds 0, 1, 2 (each: 3 A1 + 3 A2 head pairs) | `sbatch -J A_seed<s> train_A.sh <seed>` |
| 1 | B, seed 0, 9 configurations (head LR × fork LR) | `sbatch train_B.sh <head_lr> <fork_lr> 0` |
| 2 | B's winner on seeds 1, 2, only if B clearly improves on A1/A2 | `sbatch train_B.sh <head_lr> <fork_lr> <seed>` |
| (cond.) | B's winner with Muon on the fork's 2-D matrices, only if B wins (design, Training §2) | `train_B.sh` with `--fork_muon True` |

Selection (per head within A1/A2, whole configuration in B, arm by mixed score, then best seed) follows design Training §3–4 and reads the **validation** metrics of §7.5. After selection, iLTMv2 runs on every Phase 0 test (proposal §4.10), through the same forward route.
