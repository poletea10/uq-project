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
| §9 | Launch plan, selection, export |
| §10 | Legacy checkpoints (first training round) |

---

## 1. Code organisation

### 1.1 Where the code lives

Phase 1 is a **separate experiment folder that imports TabICLv2**, never a set of files added inside `external/tabicl`.

- **Reuse comes from the editable install.** Everything needed is importable: `GraphSCM` and `graph_lib`, `should_filter`, `delete_unique_features`, `check_x_y_ancestors_overlap`, the model classes, `get_scheduler`, `PriorConfig`, `seed_worker`, `Timer`.
- **The PPD must stay bit-identical to TabICLv2** (design, Arm A). Phase 0's TabICLv2 rows import the same package. Any edit to tabicl's source silently changes both.
- **Hooks instead of edits.** Read internal activations with `register_forward_hook` or a subclass in our own code. If a patch is truly unavoidable, make it one small commit on a branch of `external/tabicl` and record it in `CLAUDE.md` (which also pins the tabicl commit; no patch exists).

The folder follows the project convention (one top-level folder per experiment with `CLAUDE.md`, `scripts/`, `src/`). It is installed as its own editable package, `iltm2` (`pip install -e experiment1_uq_heads --no-deps`), so that `experiment0_baseline` can run the trained model on the Phase 0 tests through `iltm2.inference.ILTMv2Regressor` (its iLTMv2 adapter, `experiment0_baseline/src/adapters/iltmv2.py`).

### 1.2 Layout

The layout mirrors tabicl's `prior/`, `_model/`, `train/` split and keeps tabicl's CLI argument names, so flags from `scripts/train_v2_reg_stage*.sh` carry over.

```
experiment1_uq_heads/
├── CLAUDE.md                  # status, gate results, pinned tabicl commit; points to the two specs
├── phase1_design.md, phase1_protocol.md
├── pyproject.toml             # package `iltm2`; pip install -e . --no-deps
├── outputs -> /private/groups/ioannidislab/UQ_decoupling_exp_new   (symlink, §2)
├── src/iltm2/
│   ├── paths.py               # ROOT (every write), RUNS_ROOT, LOG_DIR, LEGACY_ROOT (read only, §10)
│   ├── inference.py           # ILTMv2Regressor: fit / predict / predict_uncertainty; BEST, LEGACY_BEST
│   ├── prior/
│   │   ├── _cut_graph_scm.py  # CutGraphSCM: X-graph (graph_scm mechanism) + graph_scm-style target graph Ψ(X_obs, U2)
│   │   ├── _cached_graph.py   # graph_lib root law / node / graph functions with task-level draws cached
│   │   ├── _postprocess.py    # fit/apply versions of outlier_removing / standard_scaling; the y transform g
│   │   ├── _split.py          # extrapolation tail, kNN hole, thinning, iid split, duplicates (X only);
│   │   │                      #   subsample_context (stage 1's second pass); the six query types
│   │   ├── _filter.py         # wraps check_x_y_ancestors_overlap (on the y-graph), delete_unique_features, should_filter
│   │   ├── _dataset.py        # task RNG, the two priors (carved, iid), task shapes (pipeline steps 1–3, thinning),
│   │   │                      #   sample_uq_task, UQBatch (+ its subsampled pass), CutGraphPrior, the DataLoader
│   │   ├── _graph_scm_ref.py  # unmodified graph_scm tasks (the reference prior of 00f and the bank checks)
│   │   └── _bank.py           # build / save / load the validation bank (sharded); LOSS_C, VAL_BANK_SEED
│   ├── model/
│   │   ├── _entropy.py        # quantiles → CDF (one convention), e_crps(q), crps(q, y), in torch
│   │   ├── _backbone.py       # load + freeze TabICLv2; forward → (quantiles, e_q, h_star, fork_input), via hooks
│   │   └── _heads.py          # Trunk (512→1024→1, softplus), Head, HeadPair (arm, LR, p), Fork (B), build_a_job_heads
│   └── train/
│       ├── _targets.py        # t_A, t_E, and the half-split targets used by xMSE (§7.3)
│       ├── _loss.py           # weighted MSE per head, w = (e(q) + c)^-p, p per head pair
│       ├── _metrics.py        # slices, MSE / xMSE / skill in both weightings, EMA, reference lines, Â/e(q) ratio (§7–8)
│       ├── _validate.py       # BankEvaluator (bank and mini-bank, cached inputs, guards), BestTracker
│       ├── _config.py         # our parser with tabicl's argument names; per-stage defaults, grids
│       ├── _base.py           # shared trainer skeleton: loop, Timer, stage start / resume, checkpoints, wandb
│       ├── frozen.py          # FrozenTrainer: one backbone pass → every A head pair, head LR × p (one A job)
│       └── fork.py            # ForkTrainer: one arm-B configuration (head LR, fork LR, p) per job
└── scripts/                   # each NN_name.py has an NN_name.sh SLURM wrapper alongside
    ├── 00a_test_entropy.py    # unit tests §4.1  (each 00* script: OK/FAIL per check)
    ├── 00b_test_cut_prior.py  # unit tests §4.2
    ├── 00c_test_split.py      # unit tests §4.3
    ├── 00d_test_backbone.py   # unit tests §4.4
    ├── 00e_test_heads.py      # unit tests §4.5
    ├── 00f_difficulty_match.py # prior-level gate of step 2: cut vs graph_scm difficulty (§3, step 2; GPU)
    ├── 01_make_val_bank.py    # bank + mini-bank + weight floor c + reference lines (§3, step 6)
    ├── 01b_check_val_bank.py  # the bank checks of §3 step 6 (+ integrity, reproducibility), after 01
    ├── 02_benchmark_prior.py  # throughput per stage (§3, step 4)
    ├── 03_plot_split_tasks.py # split figures (§5)
    ├── 04_smoke_A.sh          # the A trainer's smoke run, one GPU job (§6); ends with 04b_check_smoke.py
    ├── 05_export_runs.py      # the runs' key metrics as CSVs for analysis (§9.3)
    ├── train_A.sh             # <carved|iid> <seed>: one A job, stages 1 → 2 → 3 in sequence (§7.2, §9)
    └── train_B.sh             # <head_lr> <fork_lr> <p> <carved|iid> <seed>: one B job, same pattern (§9)
```

### 1.3 Conventions

- **Trainer skeleton.** Do not subclass tabicl's `Trainer`: it is built around training the full model under DDP, AMP and Muon. Its loop skeleton is copied into `_base.py` with its conventions: `Timer`, `prior_time` / `train_time` logging, `save_temp_every`, and the `RESUME_ARGS` restart pattern of `scripts/train_v2_reg_stage2.sh`.
- **Single GPU, no DDP.** The backbone runs under `torch.no_grad()`, so the only memory pressure is arm B at large N, which gradient checkpointing handles (design, Training §2; in the fork above 20K rows, `--recompute_above`).
- **`UQBatch`** is a small dataclass: `X, y_ctx, y_reps (m × R), query_type (m,), d, seq_len, train_size`, plus the group's stream position, its deterministic-task flags and, in stage 1, `sub`: the same tasks' subsampled pass, itself a `UQBatch` (`passes` lists the forwards of a group). Tabicl's micro-batch rule still applies (`validate_micro_batch`): one `(seq_len, train_size)` per micro-batch, so the task shape is sampled per micro-batch group, and the subsampled context size $n'$ is drawn once per group.
- **Deterministic task stream.** Each task's RNG depends only on `(seed, stage, step, task_idx)` and a stream id (task, split, replicates, group shape, subsampled size, subsampled rows), never on the number of DataLoader workers. This makes the A job and the B jobs on the same seed and prior see the same tasks (design, Training §5) and makes resuming exact. The `iid` prior draws from the same task distribution, not the same tasks: without carving, the pool size $P$ and the shape's RNG draws differ. Fix `PYTHONHASHSEED` in every job (proposal §4.3).
- **Forward route.** One raw training-mode forward, no `TabICLRegressor` ensemble, everywhere (design, Architecture §1, "Forward route"): training, validation, the bank, and inference (`iltm2.inference.ILTMv2Regressor`, Phase 0).
  - *"Training mode" is only PyTorch's module flag.* Every forward runs under `torch.no_grad()` with no optimizer step, so nothing is trained, and dropout is 0 in TabICLv2, so the forward is deterministic.
  - *Why not eval mode.* In eval mode tabicl's `ICLearning.forward` switches to `_inference_forward`, a different path built for `TabICLRegressor`: automatic batching, optional CPU offloading and **mixed precision** (`use_amp=True` by default). It is a cheaper, lower-precision approximation of the same function, not a more correct one. The training-mode path is the fp32 computation TabICLv2 was pretrained with (FlashAttention-3 is forced off; every forward is fp32 SDPA with cuDNN SDPA excluded, as in tabicl's trainer).
  - *Why it must stay the same everywhere.* The heads' targets, the validation bank and every validation number are computed on this exact forecast, and $\hat E_{\text{dir}}$ is trained to predict *its* divergence. A forward in eval mode would perturb $q_\theta$ and $h_*$ by mixed-precision rounding, and the heads would be evaluated against a forecast they were never trained on. Switching to eval mode would require rebuilding the bank and retraining, and would gain no accuracy.
  - *Which modules it concerns.* `FrozenBackbone` refuses an eval-mode backbone. B's fork (copies of tabicl blocks) runs in training mode in training, validation and inference alike. The A heads (Linear, GELU, softplus) behave identically in both modes.
- **Optimizer.** AdamW, weight decay 1e-5, gradient clipping 5.0, β = (0.9, 0.999), following DecoupledPFN App. B.3. Muon is not inherited from TabICLv2's scripts (design, Training §2). The trainers accept `--muon` only for tabicl compatibility and refuse `True`.
- **Precomputed bank targets.** The PPD is frozen in every arm, so the bank's targets ($\hat t_A$, $\hat t_E$, their half-split versions), $e(q_\theta)$ and the weight floor $c$ never change. They are stored once with the bank. Validation is then a backbone forward plus the heads. A jobs cache the bank's and the mini-bank's $h_*$ once per launch (about 6 GB of CPU RAM for the bank), so an A validation costs only the head MLPs. B jobs cache the fork input on the mini-bank only; on the full bank it is recomputed task by task (tens of GB, too large to cache).

---

## 2. Storage

Code lives in git under `~/iLTM-bio/UQ_project/experiment1_uq_heads`. Everything heavy lives on the group disk, under `iltm2.paths.ROOT`, which `experiment1_uq_heads/outputs` symlinks to. Every write of every script goes there:

```
/private/groups/ioannidislab/UQ_decoupling_exp_new/           (iltm2.paths.ROOT)
├── backbone/      TabICLv2 regressor checkpoint + sha256 (planned; until the copy exists, DEFAULT_BACKBONE in
│                  iltm2.model._backbone is the released checkpoint
│                  /private/groups/ioannidislab/ajaras/tabicl-dev/tabicl-regressor-v2-20260212.ckpt)
├── val_bank_v2/   the validation bank: 2,000 tasks (§3 step 6)
│   ├── shards/<bin>_<k>.pt    task records + their query tables, written atomically (a build resumes)
│   ├── queries.pt             the whole bank's query table (what validation and the reference lines read)
│   ├── mini/                  the 64-task mini-bank (§7.4): tasks.pt, queries.pt
│   ├── build_config.json      seed, R, sizes, prior config, backbone sha256 (a relaunch with another one is refused)
│   └── manifest.json          seed, git hashes, backbone sha256, loss_c, e(q) quantiles, reference lines (bank, mini)
├── val_bank_v2_smoke/ the 14-task smoke bank: code tests only (00e, the smoke run), never for training
├── runs/<run>/
│   ├── best_index.json        # where each head's current best checkpoint lives (stage, global step, val MSE)
│   └── stage{1,2,3}/
│       ├── latest.pt          # heads / fork + optimizers + schedulers + EMAs + best tracker + step (resume)
│       ├── final.pt           # end of the stage: written once, NEVER overwritten; the next stage starts from it
│       ├── best/<pair>_{A,E}.pt   # best head so far (pooled full-bank val MSE, pair's own weighting), reached here
│       ├── trace.jsonl        # per step: LR per pair, batch fingerprint, launch id, global step
│       ├── summary.json       # timings, peak GPU memory, rejection rate, projected stage wall time
│       └── last_{train,mini,val}.json   # the last logged metrics, flat {wandb key: value} (§7.1)
├── wandb/         WANDB_DIR
├── slurm_logs/    every scripts/*.sh logs here (iltm2.paths.LOG_DIR; SLURM does not create it)
└── audits/        split_figures/ (§5), throughput/ (step 4), val_bank_checks/<bank>/ (step 6), export/ (§9.3)
```

- **Checkpoints store only trainable parameters**, plus the backbone path and sha256, never a copy of the backbone. An A head is a few MB; a B checkpoint is larger, since it holds the fork's two blocks + LN with both heads.
- **The bank is the large object**: about 3 GB, mostly the 400 large tasks of up to 60K rows.
- **The training stream is not pre-generated.** Stage 1 alone is about a million tasks (30,000 steps × 32), each with its replicates. The deterministic on-the-fly stream (§1.3) gives the same pairing for free.
- **The legacy root** `/private/groups/ioannidislab/UQ_decoupling_exp/` (`iltm2.paths.LEGACY_ROOT`) holds the first training round's runs (§10). It is read, never written; only `iltm2.inference.LEGACY_BEST` points there.

---

## 3. Order of work

Each step has a gate that must pass before the next one starts. A gate is rerun whenever the code it covers changes.

| # | Build | Gate |
|---|---|---|
| 1 | `_entropy.py`, `_targets.py` | Unit tests of §4.1 pass (`00a`) |
| 2 | `_cut_graph_scm.py`, `_cached_graph.py`, `_postprocess.py`, `_filter.py`, `_dataset.py` (task shapes) | Replicate tests of §4.2 pass (`00b`); rejection rate between 5% and 30% (stage 1 slightly above it: accepted, design Training §1); difficulty-match gate passes (`00f_difficulty_match`) |
| 3 | `_split.py` (regions, thinning, duplicates, subsampled contexts) | Split tests of §4.3 pass (`00c`); split figures of §5 look right, thinned hole and thinned tail included (`03`) |
| 4 | `_dataset.py` + DataLoader, `02_benchmark_prior.py` | Throughput: prior time ≤ 0.8 × step time in every stage, stage 1 with two forwards per group (accepted deviations below) |
| 5 | `_backbone.py` | Backbone tests of §4.4 pass (`00d`), the query-independence test included |
| 6 | `01_make_val_bank.py` | Bank checks below pass (`01b`); $c$ and the reference lines of both weightings stored in the manifest |
| 7 | `_heads.py`, `_loss.py`, `_metrics.py`, `_validate.py` | Head tests of §4.5 pass (`00e`) |
| 8 | `_config.py`, `_base.py`, `frozen.py`, `train_A.sh` | The smoke run of §6 passes (`04_smoke_A.sh`) → launch the A job (§9.1) |
| 9 | `fork.py` + `train_B.sh` | No separate smoke job (§6): each B job checks at its start that the fork equals $h_*$ on the first batch and that one fork forward/backward fits at the largest stage-3 shape → launch the 18 B jobs once the Phase 0 tests are written and their baselines run (§9.1) |

The validation bank (step 6) comes **before** any trainer, because the trainer logs against it from step 0, the reference lines are what make a loss value readable (§8), and the loss's weight floor $c$ is read from its manifest.

### Step 1 — Entropy and targets

The targets are the unbiased replicate estimators of proposal §2. What step 1 checks is the torch implementation: it must reproduce Phase 0's numpy kernels (`experiment0_baseline/src`: `entropy.quantile_entropy`, `forecasts.QuantileForecast.crps`, `truth.replicate_aleatoric` / `replicate_epistemic`) under the single CDF convention of design Architecture §2: piecewise-linear through the 999 quantiles, 0 below the first and 1 above the last, the convention Phase 0 grades with (§4.1).

### Step 2 — Observed-cut prior and filter

Implements design Training §1 (pipeline steps 1–7, 10) and §1.4.

**Rejection-rate gate: 5%–30%.** TabICLv2 rejects about 25% of stage-1 regression tasks (design, Training §1.4). The observed cut removes the latent confounding between X and y and makes the observed X the only frozen-side input of the y-graph, so tasks are expected to be *more* predictable and the rate is expected to fall. Any rate in 5%–30% passes.

- **Above 30%:** the cut's private noise makes tasks noisier than the original prior. Tune the y-graph's noise scale, not the filter, because the filter defines the backbone's prior (design Training §1.4).
- **Below 5%:** check that the predictability test is actually wired in and runs on the $g$-transformed $y^{(0)}$.

Log the rate per stage and the empirical distribution of $\hat t_A$, including its zero-mass fraction (proposal §4.3, "Known caveat").

**Difficulty-match gate.** `00f_difficulty_match.py` (GPU) compares matched cut and unmodified `graph_scm` tasks (stages 1–2; stage 3 rejection rate only) on ExtraTrees held-out $R^2$ (difficulty profile) and on TabICLv2's advantage over ExtraTrees at matched difficulty (in-domain check), with the 5%–30% rejection gate; the pass rule is in design Training §1.4. It also logs both priors' rejection rates, TabICLv2 CRPS, the y-ancestor subgraph sizes and the $\hat t_A$ distribution. On a failure, tune the target graph's private-root noise (`--private-root-scale`, `--private-root-dims`), never the filters, and rerun. The calibrated defaults (`private_root_scale` 0.1, private-root dims LogInt(1, 16), no noise-ancestor check) and their stage-1 results are in design Training §1, "The observed-cut prior"; the final configuration and tolerance are recorded in design Training §1.4.

### Step 3 — Split

Implements design Training §1.1–1.2: the extrapolation tail and the kNN hole, each independently empty or thinned with probability ½ (a thinned region returns $r=\max(1,\operatorname{round}(\rho\,(\text{region size}-\text{its queries})))$ of its non-query rows to the context, $\rho$ log-uniform in [0.05, 0.2]; a region with no non-query row is always empty), duplicates, the six query types, the `iid` prior (TabICLv2's plain random split, $P=N$) and stage 1's subsampled pass (`subsample_context`, with its two relabellings: a duplicate whose source row was dropped takes its source row's type; a thinned region with no returned row left becomes empty). Once `_split.py` passes its tests, run `03_plot_split_tasks.py` (§5) and inspect the figures before moving on.

### Step 4 — Dataset and throughput

`02_benchmark_prior.py` measures tasks per second for each stage's shapes with the planned number of DataLoader workers. A step consumes 32 tasks (`BATCH_SIZE`, design Training §2); in stage 1 every group also carries its subsampled pass, so the GPU runs two forwards per group. The target is a prior time per step of at most 0.8 × the step's train time (the backbone forward over every pass, a lower bound on an A-job step). The workers build tasks while the GPU trains, so a step costs the slower of the two; the remaining 20% absorbs step-to-step variation, with at least 2 steps of items in flight. If it is not met, raise `n_jobs` first, then lower the region bounds in stage 3 (design Training §1.1, "Cost of the enlarged pool").

**Gate outcome**, at the training setting of 32 workers (34 CPU threads, about 17 physical cores, on an A5500 node):

- **Stage 1 passes** (measured ratio 0.56). The measured 1.03 s/step is optimistic, because the short run drains the prefetched items; about 1.6 s/step (ratio about 0.86) is the realistic sustained rate.
- **Stage 2 is mildly prior-bound (accepted):** prior 5.73 s/step against the backbone's 5.17 s/step (ratio 1.11). A 32-task step has only 32 single-task items, and the heavy-tailed task cost (a task can take tens of seconds after several rejections) holds the step back. Stage 2 runs at the prior's pace, about 12.7 h instead of about 11.5 h for an A job. No change to workers, batch or design.
- **Stage 3 misses only the 0.8 margin (accepted):** ratio 0.90, still GPU-bound, measured on 4 steps. The prior does not slow it down; projected 17.2 h for an A job.
- **Projected A job:** about 45 h of training (15.4 + 12.7 + 17.2 h) plus validation, within the job's time limit. B jobs add a fork backward, so they are more GPU-bound.
- No offline pre-generation of the stream.

**Loader.** Task cost is heavy-tailed in N (in stage 3, from under a second to tens of seconds per task). An in-order DataLoader lets one slow task block the delivery of all later ones and idles the other workers. `make_dataloader` therefore delivers items out of order (`in_order=False`, new work only to workers with free capacity), builds each step's groups largest N first, and keeps `prefetch_factor` 4; `iterate_steps` is a per-step reorder buffer that yields each step's groups only when all are in, in group order. Every item is a pure function of (seed, stage, step, group), so the trainer's batches are bit-identical to an in-order loader's, subsampled passes included; the benchmark checks this.

### Step 5 — Backbone wrapper

Implements design Architecture §1 "Forward route", with hooks for $h_*$ (the output of the final LN, `icl_predictor.ln`) and for arm B's fork input (the full residual stream after block $12-k$, all rows). `copy_last_blocks(k)` / `run_blocks` give arm B its trainable copy. The query-independence test (§4.4, check 6) belongs to this step: it is what allows the subsampled pass's context fractions below TabICLv2's 30%.

### Step 6 — Validation bank

`01_make_val_bank.py` builds the 2,000-task bank (design, Training §3) on the carved prior, with thinned regions and duplicates, R_val = 128, and its own seed (`VAL_BANK_SEED` = 20261116), disjoint from the training seeds and from every test:

| Bin | Tasks | Rows $N$ | Context |
|---|---|---|---|
| small | 1,000 | uniform 256–1,024 | exactly 80% |
| medium | 600 | log-uniform 1,024–10,240 | 79–81% (stage 3's rule) |
| large | 400 | log-uniform 10,240–60,000 | 79–81% |

It also builds the 64-task mini-bank (§7.4), records the weight floor $c$ and the quantiles of $e(q_\theta)$ in the manifest, and computes the reference lines of both weightings (§8.1). It stores per query: `query_type`, $n$ bin, $e(q_\theta)$, $\hat t_A$, $\hat t_E$, the score $\hat t_A+\hat t_E$, and the two half-split targets from replicates 1–64 and 65–128; per task, the region sizes, the returned rows and the realized $\rho$ of each region.

**The weight floor $c$.** $c$ is fixed at 0.1 (z-normalized $y$ units; `iltm2.prior._bank.LOSS_C`, `manifest["loss_c"]`), and every trainer reads it from the manifest (`--loss_c` overrides it). The bank median of $e(q_\theta)$ is not used: $e(q_\theta)$ is bimodal over the bank (q25 $5\cdot10^{-4}$, median 0.013, q75 0.25), the median falls in the gap between the deterministic tasks and the noisy ones, and a median-based $c$ would let the deterministic tasks dominate the $p=1$ loss (design, Architecture §3). With $c=0.1$, $w$ runs from 10 down to about 1.7. `01_make_val_bank.sh --loss-c <value>` rewrites only the manifest and the reference lines, never the shards.

**Bank checks, before any training** (`01b_check_val_bank.py`):

- **Integrity.** Counts match the manifest, every target is finite, $\hat t_A\ge0$, $e(q)\ge0$, score $=\hat t_A+\hat t_E$; query-type shares within 1 point of the design's iid 66.5 / hole 14.25 / extrapolation 14.25 / duplicate 5 (each region's empty / thinned split reported); every n bin the bank's shapes reach is populated; $r\le0.04N$ per region, and a task's region queries are `_thinned` exactly when its $r>0$; the mini-bank is exactly the bank's first 64 small tasks.
- **W: the weight floor.** $c$ is the fixed 0.1 unless set with `--loss-c`, and the reference lines were computed with it; the $e(q)$ quantiles, the share of queries at or below $c$ (and how many come from deterministic tasks) and the range of the weights are reported.
- **Reproducibility.** The first tasks of each bin, regenerated from their keys (seed, bin, index), are bit-identical to the stored ones, and their targets, recomputed with the frozen backbone, equal the stored ones. This ties the bank to the code that generates the training stream and to the forward route.
- **Target ordering by query type.** Mean $\hat t_E$ ordered duplicate < iid < thinned < empty, for the hole and for the tail, gated on the pooled bank with task-level bootstrap 95% CIs of the differences (each must exclude 0 in the right direction); hole against extrapolation reported (expected similar), and the orderings per N bin reported. Also the broken-stream red flag of §8.3: the mean $\hat t_A$ of duplicates and of iid queries must agree (CI of the difference covers 0). If the targets themselves do not show the ordering, the split or the prior is broken.
- **The frozen PPD is sane on the cut prior.** TabICLv2's mean CRPS on the bank's iid queries against unmodified `graph_scm` tasks of the same shapes (`iltm2.prior._graph_scm_ref`), per N bin: the ratio cut / graph_scm within [1/1.25, 1.25]. A much worse value means the cut moves the prior further than assumed.

**Gate outcome.** Every check passes except the PPD-sanity ratio of the medium bin, 1.255 against the 1.25 limit (CI [1.09, 1.44]), **accepted** as sampling noise: small 1.24, medium 1.255, large 0.85, pooled 1.14, so the cut is slightly harder than `graph_scm` at small and medium N and easier at large N. What the bank showed (target means by query type, the e(q) proxy's skill, the query distribution over n bins) is in design Training §3, "What the bank showed".

### Steps 7–9

Heads, loss, metrics and validation (step 7), then the two trainers (steps 8–9), each followed by the smoke checks of §6 before launching.

### Accepted deviations

Recorded once here; each was accepted by the user after its gate:

- Stage-1 rejection rate above the 30% gate (step 2; design Training §1).
- Throughput: stage 2 mildly prior-bound (5.73 against 5.17 s/step at 32 workers, about +1 h per A job), stage 3 missing only the 0.8 margin (ratio 0.90, GPU-bound) (step 4).
- Validation bank: PPD-sanity ratio of the medium bin 1.255 against 1.25 (step 6).

---

## 4. Unit tests

Project convention (root `CLAUDE.md`): no pytest. Each subsection below is one plain validation-gate script, `scripts/00{a..e}_test_*.py`, with a matching `.sh` SLURM wrapper (`short` partition, CPU, minutes each). Each script runs its checks, prints one `OK <check>` or `FAIL <check>: <detail>` line per check, and exits non-zero if any check failed. Submit the gates whose code changed before a launch and read the OK/FAIL lines in their logs.

| Script | Section |
|---|---|
| `00a_test_entropy.py` | §4.1 |
| `00b_test_cut_prior.py` | §4.2 |
| `00c_test_split.py` | §4.3 |
| `00d_test_backbone.py` | §4.4 |
| `00e_test_heads.py` | §4.5 |

### 4.1 Entropy and targets

- `e_crps(q)` and `crps(q, y)` in torch agree with Phase 0's numpy kernels (exact `e_CRPS`; CRPS on a fine numerical grid) to about 1e-5 per unit of span, on smooth and rough random quantile sets (tied quantiles included) and random `y`, including `y` outside the quantile range and on knots.
- `t_A` (sorted $O(R\log R)$ formula) and `t_E` reproduce Phase 0's numpy functions on random replicate sets.
- The half-split targets of `_targets.py` use disjoint replicates, and each half is computed with the same functions as the full target.

### 4.2 Observed-cut prior

- **X is bit-identical across replicates** (proposal §4.3, the construction's only correctness guard).
- Task-level objects are cached (`RandomNodeFunction._fit`): two replicate passes differ only through $U_2$ (the `RootPointLaw` record / replay tape).
- The target graph sees only observed values, and its structure is the one of design Training §1.
- The y transform $g$ is fitted once on the pool. All replicates are transformed with the pool's bounds, mean and std, never refitted.
- Same `(seed, step, task_idx)` gives a bit-identical task, in-process and across processes, whatever the number of workers.
- The filters and the task-shape bookkeeping behave as tabicl's.

### 4.3 Split

- **Counts are exact with thinning:** $n_{\text{ctx}}$ ($n_{\text{ctx}}-r_{\text{ext}}-r_{\text{hole}}$ iid rows plus the returned rows), $m_{\text{iid}}$, $m_{\text{hole}}$, $m_{\text{ext}}$, and $P-N$ discarded rows; the row sets partition the pool.
- **Geometry and labels:** the tail is the top-$T$ rows along $u$, the hole the $k$ nearest remaining rows to its centre; the only context rows inside a region are its returned rows, chosen among its non-query rows; a region's queries are `_thinned` exactly when $r>0$.
- **Thinning draws:** half of the regions with non-query rows are thinned, the tail and the hole independently (a quarter of the tasks in each of both empty / tail thinned / hole thinned / both thinned); $\rho$ within its range up to rounding; $r\le0.04N$; sizes known before generation ($P$ from the shape).
- Duplicates copy $x$ only: their source (`dup_src`) is a context row, and their replicates are fresh draws, never the context label.
- **The split and the subsampling never look at y:** calling them with two different y vectors on the same X gives identical indices.
- Determinism per `(seed, step, task_idx)` under the split stream.
- The `iid` prior: no tail, no hole, no duplicates, $P=N$. A stage-3-sized pool splits fast enough.
- **The subsampled pass:** $n'$ of the context rows, uniformly, relative order kept; queries, their $x$ and their replicates unchanged; a duplicate whose source was dropped takes its source row's type; a thinned region with no returned row left is empty; `CutGraphPrior`'s stage-1 groups carry it with one $n'\in[64,n_{\text{ctx}}]$ per group, stages 2–3 do not.

### 4.4 Backbone

- No backbone parameter has `requires_grad`, and the wrapper is not an `nn.Module` (it cannot enter an optimizer). The backbone's state-dict hash and outputs are unchanged after an optimizer step on a head.
- The wrapper's quantiles are `torch.equal` to tabicl's own training-mode forward of a separately loaded, hook-free model; enabling the fork input does not change them; an eval-mode call is refused.
- $h_*$ equals the final LN applied to the last TFicl block's output at the query rows, and decoder($h_*$) is $q$.
- B, for $k=1$ and 2: the fork's output is `allclose` to $h_*$ at step 0 (also with gradient checkpointing), and a training step changes the copy and leaves the backbone and the decoder input untouched.
- A zero-padded micro-batch group: each task's forecast in the group equals its forecast alone, within 1e-5 of the forecast's scale (different batch shapes give float32 round-off of a few units of that scale; float64 agrees to about 1e-14).
- **Query independence:** with the same context, a query's quantiles, $h_*$ and $e(q)$ are the same whether it is asked alone or with the $m-1$ other queries (or a different set of them), at a full context and at a 64-row subsampled one, within the same 1e-5 batch-shape tolerance. A query attends to the context rows only, so the subsampled pass's many queries per context row do not change any forecast.

### 4.5 Heads and loss

- Initialization: last-layer weights are zero and its bias is softplus⁻¹ of the mean target (design, Architecture §1). At step 0 every query of both arms gets the same output. The initialization depends on (seed, head) only: the same for every LR and every $p$, shared by A and B. Pair ids are `A_lr<LR>_p<P>`.
- **Head independence in A jobs.** The 6 pairs (3 head LRs × $p\in\{0,1\}$) share no parameter; stepping one pair (a $p=0$ and a $p=1$ pair tried), with every pair's loss at its own $p$, leaves the other five bit-identical. This catches a shared-optimizer bug.
- **Loss.** With $p=0$ the weight is exactly 1 and the loss is exactly the plain MSE; with $p=1$ it is $(e(q)+c)^{-1}$, detached; $p>0$ without $c$ is refused; group weights recombine group means into the mean over all query rows, both passes of a group included.
- Gradients reach the heads (and in B the fork) and nothing else; at step 0 only the last layer gets one.
- **Metrics in both weightings** agree with the bank's reference lines (a head at the bank constant has the constant baseline's MSE / xMSE and skill 0, at the weighted constant the weighted ones and `wskill` 0); `train_metrics` matches direct formulas, per n bin included; the EMA is bias-corrected and resumable.
- **Validation** on the smoke bank's mini-bank: cached and recomputed inputs give the same predictions for A and B, the guards fire (wrong backbone sha256, altered stored $e(q)$), and `BestTracker` tracks each pair under its own weighting and resumes.

---

## 5. Split figures (after step 3)

`03_plot_split_tasks.py` draws at least ten figures, one per random task, to check by eye that the cut-graph tasks are heteroscedastic and that the split does what it should, thinned regions included. The layout follows TabICLv2's Figure F.11: a 2-D scatter with the third quantity encoded by colour.

**Tasks.**

- Observed-cut tasks with exactly **2 features** and the stage-1 row count (N = 1,024), from a fixed figure seed disjoint from all others. About 60% of tasks are exactly deterministic (no private root reaches y); there is **no selection on noise**, so the figures show the real mix, and each title says whether the task is noisy or deterministic.
- Keep only tasks whose two features each have at least 10 unique values, as TabICLv2 does for its prior plots (Figure F.1 caption). Otherwise categorical columns stack points on top of each other.
- Ten figures are drawn, then more (up to 30) until at least two tasks with a thinned tail and two with a thinned hole are among them.
- The full pipeline runs: pool, $g$, filter, split with thinning, duplicates. For readability only, the context fraction is fixed at 80% ($n_{\text{ctx}}=819$, so the 205 queries do not crowd the panels); $T$, $k$ and the thinning are drawn as in training.
- For the figure only, **every pool row** gets $R_{\text{fig}} = 128$ replicates, so the aleatoric entropy can be shown everywhere, not just at queries.

**One figure per task, four panels side by side, same axes ($x_1$, $x_2$):**

| Panel | Points | Colour |
|---|---|---|
| 1 — Target | All $P$ pool rows | $g(y^{(0)})$, sequential colormap with a colourbar |
| 2 — Aleatoric spread | All $P$ pool rows | $\hat t_A$ from the 128 replicates, min–max scaled per task to [0, 1] (0 = lowest, 1 = highest spread in that task); a deterministic task ($\hat t_A\equiv0$) shows a flat panel saying so |
| 3 — Context | All $P$ pool rows | **blue**: context; **orange** (on top): the rows a thinned region returned to the context; **grey** (behind): every other pool row, i.e. the discarded tail / hole rows and the query rows |
| 4 — Queries | The $m$ query slots | **red** circles iid, squares hole, triangles extrapolation; **purple** circles duplicates |

Panels 3 and 4 both draw the two carved regions:

- **Hole:** a solid circle centred on the hole's centre row, with radius the distance to its farthest row (Euclidean in standardized X).
- **Tail:** the boundary as a dashed line perpendicular to the random direction $u$ at the smallest tail score; the tail lies on the side away from the pool. With $d = 2$ both are exact, not projections. Axes do not have equal scales, so the circle can look like an ellipse.
- Whether a region is thinned is read off panel 3 (orange points).

**What to look for:**

- **Panel 2 is not uniform** in most noisy tasks, i.e. the noise is heteroscedastic. A task whose panel 2 is flat is homoscedastic, which is allowed but should not be the norm. With R = 128 the scaled values carry about 6% noise, so small speckle is expected.
- **Panel 2 does not track the split.** $A$ is a property of $x$ in the task, fixed before the split, so the hole and tail are not systematically high-$A$ regions.
- **Panel 3:** no blue point inside the circle or beyond the dashed line (a region's only context rows are its returned rows); an empty region has no orange point, a thinned one a few orange points spread uniformly over it (5–20% of its non-query rows, at least one); grey points fill the regions and are scattered among the blue elsewhere (the iid queries).
- **Panel 4:** squares inside the circle, triangles beyond the dashed line, iid circles spread over the rest, purple duplicates sitting where context points are.

Save to `audits/split_figures/task{01..}.png`. These figures also go into the thesis as the illustration of the observed-cut prior and the query mixture.

---

## 6. Smoke runs before the long jobs

**One GPU job** (`scripts/04_smoke_A.sh`, 4-hour limit), because every GPU job waits in the queue: runs 1–3 below run in sequence inside it and it stops at the first failure, so an `afterok` dependency can hold the A job on it. It must pass before the A job trains. B has no smoke job of its own (user's decision): the B trainer shares the stage, resume and checkpoint logic tested here, and each B job runs its two B-specific checks when it starts (`iltm2.train.fork`: `preflight`, one fork forward/backward at N = 60,000 with gradient checkpointing, printing peak GPU memory; `check_first_batch`, the fork's output equal to $h_*$ on the first batch within 1e-5). Both print an `OK` line to the job's `.out`, and a failure stops the job before any training. Run 4 is read in the A job itself, not in a separate job.

Every stage of the smoke run takes $c$ from the real bank's manifest (`val_bank_v2` must exist), as a real job does, although stages 1–2 validate on the smoke bank.

1. **Overfit one fixed batch.** One stage-1 micro-batch group (its full pass), chosen by its A targets: among the groups of the first steps of the stream (up to 4), the one with the largest share of queries with $\hat t_A>0$, which must be at least 10%. A task flagged noisy can still have $\hat t_A=0$ at every query (the noise reaches y only in some regions of $x$), and a batch whose $\hat t_A$ is all 0 leaves the A head nothing to fit. Backbone outputs are computed once, and every pair is trained 3,000 full-batch steps at one LR (`--overfit_lr`, 1e-3) with the unweighted loss. Each head's training MSE must fall to at most 5% of its step-0 value, the constant-baseline MSE every head starts from: the heads can memorize their targets from the embeddings, so the plumbing from batch to backbone to heads to loss works. The batch's replicate-noise floor $\overline{\hat t^{\,2}-\hat t^{(a)}\hat t^{(b)}}$ is reported, not gated: deterministic tasks have noise-free $\hat t_E$, so the $\hat t_E$ floor sits far below $\hat t_E$'s spread, and a floor-based bar would ask the E head for a far larger reduction than the A head.
2. **Short stages at real sizes:** stage 1 for 200 steps (both passes), stage 2 for 50, stage 3 for 20 (N up to 60K), each starting from the previous stage's `final.pt`. Record peak GPU memory, seconds per step, `prior_time` vs `train_time`, and the time of the $h_*$ cache and of one full-bank validation (stage 3 caches $h_*$ on the real bank and validates on it; stages 1–2 validate on the small smoke bank, to keep the job short), then extrapolate the wall time of a full job at the design budget. The step-4 benchmark measured the prior and the bare backbone forward; this run adds what it could not: the heads' forward and backward for all 6 pairs, validation and checkpointing. Logs to the `TabUD` wandb project as run `smoke_A_<jobid>` in group `smoke` (tags `smoke`, `arm_A`), which also checks the logging of §7; the real A job logs to the same project in group `A`.
3. **Resume**, inside run 2 at no extra cost: stage 1 is stopped after 100 steps (`--stop_after 100`, which saves `latest.pt` as a killed job would) and relaunched. It must continue at step 100 with every pair's LR on tabicl's schedule, the batches of steps 100 and 101 (both passes) bit-identical to the same steps rebuilt from (seed, stage, step), and the same wandb run (§7.1).
   `scripts/04b_check_smoke.py` checks runs 2–3 at the end of the job:
   - stage files: each stage's `final.pt` unchanged after the later stages (never overwritten), and `best_index.json` listing both heads of the 6 pairs;
   - stage chaining: each stage starts from the previous `final.pt`, global steps continuous;
   - resume, as above;
   - logging (`last_{train,mini,val}.json`): stage 1 runs two passes per group (16 micro-batch forwards per step, stages 2–3: 32); stage 1's training xMSE is logged per n bin, `n_le256` (the subsampled pass) included; the stage-3 validation logs, for every pair and head, `mse` / `xmse` / `skill` and `wmse` / `wxmse` / `wskill` on `all`, every query type and every populated n bin; the mini-bank logs every n bin it populates;
   - resources: peak GPU memory below 22 GiB (A5500, 24 GB), and the projected wall time of a full A job (30,000 / 8,000 / 1,200 steps, ten full validations, $h_*$ cached once per stage launch) within the `--time` of `train_A.sh`.
4. **Bank-versus-stream consistency**, read in the A job after the first few thousand stage-1 steps: no separate job. Compare the training xMSE per n bin (`<pair>/train/xmse_<h>/<n bin>`, §7.3) with the mini-bank xMSE of the same bin (`<pair>/mini/xmse_<h>/<n bin>`, §7.4), per head and pair. The mini-bank's tasks (N 256–1,024 at 80% context) do not have the stage-1 shape (N = 1,024 at 30–90%, plus the subsampled contexts), so the comparison is made within context-size bins, not on the pooled values.
   - **Why both sides are comparable.** Both are noise-free in expectation, so no floor correction is needed. Training losses are computed **before** the optimizer step, so each training value is an evaluation on tasks the head has not yet trained on.
   - **What the test detects.** Each training task is seen once, so there is no stream to overfit. A gap means the bank and the stream are generated differently: a different shape distribution, filter, query mixture or transform.
   - **Gate:** the two curves agree within their noise, bin by bin.
   - **For the results:** their agreement shows that the heads learned a function of the prior that generalizes to fresh tasks, not properties of specific tasks. Report it with the Phase 1 results.

**Gate outcome.** The smoke run passed.

---

## 7. Logging (wandb)

### 7.1 One wandb run per job across all three stages

Each job (the A job of one (prior, seed), or one B configuration) is **one wandb run spanning stages 1, 2 and 3**, so each curve is continuous across the curriculum. Project `TabUD`, entity `poletea10-universitat-polit-cnica-de-catalunya`.

- **Fixed run id.** `--wandb_id` is the run name, and `wandb.init(id=..., resume="allow")` reattaches every launch of every stage to the same run. Run names: `A_<prior>_seed<S>` (e.g. `A_carved_seed0`) and `B_h<H>_f<F>_p<P>_<prior>_seed<S>` (e.g. `B_h3e-4_f1e-5_p1_carved_seed0`), with LRs written with up to 2 significant digits (`3e-4`, `2.5e-4`).
- **Global step.** Logged steps are offset by the earlier stages: stage 1 runs 0–30,000, stage 2 runs 30,000–38,000, stage 3 runs 38,000–39,200. `stage` is logged as a metric, so stage boundaries are visible.
- **Config.** `vars(args)` plus the git state (commit, dirty flag) of the repository and of tabicl, and the backbone's sha256.
- **Groups and tags.** Group = the arm (`A`, `B`; the smoke run: `smoke`). Tags: `arm_<arm>`, `prior_<prior>`, and in B `head_lr_<H>`, `fork_lr_<F>`, `p_<P>`.
- **Key namespacing in A jobs.** One process trains 6 head pairs, so every per-pair metric is prefixed by the pair id `A_lr<LR>_p<P>`, e.g. `A_lr1e-3_p1/val/xmse_E/hole`. A repeated (LR, $p$) in one job gets a version suffix (`A_lr1e-3_p1_v1`, `_v2`), so ids stay unique.
- **B jobs** train one pair, named `B`, so all 18 configurations log the same keys (`B/val/xmse_E/hole`, plus `B/train/grad_norm_fork`, `B/train/lr_fork`, `B/train/fork_dist_from_init`) and are told apart by their config (`head_lrs`, `fork_lr`, `p`, `prior`) and tags: one panel per metric overlays the runs.
- **Job-level keys** (not prefixed): `stage`, `prior_time`, `train_time`, `prior/rejection_rate`, `prior/deterministic_share`, `train/n_passes`, the target means of §7.3, `time/val_s`, `time/mini_s`, and the reference lines `ref/...` (bank) and `ref_mini/...` (mini-bank).
- **`last_*.json`.** Each stage folder also holds `last_train.json` (the last logged training step, written every 100 steps and at the stage end), `last_mini.json` and `last_val.json` (the last mini-bank and full-bank evaluation): flat `{wandb key: value}` with the global step, so a run can be checked without the wandb API.

### 7.2 Script

**One SLURM job per run runs the three stages in sequence** (`scripts/train_A.sh <carved|iid> <seed>`), each stage as its own `python -m iltm2.train.frozen --stage s` process. Three jobs chained with `afterok` would wait in the queue three times, and on this cluster a job held by a dependency gains no queue priority while it waits; one job waits once. `train_A.sh` asks for 3 days of the `gpu` partition (one A5500, 34 CPUs; 7-day partition limit): a full A job is projected at about 45 h of training plus validation (the smoke run prints the measured projection). `train_B.sh` asks for 4 days.

- **Per-stage settings come from the trainer, not the script** (`iltm2.train._config`): steps 30,000 / 8,000 / 1,200, LR divisor 1 / 8 / 40, full-bank validation every 5,000 / 4,000 / 600 stage steps and at each stage end, the subsampled pass on in stage 1 only (`--subsample`), and the stage's shapes from `iltm2.prior._dataset.STAGES`. The sweep's grids are the defaults: `--head_lrs 3e-4 1e-3 3e-3` and `--p 0 1` (A: lists, every LR × $p$ pair in the job; B: one value each, plus `--fork_lr`). `--prior carved|iid`; `--loss_c` defaults to the bank's `manifest["loss_c"]`. The other defaults are design Training §6 (AdamW, weight decay 1e-5, clipping 5.0, `cosine_with_restarts` with 1% warmup to `cosine_lr_end` 1e-7 via tabicl's own `get_scheduler` (B's fork: 1e-8, `--fork_cosine_lr_end`), clipping per head in A (`--clip_per head`) and joint fork + heads in B (`--clip_per joint`), output biases from the first batch (`--init_bias warmup`), R = 16, `--muon False`). Extra flags after the positional arguments are passed to every stage.
- **How a stage starts** (`iltm2.train._base`): if its `final.pt` exists it is done and skipped; if its `latest.pt` exists it resumes with the full optimizer state; otherwise stage s > 1 loads the heads (and the best tracker) from stage s−1's `final.pt` with fresh optimizers and schedulers, tabicl's `--only_load_model` restart (`scripts/train_v2_reg_stage2.sh`); a fresh stage 1 sets each pair's output biases to the first batch's mean targets, weighted for $p=1$ pairs (the best constant under each pair's own loss). So after a timeout or a crash the same `sbatch` command continues where the job stopped.
- **The global step** of a stage starts at the previous stage's final global step (§7.1).
- **wandb key:** kept in a netrc of its own, `~/.wandb_uq/netrc`, created once with `mkdir -p ~/.wandb_uq && NETRC=~/.wandb_uq/netrc wandb login`. The job scripts export `NETRC` to it (wandb reads `$NETRC` before `~/.netrc`), so the shared `~/.netrc`, which holds another account's key on this login, is never read or overwritten; `WANDB_NETRC=<path>` selects another file. Never pass the key as an argument (visible to other users through `scontrol show job`) or write it into a committed file.

```bash
sbatch -J A_carved_seed0 experiment1_uq_heads/scripts/train_A.sh carved 0
STAGES="2 3" sbatch -J A_carved_seed0 experiment1_uq_heads/scripts/train_A.sh carved 0     # a subset of the stages
```

`train_B.sh <head_lr> <fork_lr> <p> <carved|iid> <seed>` follows the same pattern and runs `iltm2.train.fork` (§9.1).

### 7.3 Training metrics (every step, per pair and head)

In stage 1 every training metric pools both passes of the step (full and subsampled contexts). Every metric below except the loss is **unweighted** for every pair, so $p=0$ and $p=1$ pairs compare fairly on them; only `train/loss*` uses each pair's own $p$.

**Raw MSE and xMSE.** Both are logged:

- `<pair>/train/mse_A`, `<pair>/train/mse_E`: the MSE of each head against its full 16-replicate target.
- `<pair>/train/xmse_A`, `<pair>/train/xmse_E`: the **cross-product MSE**. Split each query's 16 replicates into two disjoint halves of 8, build one target from each half, $\hat t^{(a)}$ and $\hat t^{(b)}$, and take

$$
\text{xMSE}=\overline{(\hat A-\hat t_A^{(a)})\,(\hat A-\hat t_A^{(b)})}.
$$

  For $\hat E_{\text{dir}}$, each half computes its own $\frac18\sum_r S(q_\theta,y^{(r)})-\hat t_A^{(\text{half})}$.
- `<pair>/train/xmse_A/<n bin>`, `<pair>/train/xmse_E/<n bin>`: the same per context-size bin of the query's pass (§7.5's bins). The subsampled pass fills the small bins (`n_le256` only through it), so the pooled stage-1 curve mixes contexts from 64 to about 920 rows; the per-bin curves are the ones to compare with the mini-bank (§6, run 4) and across runs.

**Why xMSE.** The raw training MSE is the head's true error plus a large constant: the variance of a 16-replicate target, which for $\hat t_E$ exceeds $E$ itself. The real decrease is a small dip on top of that offset. The two halves are independent draws from the same task law and each is unbiased, so their noise is uncorrelated and the product's expectation is exactly the head's error against the realized task,
$\mathbb E[(\hat A-A_\tau)^2]$, with the replicate-noise term gone. **xMSE is the curve that shows the downward trend.** It costs nothing extra (the loss still uses the full 16-replicate target), but single values can be negative and only its average is meaningful.

**EMA.** `<pair>/train/xmse_A_ema`, `<pair>/train/xmse_E_ema`: an exponential moving average of xMSE with smoothing 0.99 (a window of about 100 steps), bias-corrected at the start of each stage. It removes the per-step fluctuations from task-to-task variation, so the trend can be read between validations.

**Other training metrics:**

- `<pair>/train/loss`, `loss_A`, `loss_E` (weighted with the pair's $p$), `<pair>/train/lr`, `<pair>/train/grad_norm_A`, `grad_norm_E` (before clipping);
- `<pair>/train/mean_A`, `mean_E`; job-level `train/mean_tA`, `train/mean_tE`, `train/mean_eq`, `train/frac_tE_neg` (the share of $\hat t_E<0$), `train/n_passes`;
- `prior_time`, `train_time`, `prior/rejection_rate`, `prior/deterministic_share`;
- B: `B/train/grad_norm_fork`, `B/train/lr_fork`, `B/train/fork_dist_from_init` ($\lVert\theta_{\text{fork}}-\theta_{\text{fork},0}\rVert$).

### 7.4 Mini-bank (every 250 steps)

The first 64 small tasks of the validation bank (N uniform 256–1,024 at 80% context), with their R_val = 128 targets. They are the same tasks at every evaluation, so the comparison between checkpoints is paired and the curve is far smoother than the training one. Logged per pair, head and slice, with slices `all` and the n bins the mini-bank populates (`n_le256`, `n_256-512`, `n_512-1K`): `<pair>/mini/{mse,xmse,skill,wmse,wxmse,wskill,mean}_<h>/<slice>`, and the mini-bank's reference lines `ref_mini/...` (computed with the full bank's constants, which are what a head's initialization outputs). Cost: the heads only in A ($h_*$ cached), the fork and the heads in B (fork input cached).

### 7.5 Validation (ten full-bank validations per run)

At global steps 5,000, 10,000, 15,000, 20,000, 25,000, 30,000 (stage 1), 34,000, 38,000 (stage 2), 38,600, 39,200 (stage 3): every 5,000 / 4,000 / 600 stage steps and at each stage end. Per pair, head and slice, as in design Training §3. Slices: `all`, the six query types (`iid`, `hole`, `hole_thinned`, `extrapolation`, `extrapolation_thinned`, `duplicate`) and six n bins (`n_le256`, `n_256-512`, `n_512-1K`, `n_1K-4K`, `n_4K-16K`, `n_16K-60K`; n is the task's context size $n_{\text{ctx}}$). Slices, metric definitions and key names are shared with the bank's reference lines (`iltm2.train._metrics`): `<pair>/val/<metric>_<head>/<slice>` next to `ref/<line>_<head>/<slice>`.

**Two weightings, logged by every run** whatever it was trained with:

| Key | Unweighted ($p=0$ form) | Weighted ($p=1$ form) |
|---|---|---|
| MSE against the R_val = 128 target | `val/mse_<h>/<s>` | `val/wmse_<h>/<s>` |
| xMSE from the two halves of 64 | `val/xmse_<h>/<s>` | `val/wxmse_<h>/<s>` |
| Skill (§8.1) | `val/skill_<h>/<s>` | `val/wskill_<h>/<s>` |

The weighted metrics multiply each query's term by $w=(e(q_\theta)+c)^{-1}$, with the bank's $c$, and are count-normalized: the mean of $w\,x$ over the slice's queries, the form of the loss. The weighted skill is against the best constant under that weighting, the $w$-weighted mean target.

Also logged:

- `val/mean_<h>/<s>`: the mean prediction (the bias check of §8.2);
- **`val/ratio_A_eq/<s>`**: the per-slice mean of $\hat A/e(q_\theta)$, both clipped below at $\epsilon=10^{-6}$ so that queries where both are 0 do not divide by zero, and beside it the same ratio for the target, `val/ratio_tA_eq/<s>`, the mean of $\hat t_A/e(q_\theta)$. It is an unbiased per-slice reference, since $e(q_\theta)$ is fixed per query;
- the reference lines of §8.1 for both weightings, logged as constants at every validation step, so they draw as flat lines on the same panels: `ref/[w]noise_floor_<h>/<s>`, `ref/[w]const_mse_<h>/<s>`, `ref/[w]const_xmse_<h>/<s>`, `ref/[w]eq_proxy_{mse,xmse,skill}_<h>/<s>`, plus `ref/count/<s>`, `ref/mean_eq/<s>`, `ref/mean_t<h>/<s>`, `ref/ratio_t<h>_eq/<s>`, `ref/frac_tE_neg/<s>`, `ref/[w]const_value_<h>` and `ref/loss_c`;
- "is best" flags `val/is_best_A`, `val/is_best_E`. The best checkpoint of each pair and head is chosen under the pair's **own** weighting: the lowest pooled `val/mse_<h>/all` for $p=0$, `val/wmse_<h>/all` for $p=1$ (full bank only, never the mini-bank). In B a best checkpoint holds the whole model (fork + both heads).

---

## 8. Reading the curves

### 8.1 Reference lines

A raw value like `val/mse_E = 0.0031` cannot be read on its own: it depends on the target's scale and on how noisy the targets are. Three numbers, computed once on the bank with no training, put every curve on a scale. They answer three questions: how low can it go, where did it start, and does it beat what the backbone already offers for free. They are stored in the bank manifest, per head, per slice and per weighting, and drawn as flat lines on the validation panels.

| Line | What it is | What it tells you |
|---|---|---|
| **Noise floor** | The part of the raw validation MSE that is replicate noise: $\overline{\hat t^{\,2}-\hat t^{(a)}\hat t^{(b)}}$ over the bank's queries (exact in expectation, by the same independence argument as xMSE) | How much of the raw MSE no head can ever remove. It belongs on the raw-MSE panel only: on the xMSE panel the noise is already gone and the floor is 0 |
| **Constant baseline** | MSE (and xMSE) of predicting the bank-wide mean target for every query (weighted lines: the $w$-weighted mean target), evaluated per slice | Where every head **starts**: the zero-init + bias-at-mean-target initialization outputs exactly this ($p=0$ pairs at the plain mean, $p=1$ pairs at the weighted mean). Moving below it is the first sign of learning |
| **e(q) proxy** | MSE (and xMSE) of the best predictor that only sees $e(q_\theta)$: 50 equal-count bins of $e(q_\theta)$, each predicting its bin's (weighted) mean target, cross-fitted on two halves of the bank's tasks (each half scored with the bins of the other, so every query is scored out of fold) | What the backbone's total uncertainty gives for free. The thesis claims the heads extract something $e(q_\theta)$ does not carry (proposal §1.1); the backbone check already showed $e(q_\theta)$ missing the divergence in sparse regions (design, Architecture §1). Beating this line is the claim |

The floor is a lower bound, not a reachable value. The irreducible scatter of the realized truth over tasks consistent with the context sits on top of it; it is identical for every head and run, so it needs no oracle to compare runs (design, Architecture §4), and it caps the reachable skill of $\hat E_{\text{dir}}$ well below 1 (the ceiling bracket of design Training §3).

**Skill** turns the xMSE curve into one number per head and slice:

$$
\text{skill}=1-\frac{\text{xMSE}_{\text{head}}}{\text{xMSE}_{\text{const}}},
$$

and `wskill` is the same with the weighted xMSE of the head and of the weighted constant. 0 is the starting point, 1 is perfect, and the e(q) proxy has its own skill to compare against.

**Why per query type.** The split matters most where $e(q_\theta)$ rises for epistemic reasons: holes and extrapolation, empty or thinned. A head can beat the lines on the pooled bank (dominated by iid queries) and still fail exactly there. Per-type lines show which regime is solved.

**Why per n bin.** Pooled values are query-weighted, and the bank's queries sit mostly at large context: 64% of them are in `n_16K-60K` and only 0.2% at $n\le256$. The pooled number therefore says little about small contexts, which is where the toy set and the real-data tests of Phase 0 operate; read the small bins on their own.

### 8.2 Rules of thumb

These thresholds are heuristics, not derived results. What decides the final model is the Phase 0 tests (§9.2).

- **First 2,500 steps (mini-bank curve, §7.4):** both heads below the constant baseline. The first full validation comes at step 5,000; it should confirm this on the whole bank.
- **End of stage 1:** both heads clearly below the e(q) proxy, pooled and on hole and extrapolation queries (empty and thinned).
- **Skill:** around 0.7 or more for $\hat A$ suggests it works. For $\hat E_{\text{dir}}$, 0.3–0.6 is plausible: the pooled ceiling on this bank lies between about 0.44 and 0.64 (design Training §3). $\hat E_{\text{dir}}$ skill below about 0.1 is not a bug: it is the signal that $h_*$ lacks the information, which is what arm B tests (design, Architecture §4).
- **Bias:** mean $\hat E_{\text{dir}}$ against mean $\hat t_E$ on the bank within a few percent.
- **Per query type:** $\hat E_{\text{dir}}$ ordered duplicate < iid < thinned < empty, for the hole and for the tail, following the targets' own ordering (§3, step 6).
- **Ratio $\hat A/e(q_\theta)$:** follows the target's ratio per type. In particular it **drops** on hole and extrapolation queries relative to iid, where $e(q_\theta)$ rises for epistemic reasons but the true $A$ stays at the local noise level.
- **$p=1$ against $p=0$:** $p=1$ is expected to improve the low end (near-deterministic queries, duplicates), which shows on the weighted metrics and on the ratio $\hat A/e(q_\theta)$, mostly at a small cost on the unweighted ones. Each weighting favours the runs trained with it, so validation chooses the learning rate within a $p$, never between the two $p$ (§9.2).
- **Train xMSE vs mini-bank xMSE, per n bin:** agree within noise (§6, run 4).
- **Stages 2–3:** the large-n bins improve while the bins at or below 1K do not degrade. Watch `n_le256` and `n_256-512` at global steps 34,000–39,200 in particular: stages 2–3 have no subsampled pass, so small contexts may be forgotten there.

### 8.3 Red flags

- **Head collapsed to a constant.** Val xMSE stays on the constant baseline, and the standard deviation of the head's output across bank queries is near 0. Causes, in order of likelihood: targets too noisy for the signal (the MSE-optimal response to pure noise is to stay at the mean), learning rate too small to leave the initialization, or $h_*$ lacking the information. Comparing the LRs of one job and arm A against arm B tells these apart.
- **$\hat A$ just reads total spread.** No gain over the e(q) proxy **and** a flat $\hat A/e(q_\theta)$ ratio across query types. A high correlation between $\hat A$ and $e(q_\theta)$ alone is *not* a red flag: on iid queries most of the total is aleatoric, so a correct $\hat A$ should correlate strongly with $e(q_\theta)$ there.
- **Seen-bit not learned.** $\hat E_{\text{dir}}$ on duplicates ≈ on iid queries.
- **Bank and stream disagree.** Train xMSE and mini-bank xMSE of the same n bin differ by more than their noise (§6, run 4).
- **Broken stream.** Rejection rate outside 5%–30% (beyond the accepted stage-1 excess), or a gap in mean $\hat t_A$ between duplicate and iid queries (§3, step 6).
- **Unstable learning rate.** Spikes in the training loss or the validation curves at the top of the head-LR grid (3e-3), or, in B at fork LR 1e-4, a `fork_dist_from_init` that keeps growing with no gain (nothing pulls the fork towards its pretrained weights).
- **Small contexts forgotten.** `n_le256` / `n_256-512` (mini-bank and full bank) degrading over stages 2–3 (§8.2).

---

## 9. Launch plan, selection, export

Following design Training §4–5. Every job runs stages 1 → 2 → 3 in sequence inside one SLURM job, into one wandb run (§7.1, §7.2).

### 9.1 The jobs

| Order | Jobs | Command |
|---|---|---|
| 1 | Smoke run (§6) | `sbatch 04_smoke_A.sh` |
| 2 | **The A job:** `A_carved_seed0`, 6 head pairs `A_lr<LR>_p<P>` (head LR {3e-4, 1e-3, 3e-3} × $p$ {0, 1}) on one backbone stream | `sbatch -J A_carved_seed0 train_A.sh carved 0` |
| 3 | **The 18 B jobs**, deferred until the Phase 0 tests are written and their baselines run: `B_h<H>_f<F>_p<P>_carved_seed0` (head LR {3e-4, 1e-3, 3e-3} × fork LR {1e-5, 3e-5, 1e-4} × $p$ {0, 1}), paired with the A job's task stream | loop below |
| 4 | Selection and evaluation (§9.2), then the winners' extra jobs: seeds 1–2 and the `iid` prior | `train_A.sh <prior> <seed> --head_lrs <LR> --p <P>`, `train_B.sh <H> <F> <P> <prior> <seed>` |

The smoke run goes first; the A job can be submitted at the same time, held on it by a dependency, so it starts as soon as the smoke run passes and never if it fails:

```bash
SMOKE=$(sbatch --parsable experiment1_uq_heads/scripts/04_smoke_A.sh)
sbatch -J A_carved_seed0 --dependency=afterok:$SMOKE --kill-on-invalid-dep=yes \
  experiment1_uq_heads/scripts/train_A.sh carved 0
```

In the A job's first hours, run the bank-versus-stream check of §6 run 4 per n bin.

The B jobs wait until the Phase 0 tests are written and their baselines run: nothing before the selection depends on them, and Phase 0 has priority. Each needs one GPU job of about 55–60 h, so the round is about 1,000 GPU-hours:

```bash
for H in 3e-4 1e-3 3e-3; do for F in 1e-5 3e-5 1e-4; do for P in 0 1; do
  sbatch -J B_h${H}_f${F}_p${P}_carved_seed0 experiment1_uq_heads/scripts/train_B.sh $H $F $P carved 0
done; done; done
```

In their first hours, the same per-bin check, plus `B/train/fork_dist_from_init` at the higher fork LRs (§8.3).

### 9.2 Selection and evaluation

Following design Training §4.

1. **Learning rates, from validation.** Within each (arm, $p$, prior) cell, the LR with the lowest pooled validation MSE under that cell's own weighting (`val/mse_<h>/all` for $p=0$, `val/wmse_<h>/all` for $p=1$), at each head's best checkpoint (§7.5). In A per head; in B the configuration as a whole (the sum of the two heads' MSE, each relative to A's at the same $p$), since its heads share the fork.
2. **Final model, from the tests.** The four carved winners (A and B, $p=0$ and $p=1$) run on T2–T5 of Phase 0, through the same forward route, and the best is chosen. Then the final model runs on T1 and T6.
3. **Ablation.** The final model against three counterparts, each differing in one factor: the other $p$, the other arm, and the `iid` prior. All on T2–T5. The `iid` counterpart and seeds 1–2 (error bars) are trained only for the winning configurations; the exact job list follows from the selection. Learning rates are not ablated.

**Selected checkpoints.** TBD (filled after the selection). They go into `iltm2.inference.BEST`; `ILTMv2Regressor.from_best("A" | "B")` then loads them, and Phase 0's iLTMv2 rows use them (`experiment0_baseline/configs/methods.yaml`).

### 9.3 Exporting the runs for analysis

`scripts/05_export_runs.sh` (CPU) pulls the runs' key metrics from the wandb API and the run folders into small CSVs plus a data dictionary (`README.md`) in `<ROOT>/audits/export/`. Runs: every `A_<prior>_seed<S>` and `B_*` run (finished or running); not the smoke run. Every CSV is long format, one measurement per row, keyed by `run, pair, arm, head_lr, fork_lr, p, prior, seed`:

| File | One row per | Metrics |
|---|---|---|
| `val.csv` | validation step × pair × head × slice | mse, xmse, skill, wmse, wxmse, wskill, mean prediction; head A also Â/e(q) and t_A/e(q) |
| `train.csv` | 250-step window × pair | xmse_A_ema, xmse_E_ema; mean / max gradient norm per head (B: also the fork's, and its distance from init); mini-bank xMSE and weighted xMSE |
| `reference.csv` | head × slice | count, mean target, mean e(q), target / e(q) ratio, c, and for both weightings the noise floor, constant value, constant-baseline MSE / xMSE, e(q)-proxy MSE / xMSE / skill |
| `runs.csv` | run × pair | status, finished stages, the weighting of its best checkpoints, best val MSE per head (stage, global step), whether selected (§9.2), per-stage wall time, step time, validation and cache time, peak GPU memory, rejection rate |

Slices: all, the six query types and the six context-size bins.

---

## 10. Legacy checkpoints (first training round)

Checkpoints of the first training round are kept on the group disk. Their training code and full protocol are in the archived repository (`/private/groups/ioannidislab/UQ_project_v1`, commit `c1ed503d0a269051724ee61485862b96472eee45`). Paths are under `/private/groups/ioannidislab/UQ_decoupling_exp/`.

| Arm (old name) | Head | Checkpoint | Head LR / fork LR, seed | Pooled val MSE | Val skill (all / iid / hole / extrapolation / duplicate) |
|---|---|---|---|---|---|
| A1 (now A) | $\hat A$ | `runs/A_seed1/stage3/best/A1_lr1e-3_A.pt` | 1e-3, seed 1 | 0.004041 | 0.891 / 0.910 / 0.872 / 0.825 / 0.916 |
| A1 (now A) | $\hat E_{\text{dir}}$ | `runs/A_seed0/stage3/best/A1_lr1e-3_E.pt` | 1e-3, seed 0 | 0.02117 | 0.420 / 0.370 / 0.447 / 0.474 / 0.496 |
| A2 (removed) | $\hat A$ | `runs/A_seed1/stage3/best/A2_lr1e-3_A.pt` | 1e-3, seed 1 | 0.003812 | 0.898 / 0.917 / 0.880 / 0.829 / 0.922 |
| A2 (removed) | $\hat E_{\text{dir}}$ | `runs/A_seed1/stage3/best/A2_lr1e-3_E.pt` | 1e-3, seed 1 | 0.02108 | 0.423 / 0.375 / 0.443 / 0.476 / 0.526 |
| B | $\hat A$ | `runs/B_h3e-4_f3e-5_seed0/stage3/best/B_A.pt` | 3e-4 / 3e-5, seed 0 | 0.003678 | 0.902 / 0.922 / 0.886 / 0.829 / 0.924 |
| B | $\hat E_{\text{dir}}$ | `runs/B_h3e-4_f3e-5_seed0/stage3/best/B_E.pt` | 3e-4 / 3e-5, seed 0 | 0.02058 | 0.436 / 0.384 / 0.472 / 0.487 / 0.513 |

All are the end of stage 3 (global step 55,000). Skill and MSE are on the first validation bank and are not comparable with numbers on the current one.

**Shared training configuration.** 50,000 / 4,000 / 1,000 steps; 32 tasks per step; AdamW (weight decay 1e-5, clipping 5.0); cosine schedule restarted per stage, LR divided by 1 / 8 / 40; $p=0$; $R=16$.

**How they differ from the current training.**

- **Data.** Holes and tails were always fully emptied (no thinned regions). One context per task: 30–90% of 1,024 rows in stage 1, so no context below about 307 rows.
- **Batch.** One forward per task (no subsampled pass).
- **Loss.** Unweighted only.
- **Arms.** A1 is the current arm A. A2 (softmax mix of 12 residual taps) has no counterpart in the current code and needs the archived repository to load.
- **Validation.** First bank: small tasks of 1,024 rows at 30–90% context, no thinned regions, unweighted metrics.

**Loading them.** `iltm2.inference.LEGACY_BEST` holds the A1 (as arm A) and B checkpoints of the table, under `iltm2.paths.LEGACY_ROOT`; `ILTMv2Regressor.from_best("A" | "B", legacy=True)` loads them, and an A2 file is refused. Their wandb runs are in the project `iLTMv2`.
