# Phase 1 Process Outline

This document gives a thorough understanding of the whole model training phase (Phase 1), from the architecture and loss to the actual training procedure. The actual step-by-step organisation and code structure is explained in the `phase1_protocol.md` file found in the same sub-folder.

# Architecture and loss

## 1. Heads

Two scalar heads are added on top of the frozen TabICLv2 regressor. $e(q_\theta)$ is still computed in closed form from the quantile head (§2.3) but is **not** part of either head's output.

$$
\hat A(x_*,D)=\mathrm{softplus}\big(a_\theta(h_*)\big),\qquad
\hat E_{\text{dir}}(x_*,D)=\mathrm{softplus}\big(b_\theta(h_*)\big)
$$

- **$h_*$** is the representation the quantile decoder reads: the post-final-LayerNorm output of the ICL transformer (TFicl) at the query row, a 512-d vector (`_model/learning.py`: `src = self.ln(self.tf_icl(R, train_size))`, then `self.decoder(src)`). In arm B the heads read the output of a trainable fork instead, a vector of the same shape (§4).
- **Trunk.** One trunk per head, mirroring the quantile decoder (a 2-layer MLP, 512 → 1024 → 999; TabICLv2, App. A.4):

  `Linear(512 → 1024) → GELU → Linear(1024 → 1) → softplus`

  Separate trunks keep the noisier $\hat t_E$ gradients from degrading $\hat A$. Zero-init the last layer's weights and set its bias to $\mathrm{softplus}^{-1}$ of the mean target on a warm-up batch: the first batch of stage 1 (`--init_bias warmup`, the default; `--init_bias bank` would use the bank-wide constant instead). For a head pair trained with $p=1$ (§3) the mean is the $w$-weighted mean target, the best constant under its own loss, so every pair starts at its constant baseline.
- **Why softplus.** It behaves like $e^z$ for $z\ll0$, which resolves values spanning orders of magnitude (the backbone check shows $E$ from $10^{-4}$ to $10^{0}$), and like $z$ for large $z$, so squared-error gradients do not explode. The non-negativity is correctly specified: $\hat t_A\ge0$, and $\mathbb E[\hat t_E\mid x_*,D]=\mathbb E_{\tau\mid D}[d_S(q_\theta,p_\tau)]\ge0$. The price is a weak gradient on small outputs, which the loss weight of §3 compensates.
- **No coupling to $e(q_\theta)$, and no gate.** Building a head out of $e(q_\theta)$ (for example $\hat E=e(q_\theta)-\hat A$, or a gate on $e(q_\theta)$) was rejected, because it would: (1) contaminate our estimations with the amortization gap, and (2) couple the divergence and aleatoric entropy to a value that sometimes moves in the opposite direction (the backbone check shows that $e(q)$ does not follow the actual divergence spread in sparse context intervals).
- **Units.** Both heads live in TabICLv2's internal z-normalized $y$ units (the units of the pinball loss, `_finetune/regressor.py`). Multiplying by the context std $s_y$ gives raw units exactly, since CRPS entropy and divergence are positively 1-homogeneous under $y\mapsto ay+b$.
- **Forward route.** Training, validation and every Phase 0 evaluation of iLTMv2 run **one raw forward pass** of the TabICLv2 model, the same computation TabICLv2 pretraining runs (`train/_run.py: run_micro_batch`, `self.model(X, y_train, d)`):
  - The backbone runs under `torch.no_grad()` so we can get $h_*$ and not save any gradient info in that forward pass.
  - Do **not** use `model.eval()` + `model(...)`. In eval mode `ICLearning.forward` switches to `_inference_forward`, a different code path. Dropout is 0, so the training-mode path is deterministic. Unit test at startup: the hooked forward's quantiles are `torch.equal` to tabicl's own training-mode forward.
  - **No `TabICLRegressor` use for the tests.** The sklearn wrapper adds its own X preprocessing, context-based y scaling and an ensemble of `n_estimators` variants whose predictions are averaged. That averaged forecast is a different $q_\theta$. $\hat E_{\text{dir}}$ is trained to the divergence of **one specific forecast** (proposal §4.5, "checkpoint-specific"), so it is only valid for the forward that produced its targets. An ensemble would change both $e(q_\theta)$ and the excess the head was trained to predict.
  - **Consequence for Phase 0.** The iLTMv2 adapter (`phase0_protocol.md`, iLTMv2 adapter) uses this same route, with context-only y scaling at deployment (residual caveat, Training §1). Any TabICLv2 row compared against iLTMv2's PPD also uses this route, so the forecasts being compared are identical.

### Why $h_*$ alone: the link to DEUP's stationarizing features

DEUP trains an error predictor for a main predictor. In interactive settings the main predictor is retrained as data arrive, so the error at a fixed $x$ changes over time and is non-stationary as a function of $x$ alone (Lahlou et al., §3.2).

DEUP's fix is to treat the error as a function of $(x,\text{dataset})$, and to feed **stationarizing features** that summarise the dataset and the predictor's state:

- a density estimate $\hat q(x\mid\text{data})$: how much data lies near $x$;
- a model-variance estimate $\hat V$: how much plausible fits disagree at $x$;
- a bit $s$: whether $x$ is in the training set;
- optionally $x$ itself.

These are defined in the paragraph introducing $\varphi_{z^N}(x)$ in §3 of the Lahlou et al. paper. The ablations show that $x$ alone is insufficient and that these features drive reliability (App. A.4, Fig. 8; App. C, Table 6).

In a PFN this conditioning on the dataset is built in. $h_*$ is a function of $(x_*,D)$, since the query row attends to every context row in each TFicl block (`_model/encoders.py`, `train_size` masking). It is therefore a **learned proxy for all three features**:

| DEUP feature | What $h_*$ can carry |
|---|---|
| Density | How many relevant context rows the query attended to |
| Seen bit | A near-exact match between $x_*$ and a context row |
| Model variance | The spread of the forecast itself: $q_\theta$ is a 2-layer function of $h_*$, so $e(q_\theta)$ is a deterministic function of it |

So the heads read $h_*$ only. Whether $h_*$ actually keeps the information the split needs is tested in §4.

> **Ablation (not run unless a head fails to beat the e(q) proxy).** Append one scalar, the standardized $\log e(q_\theta)$, to the trunk input (512 → 513). It adds no information, since $e(q_\theta)$ is a function of $h_*$, but it hands the trunk DEUP's variance feature ready-made. Use stop-gradient on it ($\mathrm{sg}[\cdot]$, i.e. `.detach()`: used in the forward pass, treated as a constant in backpropagation).

## 2. Targets: raw squared error and replicates

Given $R$ i.i.d. replicates $y^{(1)},\dots,y^{(R)}$ of the query target from the observed-cut prior (§4.3):

$$
\hat t_A=\frac{1}{R(R-1)}\sum_{r<s}\lvert y^{(r)}-y^{(s)}\rvert,\qquad
\hat t_E=\frac1R\sum_r S\big(\mathrm{sg}[q_\theta],y^{(r)}\big)-\hat t_A .
$$

- **Raw squared error only.** Never clip $\hat t_E$ at zero, even though single targets can be negative. All of these break the conditional-mean argument of §2.4.
- **$R=16$ in training.** Each extra replicate costs one more pass of the target graph over the query rows. More replicates do not move the optimum; they cut target variance, mostly for $\hat t_E$, which is a difference of two noisy terms. Lower $R$ only if target generation becomes the bottleneck. The validation bank uses $R_{\text{val}}=128$ (Training §3).
- **The forecast the target belongs to.** $\hat t_E$ is the excess of the forecast the model actually made for that context. When the same task is seen with a subsampled context (Training §1.5), the replicates and $\hat t_A$ are unchanged, and $\hat t_E$ is recomputed from the new forecast.
- **One CDF convention throughout.** Compute $e(q_\theta)$, $S(q_\theta,y^{(r)})$ and every Phase 0 truth (`phase0_protocol.md`, ground truth) from the same CDF built from the quantiles (tails or no tails). Otherwise $\hat r$ picks up a constant offset.

## 3. Loss

$$
\mathcal L=\overline{\,w\,\big(\hat A-\hat t_A\big)^2\,}+\lambda_E\,\overline{\,w\,\big(\hat E_{\text{dir}}-\hat t_E\big)^2\,},
\qquad w=\big(\mathrm{sg}[e(q_\theta)]+c\big)^{-p},\qquad p\in\{0,1\},
$$

with the bars denoting averages over query rows (in stage 1, over the query rows of both passes of every task, Training §1.5). The pinball loss is absent because the PPD is frozen. Since the heads share no parameters, $\lambda_E$ only rescales the optimizer step of the $\hat E$ head (default 1). $p$ is set per head pair: each configuration is trained with one value, and both values are swept (Training §4).

**Why the weight.**

- *The problem, in the targets.* Target noise scales with the quantity itself. For Gaussian noise of scale $\sigma$ and $R=2$, $\hat t_A$ has mean $\sigma/\sqrt\pi$ and standard deviation $\approx0.43\sigma$. Unweighted squared error is therefore dominated by high-uncertainty queries, while relative errors on small values (dense regions, small $E$) are nearly free.
- *The problem, in the heads.* With softplus, a small output responds to its input in proportion to the output itself, so the squared-error gradient is about $2(\hat A-t)\,\hat A$. A small query ($\hat A=0.002$, $t=0$) gets a gradient of about $8\cdot10^{-6}$; a large one ($\hat A=0.5$, $t=0.3$) about $0.16$. With $p=0$ the optimiser almost never corrects small values, and a near-deterministic query can keep an $\hat A$ of the size of $e(q_\theta)$ at a squared error of order $10^{-6}$ that the MSE never sees.
- *Why it matters.* The epistemic fraction $\lambda=\hat E_{\text{dir}}/(\hat A+\hat E_{\text{dir}})$ is a ratio, the toy figures draw $E$ on a log axis, the toy rung 0 asks for $\hat A\approx0$, and the T2 heatmap compares means cell by cell down to the noise-free row. All of them look at the scale the unweighted MSE ignores.
- *Why $p=1$.* On the low end the output is about $e(q_\theta)$, so a weight of $1/e(q_\theta)$ gives back the factor of the output's scale by which softplus weakens the gradient. $p=1$ up-weights low-spread queries in proportion to $1/e(q_\theta)$; $p=0$ counts every query equally in absolute units, so large-value queries dominate. ($p=2$ would be inverse-variance weighting, the efficient weighted-least-squares choice for target noise $\propto$ scale; it is not trained.)
- *Why the weight is safe.* For a fixed input, $\mathbb E[w(a-\hat t)^2\mid x_*,D]=w\cdot\mathbb E[(a-\hat t)^2\mid x_*,D]$. Any weight that depends **only on $(x_*,D)$** leaves the minimiser at the conditional mean; it only changes how a finite head trades off errors *across* inputs. A weight depending on $y$ (e.g. $1/\hat t^2$) would bias the heads.
- *Why $e(q_\theta)$.* It is a free, frozen proxy for each query's scale computable from $(x_*,D)$. It does not need to be accurate: a poor proxy only reduces the efficiency gain and cannot bias the result.
- *The floor $c$.* Without it, a query with $e(q_\theta)$ near 0 would get a near-infinite weight; the largest weight is $1/c$. **$c$ is fixed at 0.1** (z-normalized $y$ units) and stored in the validation bank's manifest (`loss_c`). Queries with $e(q_\theta)$ well below 0.1 share roughly the same weight, at most about 6 times that of a typical noisy query, and the rest are down-weighted smoothly: $w$ runs from 10 down to about 1.7. The bank median of $e(q_\theta)$ is not used as the floor: $e(q_\theta)$ over the bank is bimodal (q25 $=4.7\cdot10^{-4}$, median $0.013$, q75 $0.25$), the median falls in the gap between the deterministic tasks (62% of the bank's tasks) and the noisy ones, and 88% of the queries below it come from deterministic tasks. With $c=0.013$ the largest weight would be 79, about 35 times a typical noisy query's, so the deterministic tasks would dominate the $p=1$ loss.
- *Which $p$ wins is not a validation question.* Each weighting of the validation metrics favours the run trained with it, so the choice between $p=0$ and $p=1$ is made with the Phase 0 tests (Training §4). $p=1$ is expected to win, since it improves small-value estimates. The weight matters most for $\hat E_{\text{dir}}$, whose range is widest.

> **Epistemic fraction.** Report $\lambda=\hat E_{\text{dir}}/(\hat A+\hat E_{\text{dir}})\in[0,1]$, the share of the forecast's expected score that is reducible (since $\hat A+\hat E_{\text{dir}}$ estimates the expected score of $q_\theta$, §2.2). It is scale-free and comparable across datasets, which resolves the note of §4.6.

## 4. Connecting the heads to the backbone

### The caveat

Pinball pretraining only asks $h_*$ to encode the posterior *predictive* $q_\theta$, a mixture of task laws. The decomposition needs the posterior *over tasks*: $A=\mathbb E_{\tau\mid D}[e(p_\tau)]$ is not a function of the mixture alone (Bergna et al., Prop. 1).

The PFN amortizes the posterior and never had to represent it explicitly. So nothing forced $h_*$ to keep what tells "a mix of sharp laws" apart from "one wide law". It may have kept it (TabICLv2's quantile bands do track heteroscedastic noise, TabICLv2 App. I), but this is not guaranteed.

### The two arms

Two arms give the heads different access to the backbone's information. **A** reads $h_*$ only, at the cost of the heads alone. **B** gives the heads a trainable copy of the last blocks, which can re-attend to the context under the UQ objective, at the cost of one job per configuration and two extra blocks at every inference. Comparing them turns "is $h_*$ enough?" into a measurement: if B adds little over A, the decomposition information is readable from $h_*$, which is itself a result about what PFNs keep when they amortize the posterior.

**Comparing runs needs no ground truth.** On held-out cut-graph tasks (the validation bank, Training §3),

$$
\mathbb E\big[(\hat A-\hat t_A)^2\big]=\mathbb E\big[(\hat A-A)^2\big]+\text{target-noise variance}.
$$

The noise term is identical across arms, learning rates and seeds (under one weighting, Training §3), so a lower validation loss *is* a lower true error. The Phase 0 tests (T2 heatmap, T3–T5) then grade the selected models on what the validation loss cannot see.

| Term | Meaning |
|---|---|
| Frozen | Weights not updated during training |
| Residual stream | Each transformer block adds its update to a running vector per row; $r_\ell$ is that vector after block $\ell$, and $h_*$ is the normalized $r_{12}$ at the query row |
| LayerNorm (LN) | Rescales a vector to zero mean and unit variance across its components, then applies learned per-component gains. TabICLv2 is pre-norm, so the stream's magnitude grows with depth and LN puts different depths on one scale |
| Fork | A branch split off the main computation path: a copy that continues separately |

```
R ─► TFicl blocks 1 … 12 (frozen) ─► LN (frozen) ─► h_* ─► quantile decoder (frozen) ─► q_θ ─► e(q_θ)
          │                                          │
          │                                          └─[A]──────────────────────────────► trunk_A ─► Â
          │                                                                               └► trunk_E ─► Ê_dir
          └─ r_{12−k} ──► [B: trainable copy of blocks 13−k … 12 + LN] ─► h_B ──────────► trunk_A ─► Â
                                                                                          └► trunk_E ─► Ê_dir
```

#### Arm A — MLP trunks on $h_*$

- **Procedure.** The trunks of §1 read $h_*$. The backbone is frozen and run under `torch.no_grad()`, which saves most activation memory. The PPD is bit-identical to TabICLv2.
- **What it tests.** Whether the split is present in $h_*$, possibly entangled nonlinearly.
- **Cost.** Nothing upstream of the heads trains, so one backbone pass feeds every configuration of a job (Training §5), and the bank's $h_*$ can be cached once per job, so a validation costs only the head MLPs.

#### Arm B — trainable fork of the last $k$ blocks

- **Procedure.**
  - Copy the last $k=2$ TFicl blocks and the final LN, initialized with the pretrained weights.
  - The copy receives the same input as the originals, the residual stream after block $12-k$, for **all rows**. Context rows must be processed too, because in every block the query attends to them.
  - The copy's output feeds **only** the heads, and its weights are trained with the UQ loss. The original blocks stay frozen and keep feeding the decoder.
- **Why it beats unfreezing shared layers.** Bergna et al. unfreeze the final two shared layers, which changes the PPD (DecoupledPFN, App. B.2). Here:
  - The quantile path is untouched, so the PPD stays bit-identical by construction.
  - At step 0 the copy outputs exactly $h_*$, and the trunks get the same initialization as arm A's for the same seed, so B starts where A starts and can only add.
  - $\hat E_{\text{dir}}$'s target stays fixed, so no $\lambda_E=0$ warm-up is needed.
- **Why it can recover posterior information.** The copied blocks re-attend to the context under a new objective. They can learn to compute what the original last layers had no incentive to compute, such as how many context rows are relevant to $x_*$ and how much their targets disagree.
- **Limitation.** It can only recompute from what survives in the stream at block $12-k$; if earlier layers dropped the information, $k$ must grow.
- **Settings.** Head LR in {3e-4, 1e-3, 3e-3} and fork LR in {1e-5, 3e-5, 1e-4} (Training §4). Bergna et al.'s split (about $10^{-5}$ for the copy versus about $3\times10^{-4}$ for the heads, App. B.3) is the low corner of that grid. There is no L2 penalty pulling the copy toward its pretrained weights; its distance from them (`fork_dist_from_init`) is logged, and an L2 penalty is the remedy if the fork overfits the cut-graph prior.
- **Cost.** About $k/12$ extra TFicl compute, plus backpropagation through $k$ blocks. Each configuration needs its own job (the fork changes the forward pass), and the fork's input cannot be cached on the full bank.

**Both arms are trained** on the same seeds and task streams, so every comparison between them is paired. The learning rates are chosen from validation within each arm; the arm and $p$ of the final model are chosen with the Phase 0 tests (Training §4).

# Training procedure and job layout

## 0. Background: how TabICLv2 is trained, and what we keep

TabICLv2 is meta-trained. One training example is a whole synthetic dataset, not a row.

- **Each step.** Each optimizer step draws 64 fresh datasets from the `graph_scm` prior and splits each at a random `train_size`. The first rows are the **context**, given with their labels. The rest are the **queries**, whose labels are hidden and used only in the pinball loss (`train/_run.py: run_micro_batch`; `prior/_dataset.py: sample_train_size`).
- **Curriculum.** Training runs in three stages with a fixed number of steps (TabICLv2 §4.1, App. B.1):

  | Stage | Steps | Rows per dataset | Context fraction | Max LR |
  |---|---|---|---|---|
  | 1 | 500K | 1,024 | 30–90% | 8e-4 |
  | 2 | 40K | 400–10,240 (log-uniform) | ≈80% | 1e-4 |
  | 3 | 10K | 400–60,000 (log-uniform) | ≈80% | 2e-5 |

  The cosine schedule restarts per stage, and the number of features is drawn at random per dataset (up to 100).
- **No validation split and no early stopping.** The loop in `_run.py` runs to `max_steps` and saves checkpoints. Every batch is new data, so the training loss is already a held-out loss. The only thing that can be overfitted is the prior itself, which is why the authors monitor real data instead (the 60 TabPFNv2 development datasets, Fig. 10 caption) and test on held-out benchmarks (TALENT, App. K.1).

We keep the same logic: an infinite stream of fresh tasks, a fixed step budget, no early stopping, and a fixed validation bank used to *compare* runs and checkpoints.

## 1. Task construction (one training task)

All tasks come from the **observed-cut prior** (proposal §4.3; code in `src/iltm2/prior/`, mainly `_cut_graph_scm.py`), described below. Because y is produced only after X is fixed, **we build the whole split on X first and draw y last.**

### The observed-cut prior: how it differs from `graph_scm`

**The original prior.** TabICLv2 was pretrained only on `graph_scm`. Per dataset it samples a random DAG (Cauchy edge model, 2–32 nodes), puts random-point vectors at the root nodes and pushes them through random node functions (MLPs, trees, GPs, discretizations, …). Each node is a vector of LogInt(1, 32) latent dims plus the dims its converters read; features and the target y are read from random nodes by converters (numerical, with optional warping, or categorical). There is no node noise (`graph_noise False`): all aleatoric spread is omitted-variable noise, the dimensions upstream of y that the features do not pin down. A dataset is then post-processed (outlier clamp, standardization) and filtered (graph check, constant columns, ExtraTrees predictability test).

**Why it cannot be used as is.** X and y share upstream randomness U, so an exact replicate of y at a fixed x needs U from the abduction posterior $p(U\mid\Phi(U)=x)$, which these dense nonlinear graphs make intractable (proposal §4.3, CLAUDE.md "abduction rule").

**What differs.** Each difference is the smallest one that makes replicates exact or brings the cut's difficulty back to `graph_scm`'s.

| # | Difference | Why |
|---|---|---|
| 1 | **The cut.** X comes from an X-graph with `graph_scm`'s mechanism but no y node. y comes from a second, `graph_scm`-style **target graph** (same DAG sampler and size range, same node functions): $y=\Psi(X_{\text{obs}},U_2)$, $U_1\perp U_2$. | Nothing has to be inferred from X: a replicate is "hold X, redraw $U_2$, rerun Ψ". |
| 2 | **X enters as a parent value.** The target graph's X-input nodes are a random subset of the X-graph's feature-hosting nodes, placed on random DAG positions whose incoming edges are cut. The observed post-converter values of a host's columns are the node's parent value; the node is an ordinary non-root `graph_scm` node (`RandomMultiFunction` of the block, LogInt(1, 32) latent dims, std → importance weights → L2 → node importance), so it is a deterministic function of X. | Randomness enters only at roots, as in `graph_scm`. Adding fresh noise dims to the X-input nodes instead makes cut tasks far noisier (median ExtraTrees $R^2$ 0.44 vs 0.92, 52% rejections). |
| 3 | **Observed values only; the `*_id` write-back is taken out.** X-input values are recomputed from the observed, outlier-clamped columns. For the categorical modes `neighbor_id` and `softmax_id`, whose write-back is the continuous latent (not determined by the observed category), the category's center / embedding is used instead. | y may depend on $U_1$ only through what the model sees; otherwise replicates are not i.i.d. draws from $p(y\mid x)$. |
| 4 | **No noise-ancestor check.** Only the proposal's graph check remains (an X input is an ancestor of y). When no private root is an ancestor of y, the task is exactly deterministic ($A=0$ at every x). | Requiring a private root to reach y forced irreducible noise into every task and removed `graph_scm`'s near-deterministic mass (median ExtraTrees $R^2$ 0.26–0.56 with the check, 0.92–0.94 without). About 60% of accepted stage-1 tasks are exactly deterministic. |
| 5 | **Lower private-root noise scale:** private roots' outputs × 0.1 (`private_root_scale`, i.e. a 10× smaller node importance). | Private roots are $U_2$, the only randomness; X never sees them, so their whole effect on y is irreducible. The scale lowers it. |
| 6 | **Lower private-root dimension:** LogInt(1, 16) dims instead of LogInt(1, 32) (`private_root_dims_range`). | Tree-type node functions split on random points of the data, so they are blind to an input's scale; fewer noise dims also weakens private roots there. With 5 alone the rejection rate stays at 35%; with 5 + 6 it falls to 25%. |

Two implementation details keep the replicates exact: replicates are drawn as **R separate passes over the same rows**, so the $U_2$-independent part of y is bit-identical across replicates (a single pass over tiled rows lets float32 round-off, amplified by tree splits and argmins, put spurious spread of up to 0.2 std(y) on deterministic tasks); and tasks whose target-graph inputs, $y^{(0)}$ or replicates are non-finite are rejected.

**How the cut is checked against `graph_scm`: the difficulty-match gate** (`scripts/00f_difficulty_match.py`, §1.4). It generates matched pairs of cut and unmodified `graph_scm` tasks (same N, $n_{\text{ctx}}$, d; plain i.i.d. split; `graph_scm` with TabICLv2's pretraining config) and requires, per stage: (1) a cut rejection rate in 5–30%; (2) a matching **difficulty profile**: ExtraTrees held-out $R^2$ quantiles q10–q90 within 0.10 of `graph_scm`'s; (3) an **in-domain check**: in 4 equal-count bins of ExtraTrees $R^2$, the median of $R^2(\text{TabICLv2}) - R^2(\text{ExtraTrees})$ agrees between the priors (within 0.05 or overlapping bootstrap 95% CIs; top, saturated bin not gated). Item 3 compares the backbone's advantage over a prior-free model at the same difficulty: a harder slice of the prior keeps it, a slice outside the backbone's domain loses it. The cut also reports an $R^2$ ceiling $1-\mathbb E[\operatorname{Var}(y\mid x)]/\operatorname{Var}(y)$ from its replicates, which splits its difficulty into noise (low ceiling) and complexity (ceiling far above ExtraTrees $R^2$).

**Result for this design (stage 1, 256 pairs):**

| Measure | Cut | `graph_scm` |
|---|---|---|
| Rejection rate | **29.3%** (106/362; 47.4% of attempts with a private-root ancestor of y, 8.3% of deterministic ones) | 19.7% |
| ExtraTrees $R^2$ q10 / q25 / q50 / q75 / q90 | 0.061 / 0.414 / 0.881 / 0.996 / 1.000 | 0.156 / 0.570 / 0.926 / 0.999 / 1.000 |
| Quantile difference (cut − `graph_scm`) | **−0.095 / −0.156 / −0.044 / −0.003 / +0.000** | |
| In-domain check | all 3 gated bins agree | |
| Exactly deterministic tasks | 46.4% of attempts, 60.2% of accepted tasks | – |
| $\hat t_A$ zero-mass fraction ($R=16$) | **62.5% of queries**; 63% of tasks have $\hat t_A\equiv0$ | – |
| $\hat t_A$ q10 / q25 / q50 / q75 / q90 | **0 / 0 / 0 / 0.051 / 0.377** | – |
| $R^2$ ceiling q10 / q50; share ≥ 0.9 | 0.35 / 1.00; 77.7% | – |

The design passes the rejection-rate and in-domain checks. The difficulty profile is within 0.10 at every quantile except q25 (−0.156): cut tasks are slightly harder in the lower-middle of the distribution, with matching medians and upper tail. This deviation was **accepted**: TabICLv2's advantage over ExtraTrees at matched difficulty is the same on both priors, so the cut stays inside the backbone's domain, and the large exactly-deterministic share is acceptable for the heads (it is the $A=0$ end they must learn). The zero-mass fraction above was measured with replicates drawn in one pass over tiled rows; the same tasks with separate passes give 63% of tasks with $\hat t_A\equiv0$ and no deterministic task with $\hat t_A>0$. Stage 2 and stage 3 results: TBD (preliminary stage-2 cut rejection rate: 34.7%).

**Rejection rate of the training stream (accepted).** In the full training stream (the pipeline above plus splits and replicates, measured by the step-4 throughput benchmark), the stage-1 rejection rate sits somewhat above the 5–30% bound (the validation bank, built with the same pipeline, has 30.6%). This was **accepted** as a documented deviation, not tuned away: the excess is small and was already seen around the calibration above; the rejections come from tabicl's own filters, which define the backbone's prior and are never changed; and its only cost is CPU time per accepted task, already included in the step-4 timings.

### Fixed sizes

Every task has exactly TabICLv2's sizes for the current stage:

$$
n_{\text{ctx}}=\operatorname{round}(f\,N),\qquad m=N-n_{\text{ctx}},
$$

where $N$ is the stage's row count and $f$ the context fraction. Hole and extrapolation queries are carved out of regions whose other rows are **discarded** (an empty region) or mostly discarded (a thinned region returns a few of them to the context, §1.1), so that the context never or rarely sees them. To keep $n_{\text{ctx}}$ and $m$ exact, we generate an **enlarged pool**: the $N$ kept rows plus exactly the rows that will be discarded. The stage-1 subsampled pass (§1.5) reduces the context of an already built task; it does not change these sizes.

### Pipeline

1. **Sample the task shape** from the current stage's settings: $N$, $f$ and the number of features $d$ (TabICLv2's random feature-count sampling). This fixes $n_{\text{ctx}}$ and $m$.
2. **Fix the query budgets** from the query mixture (§1.1):
   $$
   m_{\text{ext}}=\operatorname{round}(p_{\text{extrap}}\,m),\qquad m_{\text{hole}}=\operatorname{round}(p_{\text{hole}}\,m),\qquad m_{\text{iid}}=m-m_{\text{ext}}-m_{\text{hole}}.
   $$
3. **Draw the region sizes per task**, in rows:
   - tail size $T\sim\text{log-uniform}\,[\,m_{\text{ext}},\ \max(m_{\text{ext}},\,0.2N)\,]$;
   - hole size $k\sim\text{log-uniform}\,[\,m_{\text{hole}},\ \max(m_{\text{hole}},\,0.2N)\,]$;
   - the returned rows of each region, $r_{\text{ext}}$ and $r_{\text{hole}}$ (0 for an empty region; the thinning rule of §1.1), drawn independently.
4. **Generate the X pool.** Run the X-graph once to get
   $$
   P=N+(T-m_{\text{ext}}-r_{\text{ext}})+(k-m_{\text{hole}}-r_{\text{hole}})
   $$
   rows. All sizes are drawn before generation, so $P$ is known in advance.
5. **Draw $y^{(0)}$ for every pool row** by forward-passing the y-graph with fresh private noise. The context's $y^{(0)}$ will be the labels the model sees.
6. **Post-process y exactly as TabICLv2's prior does, fitted once, applied everywhere.** Fit the y transform $g$ on the pool's $y^{(0)}$: robust clamp at mean ± 4σ, then standard scaling (`outlier_removing` → `standard_scaling`, `prior/_graph_scm.py`, `prior/_reg2cls.py`), and apply it to $y^{(0)}$.
7. **Filter the task (§1.4).** If it is rejected, discard everything and restart from step 1 with a fresh task.
8. **Split on X with the mixed rule (§1.1).** This yields exactly $m_{\text{ext}}$ extrapolation queries, $m_{\text{hole}}$ hole queries, $m_{\text{iid}}$ i.i.d. queries, $n_{\text{ctx}}$ context rows (of which $r_{\text{ext}}+r_{\text{hole}}$ are returned region rows), and $P-N$ discarded rows.
9. **Make duplicates (§1.2).** Overwrite a fraction of the query slots with x-values copied from context rows. The sizes do not change.
10. **Draw the replicates.** Query rows get R = 16 i.i.d. draws from the y-graph with fresh private noise, used only to build the targets $\hat t_A,\hat t_E$. The queries’ own $y^{(0)}$ served only $g$ and the previous filtering step, and are not reused as replicates. Apply the same $g$ (same bounds, mean and std) to all R replicates.
11. Run the frozen backbone forward once on the $N$ kept rows, along the forward route of Architecture §1, and the heads.
12. **Stage 1 only: the subsampled pass (§1.5).** Run the backbone and the heads a second time on the same task with its context reduced to $n'$ rows; queries and replicates unchanged.

> **Why this normalization.** It reproduces the backbone's prior and keeps targets in the units the heads predict in (proposal §4.3, "fit/apply").
>
> - **Fit $g$ on the whole pool, before the split.** TabICLv2 fits its y transform on all rows of the dataset, context and queries together. Fitting on the full i.i.d. pool, before any hole, tail or duplicate manipulation, reproduces that exactly, and keeps $g$ independent of our split rule.
> - **Clamp the replicates with the fixed bounds.** TabICLv2's "outlier removal" is a clamp, not a row deletion. It is part of the definition of the task law the backbone was pretrained on. Clamped replicates $g(y^{(r)})$ are i.i.d. draws from the clamped law $g_\#p_\tau$, so $\hat t_A$ is **unbiased** for the aleatoric entropy of that law, which is the law the PPD approximates.
> - **Two rules remain.** Never refit $g$ per replicate (that would make the transform random across draws). Never clip the *targets* $\hat t_A,\hat t_E$ themselves (that biases the estimator, §4.5).
> - **Residual caveat.** At deployment only context labels exist, so any inference-time scaling is context-only. TabICLv2 pretraining carries the same small mismatch, which is negligible for i.i.d. queries. Evaluate every Phase 0 test with the same pipeline as training.

### Why manipulating the split is allowed

Every split decision depends **only on X** (and on random numbers independent of y): the tail and the hole, which rows a thinned region returns, the duplicates, and which context rows the subsampled pass keeps. With the y-law generated by its own graph, an X-only selection carries no information about it. So the posterior over the y-law given the manipulated context is unchanged, and the targets keep their meaning as $\mathbb E_{\tau\mid D}[\cdot]$.

Selecting on y, for example dropping context rows with large residuals, would break this. No rule below does that.

### 1.1 Query-type mixture (within every task)

Each task's $m$ queries mix three types. They are built from the pool of $P$ rows in this order, each step removing a region from what is left. **No new rows are created:** every query is an existing pool row, a real draw from the prior. A region's rows are selected as queries, returned to the context (a thinned region only) or discarded, so the context sees a region either not at all or only thinly.

| Type | Default share | Construction (rows left after the step) |
|---|---|---|
| **Extrapolation** | 15% | **1.** Draw a random unit direction $u$ in standardized X space and score all $P$ rows by $s_i=u^\top x_i$.<br>**2.** The $T$ rows with the largest $s_i$ are the **tail**; remove them from the pool.<br>**3.** From those $T$ tail rows, select $m_{\text{ext}}$ uniformly at random as extrapolation queries; of the other $T-m_{\text{ext}}$, return $r_{\text{ext}}$ uniformly at random to the context and discard the rest.<br>*Left:* $P-T$ rows. |
| **kNN hole** | 15% | **1.** Pick a random centre row among the $P-T$ remaining rows.<br>**2.** The centre and its nearest neighbours, $k$ rows in total (standardized X; a single centre, so one distance computation over the remaining rows, fast even for stage-3 pools), are the **hole**; remove them from the pool.<br>**3.** From those $k$ hole rows, select $m_{\text{hole}}$ uniformly at random as hole queries; of the other $k-m_{\text{hole}}$, return $r_{\text{hole}}$ uniformly at random to the context and discard the rest.<br>*Left:* $P-T-k$ rows. |
| **i.i.d.** | 70% | **1.** The $P-T-k=n_{\text{ctx}}-r_{\text{ext}}-r_{\text{hole}}+m_{\text{iid}}$ remaining rows are an ordinary i.i.d. pool.<br>**2.** Select $m_{\text{iid}}$ of them uniformly at random as i.i.d. queries, exactly TabICLv2's random split.<br>**3.** The rest, $n_{\text{ctx}}-r_{\text{ext}}-r_{\text{hole}}$ rows, join the returned rows as the **context**: exactly $n_{\text{ctx}}$ rows, in random order.<br>*Left:* nothing. |

**Bookkeeping.** Queries $=m_{\text{ext}}+m_{\text{hole}}+m_{\text{iid}}=m$; context $=n_{\text{ctx}}$; discarded $=(T-m_{\text{ext}}-r_{\text{ext}})+(k-m_{\text{hole}}-r_{\text{hole}})=P-N$. The model sees exactly $N=n_{\text{ctx}}+m$ rows.

- **Why randomize T and k separately from the query budgets.** It decouples *how far from support* a query is from *how many* queries there are. Hole and extrapolation queries then cover a range of divergences, not one fixed regime. The upper bounds (0.2N for both the tail and the hole) are defaults and can be changed.
- **Why use counts, not radii.** Defining T and k as row counts keeps region sizes meaningful in any dimension, where distances concentrate as d grows.
- **Cost of the enlarged pool.** At most $0.4N$ extra X rows, each with one extra $y^{(0)}$ draw, which is cheap next to the R replicate passes. In stage 3 ($N$ up to 60K) the pool can reach about 84K rows; lower the region bounds there if X generation or the kNN search becomes the bottleneck.

#### Thinned regions

**Why.** A context is rarely either full or empty around a query. Regions with a few rows are where the question "is this spread noise or lack of data?" is hardest, and they are what context retrieval and active learning operate on. So the carved regions cover the whole range from empty to thin.

**Rule.** For each carved region (the extrapolation tail and the kNN hole, drawn independently):

- with probability 0.5 the region is **empty**: no row of it enters the context;
- otherwise it is **thinned**: a share $\rho$, log-uniform in [0.05, 0.2], of its non-query rows is returned to the context.

Log-uniform because the effect of density is roughly multiplicative: halving it matters about as much at 20% as at 5%.

**Sizes.** With tail size $T$, hole size $k$ and query budgets $m_{\text{ext}}$, $m_{\text{hole}}$, a thinned region returns

$$
r_{\text{ext}}=\max\big(1,\operatorname{round}\big(\rho_{\text{ext}}(T-m_{\text{ext}})\big)\big),\qquad
r_{\text{hole}}=\max\big(1,\operatorname{round}\big(\rho_{\text{hole}}(k-m_{\text{hole}})\big)\big),
$$

and an empty region returns $r=0$. The pool size is $P=N+(T-m_{\text{ext}}-r_{\text{ext}})+(k-m_{\text{hole}}-r_{\text{hole}})$ (pipeline step 4).

- Returned rows are chosen uniformly among the region's non-query rows and join the context at random positions.
- The i.i.d. remainder supplies $n_{\text{ctx}}-r_{\text{ext}}-r_{\text{hole}}$ context rows, so the context is exactly $n_{\text{ctx}}$ and the queries exactly $m$.
- All sizes are drawn before generation, so $P$ is known in advance.
- $r\le0.04N$ per region (or $r=1$), far below the smallest context.
- A thinned region returns at least one row, so each region is empty or thinned with probability ½ each, independently of the other: a quarter of the tasks have both regions empty, a quarter only the tail thinned, a quarter only the hole thinned, a quarter both thinned. Only a region with no non-query row ($T=m_{\text{ext}}$ or $k=m_{\text{hole}}$) is always empty.

**Still an X-only rule.** Which rows are returned depends on $X$ and on random numbers independent of $y$, so the targets keep their meaning.

**Query types.** Six types are logged: `iid`, `hole`, `hole_thinned`, `extrapolation`, `extrapolation_thinned`, `duplicate`. A query in a region with zero returned rows is `hole` / `extrapolation`; in a region with returned rows it is the `_thinned` type. Each region's sizes and realized $\rho$ ($r/(\text{region size}-\text{its queries})$) are stored per task.

### 1.2 In-context duplicates

After the split, 5% of query slots (chosen uniformly across all three types) get their x overwritten by the x of a random context row (an i.i.d. context row or a returned one). Only x is copied: the context row’s label is never reused, and the duplicate receives R fresh y draws like any other query. Each duplicate slot records which context row it copies, so that the subsampled pass (§1.5) can relabel it if that row is dropped.

- This teaches the low-divergence end: at an observed x, E is small but not zero under noise. It mirrors DEUP training the error predictor on in-sample errors and its seen-bit feature (DEUP §3, interactive-learning paragraph).
- The overwrite happens before the replicates are generated, so it costs nothing.
- Because duplicates replace slots, the effective mix is 66.5 / 14.25 / 14.25 / 5 (i.i.d. / hole / extrapolation / duplicate), with each carved share split about evenly between its empty and its thinned type.

### 1.3 Mixture as a hyperparameter

```text
query_mix        = [p_iid, p_hole, p_extrap]   # default [0.70, 0.15, 0.15], must sum to 1
region_max_frac  = 0.2                          # upper bound of T and k, as a share of N
thin_empty_prob  = 0.5                          # probability that a carved region is empty (else thinned)
thin_rho_range   = (0.05, 0.2)                  # log-uniform share of a thinned region's non-query rows returned
dup_fraction     = 0.05                         # share of query slots overwritten with context x (CutGraphConfig)
R                = 16                           # training replicates per query
```

The mixture does not bias the targets: the squared-error minimizer is $\mathbb E[\hat t\mid x_*,D]$ at every input, whatever the mix. It only decides **where the heads' capacity is spent**, i.e. how much training coverage the high-divergence and low-divergence regimes get.

### 1.4 Task filtering

TabICLv2 pretrains only on tasks that pass its filters (TabICLv2 §3 "Data filtering", App. E.14; `filter_unpredictable_graphs` and `filter_unpredictable_datasets` are on in every `scripts/train_v2_reg_stage*.sh`). About 25% of stage-1 regression tasks are rejected. The backbone's posterior is therefore conditioned on passing them, so our tasks pass the same filters.

| Check | TabICLv2 code | What it rejects | Our use |
|---|---|---|---|
| **Graph check** | `check_x_y_ancestors_overlap` (`prior/graph_lib/_dataset.py`) | Graphs where no node is a common ancestor of an x node and the y node, so y ⟂ x by construction; the graph is resampled | **Adapted.** The X-graph has no y node, so this filter must not be used in this step. The same function is applied to the **y-graph**, with its X-input nodes tagged as `x`: the y-graph is resampled until at least one X input is an ancestor of y. |
| **Constant columns** | `delete_unique_features` (`prior/_dataset.py`) | Feature columns with a single value are removed; a task with no feature left is rejected | **As-is**, on the X pool. |
| **Predictability test** | `should_filter` (`prior/_dataset.py`) | ExtraTrees (25 trees, depth 6, bootstrap) is fit on the dataset; the task is rejected if its out-of-bag MSE does not beat the mean-label baseline on ≥ 95% of 200 bootstrap resamples (p ≥ 0.05) | **As-is**, on $N$ uniformly drawn pool rows (TabICLv2 tests datasets of $N$ rows) with their $g$-transformed $y^{(0)}$. |
| **Trivial-task test** | `should_filter` with `remove_trivial_datasets` | Tasks where ExtraTrees is almost perfect | **Off**, as in TabICLv2's scripts. Near-deterministic tasks are the $A\approx0$ end the aleatoric head must learn. |
| **Non-finite values** | NaN guard in `GraphSCM` (a NaN dataset becomes constant and is then rejected) | Tasks with NaN or infinite values | Reject the task if any $y^{(0)}$ **or any replicate** is non-finite. |

- **Task-level rejection, before the split.** A rejected task is discarded entirely, and graphs, pool and noise are all redrawn (TabICLv2 retries with a fresh graph each time). It is never repaired by redrawing only the noise. Because the filter runs before the split and the replicates, rejected tasks cost no kNN search and no replicate passes.
- **Why the targets stay unbiased.** The predictability test uses $y^{(0)}$, so it selects on y, but only at the task level, exactly as in TabICLv2: it changes which tasks enter the prior. Given an accepted task, the R replicates are fresh draws from $p_\tau(\cdot\mid x_*)$ that the test never saw, so $\hat t_A,\hat t_E$ stay unbiased, and the heads converge to $\mathbb E[\cdot\mid x_*,D,\text{accepted}]$, the posterior of the filtered prior the backbone amortizes. Query $y^{(0)}$ values are used only by $g$ and the filter, never as targets.
- **Log the rejection rate** per stage (TabICLv2: about 25% in stage 1). The private noise of the observed cut can make tasks noisier than the original prior and raise it. If it is much higher, tune the y-graph's noise scale, not the filter: the filter defines the backbone's prior.
- **Known consequence.** The test removes near-unpredictable tasks, truncating the high-aleatoric tail (proposal §4.3, "Known caveat"). Neither the heads nor the backbone see tasks where y barely depends on X. The empirical distribution of $\hat t_A$ is logged.
- **Same filter everywhere.** The validation bank uses the same filter, so every validation grade is on the backbone's prior.
- **Difficulty-match gate (prior level).** Before training, matched banks of cut and unmodified `graph_scm` tasks are compared (`scripts/00f_difficulty_match.py`); if the cut drifts from `graph_scm`, tune the target graph's private-root noise (`private_root_scale`, `private_root_dims_range`), never the filters. The pass rule, the calibrated configuration and its results are in "The observed-cut prior" above.

### 1.5 Subsampled contexts (stage 1)

**Why.** The smallest context in stage 1's tasks is about 307 rows (30% of 1,024), while the toy set and the real-data tests of Phase 0 use 64–256. Without a smaller context in training, the heads would have no supervision below 307 rows.

**Rule.** In stage 1, every task is used twice in its step:

1. **Full pass:** the task as built (context fraction 30–90% of 1,024 rows).
2. **Subsampled pass:** the same task with its context reduced to $n'$ rows, $n'$ log-uniform in $[64,n_{\text{ctx}}]$. The rows are dropped uniformly at random (the kept rows keep their relative order). Queries and replicates are unchanged.

Details:

- One carving per task: the subsampled pass reuses the task's split.
- $\hat t_A$ is identical in both passes. $\hat t_E$ is recomputed, because the forecast changes.
- One $n'$ per micro-batch group, so a group still has one `(seq_len, train_size)`.
- **Relabelling.** The query types follow the context the model actually sees in that pass:
  - a duplicate whose source context row was dropped takes its source row's type: `iid` if the source was an i.i.d. context row, the region's type if it was a returned row;
  - a thinned region with no returned row left in the subsampled context is empty: its `_thinned` queries become `hole` / `extrapolation`.
- Uniform dropping keeps the relative density of thinned regions.
- A step is 32 tasks and 64 forwards. The loss averages over the query rows of both passes.
- Stages 2 and 3 have one pass per task.

**What it also buys.** The same query is seen under two contexts of the same task, which shows the heads directly how the excess changes with the amount of context.

**Two things to know.**

- TabICLv2 itself was pretrained with at least about 307 context rows. Below that its forecast is off its pretraining range. $\hat E_{\text{dir}}$ is trained on the frozen forecast as it is, so it should learn the larger excess.
- The subsampled pass has a context fraction below 30%. In TabICL a query attends to context rows only, so its forecast should not depend on how many other queries there are. A unit test confirms it: same context, 1 query against $m$ queries, identical quantiles within float32 round-off.

### 1.6 The two task priors

- **carved** (the design): the full pipeline of §1.1–1.2: extrapolation tail, kNN hole, thinned regions, duplicates. Every sweep run uses it.
- **iid** (ablation only): the same generator with TabICLv2's plain random split only: query mix (1, 0, 0) and no duplicates, so $T=k=0$, no thinning and $P=N$. It is trained only for the winning configurations (Training §4).

Both use the stage-1 subsampled pass (§1.5), so the ablation isolates the carving. The iid prior draws the same task distribution as the carved one, not the same tasks: removing the carving changes $P$ and the shape's random draws. Both are validated on the same (carved) bank.

## 2. Curriculum, batch, and optimization

| Stage | Steps (default) | Rows N per task | Context | Passes per task | Head max LR | Fork max LR (B) |
|---|---|---|---|---|---|---|
| 1 | 30,000 | 1,024 | 30–90% (full pass); $n'$ log-uniform in $[64,n_{\text{ctx}}]$ (subsampled pass) | 2 | `lr` | `fork_lr` |
| 2 | 8,000 | 400–10,240 (log-uniform) | ≈80% | 1 | `lr` / 8 | `fork_lr` / 8 |
| 3 | 1,200 | 400–60,000 (log-uniform) | ≈80% | 1 | `lr` / 40 | `fork_lr` / 40 |

- **Step budget.** 40,000 global steps. Stage 1 has 30,000 steps because, in this setup, more stage-1 steps do not give a significant improvement. Stages 2 and 3 have 8,000 and 1,200 steps, each with two full-bank validations (§3), so the effect of the large-context stages is resolved. The steps per stage are inputs.
- **Learning rate.** `lr` (head) and, in B, `fork_lr` are the swept values (§4), both stage-1 values. Stages 2–3 scale both down by TabICLv2's own ratios (8e-4 → 1e-4 → 2e-5, i.e. /8 and /40), as TabICLv2 scaled its whole model. The ratios are fixed, not swept per stage: a configuration is one stage-1 value per LR, and the same ratios set its stages 2–3. Use a cosine schedule **restarted per stage**, as in TabICLv2's scripts: each stage is a separate launch that loads only the previous stage's weights (`--only_load_model True`, so optimizer and scheduler state reset), then warms up for 1% of its steps to its max LR and decays with a single cosine cycle to 1e-7 (`--scheduler cosine_with_restarts --cosine_num_cycles 1 --warmup_proportion 0.01 --cosine_lr_end 1e-7`, `scripts/train_v2_reg_stage{2,3}.sh`). The fork's schedule (B) ends lower, at **1e-8** in every stage (`--fork_cosine_lr_end 1e-8`): its smallest swept LR reaches 1e-5/40 = 2.5e-7 in stage 3, close to the heads' end value of 1e-7 (tabicl's schedule requires the end LR to be below the max LR). The heads keep 1e-7.
- **Optimizer.** AdamW for the heads (both arms) and the fork (B), with weight decay $10^{-5}$, gradient-norm clipping at 5.0 and $\beta=(0.9,0.999)$. In A jobs each head is clipped on its **own** gradient norm (`--clip_per head`): clipping a pair on the joint norm of its two heads would let a large $\hat E_{\text{dir}}$ gradient shrink $\hat A$'s step, coupling heads that share no parameter. In B the fork and both heads are clipped on their joint norm (`--clip_per joint`), since the heads share the fork. These follow DecoupledPFN's defaults (Bergna et al., App. B.3), the closest precedent.
- **Muon is deliberately not inherited from TabICLv2's scripts** (`--muon True` in `scripts/train_v2_reg_stage*.sh`; pass `--muon False` if their flags are reused). Muon helped TabICLv2's pretraining of the full model from scratch (TabICLv2 §4.1, ablation Fig. C.1). It is a pretraining tool, not one for fine-tuning or for training small decoders like the A trunks.


- **Batch.** 32 tasks per step, half of TabICLv2's 64, with the step budget unchanged. This halves the wall time per step: the step time of every stage is proportional to the tasks per step, whether it is set by the CPU workers that generate the tasks or by the backbone forward (step-4 throughput gate, protocol §3). The cost is half as many training tasks per run and noisier gradients per step. Micro-batch groups keep TabICLv2's sizes (4 tasks in stage 1, so 8 groups per step, each with its full and its subsampled pass: 64 forwards; 1 task in stages 2–3, so 32). The backbone runs frozen under `torch.no_grad()`, so no backbone activations are stored. Stage 1 (1,024 rows) fits comfortably on mid-range GPUs. In stages 2–3 (up to 60K rows), reach 32 by **gradient accumulation over micro-batches** (TabICLv2's own mechanism, `micro_batch_size` in `_run.py`). Each group-pass is weighted by its share of the step's query rows, so the step's loss is the mean over all its query rows.
- **Arm B.** It backpropagates through the k = 2 forked blocks, so it needs gradient checkpointing above about 20K rows (TabICLv2 App. B.2 does the same).
- **Feature counts** follow TabICLv2's prior settings (random per task, up to 100 features).

## 3. Validation

### The bank

- **2,000 tasks**, pre-generated once with a fixed seed (`VAL_BANK_SEED = 20261116`), disjoint from the training seeds and from every Phase 0 test.
- The same pipeline as training on the **carved** prior: query mixture, thinned regions and duplicates follow the training distribution. Every run, carved or iid, is validated on this same bank.
- **R_val = 128** replicates per query.
- **Stratified by rows N.** Following the step proportions would give almost only 1,024-row tasks, so large-n bins would be nearly empty. To resemble typical dataset size distribution, the bank is split by rows N:

  | Bin | Rows N | Context fraction | Share | Tasks |
  |---|---|---|---|---|
  | Small | Uniform in 256–1,024 | 80% | 50% | 1,000 |
  | Medium | 1,024–10,240 (log-uniform) | 79–81% | 30% | 600 |
  | Large | 10,240–60,000 (log-uniform) | 79–81% | 20% | 400 |

  ScoringBench subsamples every dataset to at most 3,000 rows (2,400 train + 600 test; ScoringBench App. C.2), and about 70% of TabArena's 51 datasets have fewer than 10K rows (15 have 10K–100K; TabPFN-3 §3.1.1).
- **Precomputed targets.** The PPD is frozen in every arm, so the bank stores, per query, $e(q_\theta)$, $\hat t_A$ and $\hat t_E$ from all 128 replicates, and the half-split targets (replicates 1–64 and 65–128) for the xMSE. No target is recomputed during training.
- **The weight floor** $c=0.1$ (Architecture §3) is stored in the bank's manifest, together with the quantiles of $e(q_\theta)$.
- **The mini-bank** is the first 64 small tasks (N 256–1,024 at 80% context), stored on its own so it loads in a second.

**Slices.** Every metric is reported pooled (`all`), per query type (`iid`, `hole`, `hole_thinned`, `extrapolation`, `extrapolation_thinned`, `duplicate`) and per context-size bin (`n_le256`, `n_256-512`, `n_512-1K`, `n_1K-4K`, `n_4K-16K`, `n_16K-60K`; n is the task's context size $n_{\text{ctx}}$).

**Metrics: two weightings.** Every run reports both, whatever $p$ it was trained with, so $p=0$ and $p=1$ runs compare fairly on either:

- **unweighted** ($p=0$): MSE against the 128-replicate target, xMSE from the two half-split targets (replicate noise removed), and skill $=1-\text{xMSE}/\text{xMSE}_{\text{const}}$;
- **weighted** ($p=1$): the same three with each query's term multiplied by $w=(e(q_\theta)+c)^{-1}$. Weighted means are count-normalized ($\overline{w\,x}$, the form of the loss). The weighted skill uses the best constant under that weighting, the weighted mean target $\sum w\,t/\sum w$, as its baseline.

The reference lines (noise floor, constant baseline, e(q) proxy; protocol §8.1) are computed once on the bank, for both weightings, and stored in its manifest.

> **What the bank showed** (built bank, `VAL_BANK_SEED`; checks of protocol §3 step 6).
> - **Composition.** 2,863,197 queries. Query shares: iid 66.5%, hole 14.2% (6.9 empty + 7.4 thinned), extrapolation 14.2% (7.1 + 7.2), duplicate 5.0%; each carved region is thinned in about half of the tasks. Rejection rate 30.6%; 62.4% of the tasks are exactly deterministic; $\hat t_A$ has a zero mass of 61.7% of the queries.
> - **Context sizes.** 64% of the queries are in `n_16K-60K` and only 0.2% at $n\le256$: the pooled metrics are dominated by the largest contexts, and validation barely sees the smallest ones (the smallest bank context is about 205 rows, 80% of 256), so the effect of the subsampled pass is mostly invisible in validation. Read the per-bin slices, not only the pooled value.
> - **The epistemic target is ordered as expected.** Mean $\hat t_E$: duplicate 0.011 < iid 0.033 < hole_thinned 0.039, extrapolation_thinned 0.044 < hole 0.074, extrapolation 0.107. Pooled, every step of the ordering duplicate < iid < thinned < empty holds with a task-level bootstrap 95% CI excluding 0; within single N bins some thinned-vs-iid and thinned-vs-empty differences are not resolved. Returning even a few rows to a region more than halves its excess. Extrapolation queries carry more epistemic uncertainty than hole queries (CI of hole − extrapolation excludes 0, pooled and in every N bin): predicting beyond the support of the context is harder than filling a gap inside it. Rules of thumb that expect hole ≈ extrapolation (protocol §8.2) should be read with this in mind. Mean $\hat t_A$ is the same for duplicates and iid queries (CI includes 0), as it must be.
> - **The PPD's own uncertainty tracks the aleatoric target, not the epistemic one** (a result for the paper). Test: the **e(q) proxy** reference line (protocol §8.1) is the best predictor of a target that sees only the backbone's CRPS entropy $e(q_\theta)$: 50 equal-count bins of $e(q_\theta)$, each predicting its bin's mean target, fitted on half of the bank's tasks and scored on the other half (cross-fitted on two halves of the tasks, so no query is scored by bins fitted on its own task). Its **skill** is $1-\text{xMSE}_{\text{proxy}}/\text{xMSE}_{\text{const}}$, the share of the target's variance (beyond the constant baseline) that $e(q_\theta)$ explains, with replicate noise removed by the xMSE. On the pooled bank: **skill 0.70 for $\hat t_A$** (xMSE 0.0106 vs 0.0355 for the constant) and **skill 0.12 for $\hat t_E$** (xMSE 0.0306 vs 0.0346); weighted, 0.72 and 0.10. The noise floors (3.7e-4 for $\hat t_A$, 4.6e-5 for $\hat t_E$) are tiny next to these, so the gap is not target noise. So the backbone's total predictive spread is largely a measure of aleatoric uncertainty and says little about how far its forecast is from the task's true law: the epistemic part must be extracted by a dedicated head, which is the thesis of this project (proposal §1.1). For the paper: report both skills as the baseline the Ê head must beat, and as direct evidence that PPD spread is not an epistemic signal.
> - **The weight floor.** $e(q_\theta)$ quantiles q25 / q50 / q75: $4.7\cdot10^{-4}$ / 0.013 / 0.25; 64% of the queries have $e(q_\theta)<c$; with $c=0.1$ the weights run from 10 ($e(q_\theta)\to0$) down to 1.7 at the $e(q_\theta)$ q90 and 1.1 at its q99.
> - **PPD sanity** (mean CRPS of the backbone on iid queries, cut tasks vs `graph_scm` tasks of the same shapes; limit 1.25): small 1.24, medium 1.255, large 0.85, pooled 1.14. The medium bin's 1.255 (CI [1.09, 1.44]) was **accepted** as sampling noise: the cut is slightly harder for the backbone than `graph_scm` at small and medium N, and easier at large N.

### The ceiling of the epistemic skill

An $\hat E_{\text{dir}}$ skill well below 1 is expected: 1 is not reachable by any estimator.

**Scatter.** Fix one context $D$ and one query $x_*$. Many tasks are consistent with that context. Each has its own true law $p_\tau(\cdot\mid x_*)$, so each gives the forecast a different realized divergence $d_S(q_\theta,p_\tau)$. The head outputs one number for $(x_*,D)$. The best number is the average over those tasks,

$$
\bar E(x_*,D)=\mathbb E_{\tau\mid D}\big[d_S(q_\theta,p_\tau)\big],
$$

which is also what the squared-error loss converges to. The task that was actually drawn still differs from this average. That difference is the **scatter**. It is not a defect of the head: knowing which way the realized value deviates would mean knowing more about the task than the context contains, and then the forecast itself could be improved. The xMSE removes the replicate noise, not the scatter.

**A concrete case.** Take a deterministic task and a forecast $q=\mathcal N(0,1)$ that is calibrated, so the true value behaves like a draw from $\mathcal N(0,1)$.

| Where the truth lands | Realized divergence |
|---|---|
| At the centre (0) | 0.23 |
| One s.d. away (1) | 0.60 |
| Two s.d. away (2) | 1.45 |
| Average over all landings | 0.56 |

A perfect head says 0.56 every time. The realized values have a standard deviation of 0.40 around it.

**From the coefficient of variation to the best possible error.** $\mathrm{CV}$ is the scatter's standard deviation divided by its mean (here $0.40/0.56=0.72$). Write the target as $t=\bar E\,\xi$, where $\xi$ is "realized over expected", with mean 1 and standard deviation CV. The perfect head predicts $\bar E$, so its mean squared error is $\mathrm{CV}^2\,\mathbb E[\bar E^2]$, while $\mathbb E[t^2]=(1+\mathrm{CV}^2)\,\mathbb E[\bar E^2]$. Dividing the two:

$$
\text{xMSE}_{\text{best}}=\frac{\mathrm{CV}^2}{1+\mathrm{CV}^2}\;\mathbb E[t^2],\qquad
\text{skill}_{\max}=1-\frac{\text{xMSE}_{\text{best}}}{\text{xMSE}_{\text{const}}}.
$$

This assumes one CV for the whole slice. In reality CV varies by query, and the slice value is an average dominated by high-$E$ queries.

**What the true CV could be.** For an idealised, exactly Bayesian Gaussian forecaster (computed numerically):

| Regime | CV |
|---|---|
| All uncertainty is epistemic (the case above) | 0.72 |
| Epistemic variance 3× the noise variance | 1.07 |
| Epistemic and noise variance equal | 1.21 |
| Noise dominates | 1.41 |

**Implied CV of a head.** Running the formula backwards gives the CV at which a given head would be perfect: $\mathrm{CV}^2/(1+\mathrm{CV}^2)=\text{xMSE}_{\text{head}}/\mathbb E[t^2]$. It is an upper bound on the true CV: if the true scatter is smaller, the gap is error the head could still remove. Equivalently, a head's own skill is a lower bound on the ceiling.

**The bracket.**

- *Upper end, from the bank alone.* On this bank pooled $\mathbb E[t^2]=0.0346+0.0412^2=0.0363$ and $\text{xMSE}_{\text{const}}=0.0346$, so the idealised smallest CV (0.72) gives a pooled $\hat E_{\text{dir}}$ skill ceiling of **0.64**.
- *Lower end, from the best head.* The best pooled $\hat E_{\text{dir}}$ skill reached so far is **0.44**, by the legacy B heads on their own bank (implied pooled CV 1.07; `phase1_protocol.md`, "Legacy checkpoints"). On this bank the lower end is the skill of the selected $\hat E_{\text{dir}}$ head (TBD, after selection).

So the ceiling of the pooled $\hat E_{\text{dir}}$ skill lies between about 0.44 and 0.64, and never at 1. Three observations point to its lower half: the implied CVs of trained heads fall inside the theoretical range; their ordering across query types matches theory (i.i.d. queries, the most noise-dominated, have the highest implied CV; extrapolation queries, the most epistemic, the lowest); and giving the heads more access to the backbone (A → B) moves the skill only a little. This is a consistency argument under idealised assumptions, not a measurement of the ceiling.

**The same point in DEUP.** Lahlou et al. (2023, Fig. 6c and App. D) meet the same limit when they correlate predicted uncertainty with model residuals: their best-case upper bound, from residuals sampled from each predictive distribution, is far from 1 (0.63 ± 0.05 for DEUP, whose observed correlation is 0.47 ± 0.03), and they report the ratio of observed to achievable as the comparison metric.

**A check that would tighten the bracket (TBD).** On the bank's deterministic tasks, replace the idealised Gaussian by the real forecast: for each query, compute the mean and standard deviation of $\mathrm{CRPS}(q_\theta,y)$ for $y$ drawn from $q_\theta$ itself, using the stored quantiles. That gives the true CV on those queries, under the assumption that the backbone is calibrated there.

### What is logged every N steps

- **Default cadence:** full bank every 5,000 stage steps in stage 1, every 4,000 in stage 2 and every 600 in stage 3, and at every stage end: global steps 5,000, 10,000, 15,000, 20,000, 25,000, 30,000 (stage 1), 34,000, 38,000 (stage 2), 38,600, 39,200 (stage 3). Ten validations per run. The best checkpoints are therefore chosen at that resolution. A 64-task mini-bank is evaluated every 250 steps for a finer, cheap curve (protocol §7.4), pooled and per context-size bin; it is not used for checkpoint selection.
- **Cost.** An A job caches $h_*$ on the full bank and the mini-bank once per launch, so an A validation costs only the head MLPs. B's fork cannot be cached on the full bank (its input is the full residual stream of every row), so a B validation reruns the backbone and the fork task by task; only the mini-bank caches the fork input.
- **Metrics:** for each head, MSE, xMSE and skill in both weightings, logged **separately** for $\hat A$ and $\hat E_{\text{dir}}$, plus the ratio $\hat A/e(q_\theta)$ next to the target's own ratio. What happens in the low-uncertainty regime is read from the weighted metrics and the per-type and per-bin breakdowns.
- **Breakdowns:** the slices above.
- **Reading the curves.** Under a given weighting, the target-noise term of the MSE is identical for every checkpoint and run, so differences between runs are differences in true error. Because the bank is fixed, every comparison is paired (same tasks), which removes task-to-task noise.

> **Scoring a configuration.**
> - *Arm A: per head.* Every run reports $\text{MSE}_A$ and $\text{MSE}_E$ separately, and learning rates are chosen **per head**. In A the heads share no parameters, so the best $\hat A$ head and the best $\hat E_{\text{dir}}$ head may come from different learning rates (same $p$ and prior) and are combined into one model.
> - *Arm B: the configuration as a whole.* The heads share the fork, so heads cannot be mixed. A configuration is scored by
> $$\text{score}=\frac{\text{MSE}_A}{\text{MSE}_A^{\text{ref}}}+\frac{\text{MSE}_E}{\text{MSE}_E^{\text{ref}}},$$
> where $\text{MSE}_A^{\text{ref}}$ and $\text{MSE}_E^{\text{ref}}$ are those of arm A's selected heads at the same $p$ (and prior). Each term is that head's error relative to A's, so a score below 2 means better than A overall.
> - *Weighting.* Every MSE above is the pooled validation MSE under the configuration's own weighting: unweighted for $p=0$, weighted for $p=1$. Scores are never compared across $p$ (Architecture §3).

### No stopping rule, best-per-head checkpoints

There is **no early stopping**, as in TabICLv2. Each run trains to its fixed budget. The validation curve serves to:

- check that training improves, and
- compare runs and checkpoints: learning rates, arms and seeds (§4).

**Checkpoints kept per run and head pair, for both arms** (updated at every full validation):

- **latest:** overwritten each time;
- **best $\hat A$ so far** and **best $\hat E_{\text{dir}}$ so far:** each overwritten only when that head's pooled full-bank validation MSE improves, under the pair's own weighting ($p=0$: MSE; $p=1$: weighted MSE).

A run's $\hat A$ head is its best-$\hat A$ checkpoint, and its $\hat E_{\text{dir}}$ head its best-$\hat E_{\text{dir}}$ checkpoint. This is checkpoint selection, not early stopping: training always runs to the end.

> **Comment on B.** In A the two heads are independent, so selecting each one at its own best checkpoint is exact. In B the heads share the fork, so the best-$\hat A$ and best-$\hat E_{\text{dir}}$ checkpoints are two different states of the fork (each stores the whole model: fork and both heads), and per-head selection may not be the best criterion: the fork can become very good for one head at the cost of the other. Both heads are logged at every validation, so this trade-off is visible.

## 4. Sweeps, seeds, and selection

### Sweep grid (seed 0, carved prior)

| Arm | Head LR | Fork LR | k | p | Configurations |
|---|---|---|---|---|---|
| A | {3e-4, 1e-3, 3e-3} | — | — | {0, 1} | 6 head pairs, one job |
| B | {3e-4, 1e-3, 3e-3} | {1e-5, 3e-5, 1e-4} | 2 | {0, 1} | 18, one job each |

The grids are set so that the best values are not at their edges: the head grid is centred on 1e-3, and the fork grid reaches ten times Bergna et al.'s fork LR. There is no pull-to-pretrained penalty in B. Every run logs both weightings (§3).

**Seeds.** Every configuration of every arm uses the **same seeds**: seed $s$ fixes both the task stream and the head initialization, so all comparisons are paired, and the A job and the B jobs of seed $s$ see the same tasks. The sweep runs seed 0 only; seeds 1 and 2 are run only for the winning configurations, to put error bars on the final model and the ablation.

### Procedure

1. **Learning rates, from validation.** Within each (arm, $p$, prior) cell, pick the learning rates by the pooled validation MSE under that cell's own weighting (§3, "Scoring a configuration"): in A per head, in B the configuration as a whole.
2. **Final model, from the tests.** The four carved winners (A and B, $p=0$ and $p=1$) are run on the Phase 0 tests T2, T3, T4 and T5, and the best is chosen. Validation cannot make this choice: each weighting favours the run trained with it, and the arms differ in cost as well as in error. $p=1$ is expected to win, since it improves small-value estimates.
3. **Ablation (appendix).** The final model against three counterparts, each differing in one factor: the other $p$, the other arm, and the iid prior (§1.6). All on the heatmap, the PRR tests and active learning, with seeds 0–2. Learning rates are not ablated.
4. **The selected checkpoints** (per head for A) are recorded in `phase1_protocol.md` and in `iltm2.inference.BEST`; the final model then runs on T1 and T6.

> **Comments.**
> - *Selection on the test data.* The final model and $p$ are chosen with T2–T5, and the same tests are then reported. That is mildly optimistic. Either state it with the results, or select on a fixed subset (for example T2 plus one dataset) and report on the rest (open).
> - *What "seed" means.* A seed changes **both** the head initialization and the task stream (§5 explains why this matters for the shared jobs). The iid prior on seed $s$ draws different tasks from the carved prior on seed $s$ (§1.6).

### Risks to watch

- **Higher learning rates.** Head LR 3e-3 may be unstable (loss spikes around step 15,000 have been seen at 1e-3). Fork LR 1e-4 has no penalty pulling the fork towards its pretrained weights: watch `fork_dist_from_init` in the first hours of the B jobs.
- **Small contexts may be forgotten in stages 2–3.** Stage 1 is the only stage with subsampled contexts, and stages 2–3 are long. Watch `n_le256` and `n_256-512` at global steps 34,000–40,000. If they degrade, mix a share of stage-1-shape tasks into stages 2–3.
- **Validation barely sees the smallest contexts** (0.2% of the bank's queries at $n\le256$, none below about 205 rows; §3). Drawing the small tasks from 80 rows up (context 64) would cover the range of the subsampled pass (open).

## 5. Job layout and the shared-forward saving

### The saving

In arm A nothing upstream of the heads trains. For a given batch, the backbone forward (TFcol → TFrow → TFicl → LN → decoder → $q_\theta$, $e(q_\theta)$) produces **the same tensors no matter which head reads them**. That forward is almost all of the compute per step. A head pair is about 1M parameters, a negligible cost next to a 12-block transformer over thousands of rows. $p$ only changes a head's loss, so both values ride on the same pass.

So **one job can run the backbone once per pass and feed many independent heads**:

```python
for step in range(total_steps):
    groups = sample_step()                                       # 32 tasks: X → split → thinning → duplicates → y (§1)
    for group in groups:
        for view in group.passes:                                # stage 1: full + subsampled context; stages 2-3: full
            with torch.no_grad():
                h_star, q = backbone(view)                       # h_* at the query rows; q_θ
                e_q = crps_entropy(q)
                tA, tE = targets(q, view.y_reps)                 # R replicates; t_A is the same in both passes
            for pair in pairs:                                   # every head LR × p of the job
                loss = pair.loss(h_star, tA, tE, e_q)            # w = (e_q + c)^(-p), the pair's own p
                (loss * view.share_of_step_queries).backward()   # mean over all the step's query rows
    for pair in pairs:
        pair.clip_per_head(); pair.opt.step(); pair.sched.step(); pair.opt.zero_grad()
```

- Each head pair has its **own optimizer and LR schedule**, so the pairs stay fully independent: separate parameters, separate gradients.
- In B the fork changes the forward pass and $p$ changes the fork's gradients, so each configuration needs its own job.

### Layout

| Round | Job | Contents | Count |
|---|---|---|---|
| Sweep | `A_carved_seed0` | 6 head pairs (3 head LR × 2 $p$), ids `A_lr<LR>_p<P>` (e.g. `A_lr1e-3_p1`), on one backbone stream | 1 |
| Sweep | `B_h<H>_f<F>_p<P>_carved_seed0` | one fork and one head pair per job | 18 |
| Winners | the winning configurations on seeds 1, 2 (`A_carved_seed<S>`, `B_…_carved_seed<S>`) | error bars for the final model and the ablation | TBD |
| Ablation | the winning configurations on the iid prior, seeds 0–2 (`A_iid_seed<S>`, `B_…_iid_seed<S>`) | the carving ablation (§4, step 3) | TBD |

- **Names.** Learning rates are written with up to 2 significant digits; a repeated (LR, $p$) in one A job gets a version suffix (`_v1`, `_v2`). An A job can train any subset of pairs (`--head_lrs`, `--p`).
- **Stages.** One SLURM job per run, stages 1 → 2 → 3 in sequence, one wandb run.
- **Order.** The A job runs first. The 18 B jobs follow once the Phase 0 tests are written and their baselines run; then selection and evaluation (§4).

Each job varies both initialization and data through its seed, and the B jobs of seed $s$ use the same task stream as the A job of seed $s$ (the stream does not depend on the number of workers), so arm comparisons are paired too.

## 6. Defaults at a glance

| Setting | Default |
|---|---|
| Task prior | carved (every sweep run); iid only for the winning configurations' ablation |
| Query mix (i.i.d. / hole / extrapolation) | 0.70 / 0.15 / 0.15 (iid prior: 1 / 0 / 0) |
| Duplicate fraction | 0.05 of query slots (iid prior: 0) |
| Hole | random centre; $k$ log-uniform in $[m_{\text{hole}},\max(m_{\text{hole}},0.2N)]$ per task |
| Extrapolation tail | random direction; $T$ log-uniform in $[m_{\text{ext}},\max(m_{\text{ext}},0.2N)]$ per task |
| Thinning | each carved region independently: empty with probability 0.5, else $r=\max(1,\operatorname{round}(\rho\cdot\text{non-query rows}))$ returned, $\rho$ log-uniform in [0.05, 0.2] |
| Subsampled pass (stage 1) | every task; $n'$ log-uniform in $[64,n_{\text{ctx}}]$, one per micro-batch group; queries and replicates unchanged |
| R (train) / R_val | 16 / 128 |
| Steps per stage | 30K / 8K / 1.2K (40K global steps) |
| Batch | 32 tasks (half of TabICLv2's 64; 64 forwards in stage 1), with micro-batch accumulation in stages 2–3 |
| Head LR (stage 1); stages 2/3 | swept {3e-4, 1e-3, 3e-3}; ×1/8, ×1/40 |
| Fork LR (B, stage 1); stages 2/3 | swept {1e-5, 3e-5, 1e-4}; ×1/8, ×1/40 |
| Cosine schedule | restarted per stage, 1% warm-up, one cycle; end LR heads 1e-7, fork (B) 1e-8 |
| Fork depth k (B) | 2 |
| Optimizer | AdamW (heads and fork), weight decay 1e-5, β = (0.9, 0.999); no Muon |
| Gradient clipping | 5.0 (per head in A jobs; fork + heads jointly in B) |
| Output-bias initialization | $\mathrm{softplus}^{-1}$ of the first batch's mean target (weighted mean for $p=1$), zero last-layer weights |
| Forward route | one raw training-mode forward (`_icl_predictions`), no `TabICLRegressor` ensemble, everywhere: training, validation, Phase 0 |
| Loss weight | $w=(\mathrm{sg}[e(q_\theta)]+c)^{-p}$, $p\in\{0,1\}$ per head pair, $c=0.1$ (bank manifest) |
| Seeds | 0 for the sweep; 1, 2 for the winning configurations |
| Validation | 2,000 tasks (1,000 / 600 / 400 by N), both weightings, 6 query types × 6 context-size bins; global steps 5K, 10K, …, 30K, 34K, 38K, 38.6K, 39.2K; mini-bank (64 small tasks) every 250 steps |
| Stopping | none; keep latest + best-$\hat A$ + best-$\hat E_{\text{dir}}$ checkpoints per pair, under the pair's own weighting |
| Selection | LR per (arm, $p$, prior) cell from validation (A per head, B whole configuration) → final model among the four carved winners from T2–T5 → ablation (other $p$, other arm, iid prior) |
| Task filtering | graph check (on the y-graph), constant columns, ExtraTrees predictability test (p < 0.05); trivial-task test off |
