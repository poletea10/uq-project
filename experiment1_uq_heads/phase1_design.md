# Phase 1 Process Outline

This document gives a thorough understanding of the whole model training phase (Phase 1), from the new architecture and loss to the actual training procedure. The actual step-by-step organisation and code structure is explained in the `phase1_protocol.md` file found in the same sub-folder.

# Architecture and loss

## 1. Heads

Two scalar heads are added on top of the frozen TabICLv2 regressor. $e(q_\theta)$ is still computed in closed form from the quantile head (§2.3) but is **not** part of either head's output.

$$
\hat A(x_*,D)=\mathrm{softplus}\big(a_\theta(h_*)\big),\qquad
\hat E_{\text{dir}}(x_*,D)=\mathrm{softplus}\big(b_\theta(h_*)\big)
$$

- **$h_*$** is the representation the quantile decoder reads: the post-final-LayerNorm output of the ICL transformer (TFicl) at the query row, a 512-d vector (`_model/learning.py`: `src = self.ln(self.tf_icl(R, train_size))`, then `self.decoder(src)`).
- **Trunk.** One trunk per head, mirroring the quantile decoder (a 2-layer MLP, 512 → 1024 → 999; TabICLv2, App. A.4):

  `Linear(512 → 1024) → GELU → Linear(1024 → 1) → softplus`

  Separate trunks keep the noisier $\hat t_E$ gradients from degrading $\hat A$. Zero-init the last layer's weights and set its bias to $\mathrm{softplus}^{-1}$ of the mean target on a warm-up batch: the first batch of stage 1 (`--init_bias warmup`, the default; `--init_bias bank` would use the bank-wide constant instead).
- **Why softplus.** It behaves like $e^z$ for $z\ll0$, which resolves values spanning orders of magnitude (the backbone check shows $E$ from $10^{-4}$ to $10^{0}$), and like $z$ for large $z$, so squared-error gradients do not explode. The non-negativity is correctly specified: $\hat t_A\ge0$, and $\mathbb E[\hat t_E\mid x_*,D]=\mathbb E_{\tau\mid D}[d_S(q_\theta,p_\tau)]\ge0$.
- **No coupling to $e(q_\theta)$, and no gate.** That was the first idea we had, but we saw that this would: (1) contaminate our estimations the amortization gap, and (2) couple the divergence and aleatoric entropy to a value that sometimes moves in opposite direction (we saw in the backbone check that e(q) does not follow the actual divergence spread in sparse context intervals).
- **Units.** Both heads live in TabICLv2's internal z-normalized $y$ units (the units of the pinball loss, `_finetune/regressor.py`). Multiplying by the context std $s_y$ gives raw units exactly, since CRPS entropy and divergence are positively 1-homogeneous under $y\mapsto ay+b$.
- **Forward route.** Training, validation and every Phase 0 evaluation of iLTMv2 run **one raw forward pass** of the TabICLv2 model, the same computation TabICLv2 pretraining runs (`train/_run.py: run_micro_batch`, `self.model(X, y_train, d)`):
  - The backbone runs under `torch.no_grad()` so we can get $h_*$ and not save any gradient info in that forward pass.
  - Do **not** use `model.eval()` + `model(...)`. In eval mode `ICLearning.forward` switches to `_inference_forward`, a different code path. Dropout is 0, so the training-mode path is deterministic. Unit test at startup: the hooked forward's quantiles are `torch.equal` to tabicl's own training-mode forward.
  - **No `TabICLRegressor` use for the tests.** The sklearn wrapper adds its own X preprocessing, context-based y scaling and an ensemble of `n_estimators` variants whose predictions are averaged. That averaged forecast is a different $q_\theta$. $\hat E_{\text{dir}}$ is trained to the divergence of **one specific forecast** (proposal §4.5, "checkpoint-specific"), so it is only valid for the forward that produced its targets. An ensemble would change both $e(q_\theta)$ and the excess the head was trained to predict.
  - **Consequence for Phase 0.** The iLTMv2 adapter (`phase0_protocol.md` §5.5) uses this same route, with context-only y scaling at deployment (residual caveat, Training §1). Any TabICLv2 row compared against iLTMv2's PPD also uses this route, so the forecasts being compared are identical.

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

> **Ablation (only if the first experiments fail).** Append one scalar, the standardized $\log e(q_\theta)$, to the trunk input (512 → 513). It adds no information, since $e(q_\theta)$ is a function of $h_*$, but it hands the trunk DEUP's variance feature ready-made. Use stop-gradient on it ($\mathrm{sg}[\cdot]$, i.e. `.detach()`: used in the forward pass, treated as a constant in backpropagation).

## 2. Targets: raw squared error and replicates

Given $R$ i.i.d. replicates $y^{(1)},\dots,y^{(R)}$ of the query target from the observed-cut prior (§4.3):

$$
\hat t_A=\frac{1}{R(R-1)}\sum_{r<s}\lvert y^{(r)}-y^{(s)}\rvert,\qquad
\hat t_E=\frac1R\sum_r S\big(\mathrm{sg}[q_\theta],y^{(r)}\big)-\hat t_A .
$$

- **Raw squared error only.** Never clip $\hat t_E$ at zero, even though single targets can be negative. All of these break the conditional-mean argument of §2.4.
- **Start with $R=16$.** Each extra replicate costs one more pass of the target graph over the query rows. More replicates do not move the optimum; they cut target variance, mostly for $\hat t_E$, which is a difference of two noisy terms. Lower $R$ only if target generation becomes the bottleneck.
- **One CDF convention throughout.** Compute $e(q_\theta)$, $S(q_\theta,y^{(r)})$ and the grading of Phase 0 (`phase0_protocol.md` §5.4) from the same CDF built from the quantiles (tails or no tails). Otherwise $\hat r$ picks up a constant offset.

## 3. Loss

$$
\mathcal L=\overline{\,w\,\big(\hat A-\hat t_A\big)^2\,}+\lambda_E\,\overline{\,w\,\big(\hat E_{\text{dir}}-\hat t_E\big)^2\,},
\qquad w=\big(\mathrm{sg}[e(q_\theta)]+c\big)^{-p},
$$

with the bars denoting averages over query rows. The pinball loss is absent because the PPD is frozen. Since the heads share no parameters, $\lambda_E$ only rescales the optimizer step of the $\hat E$ head.

**Why the weight.**

- *The problem.* Target noise scales with the quantity itself. For Gaussian noise of scale $\sigma$ and $R=2$, $\hat t_A$ has mean $\sigma/\sqrt\pi$ and standard deviation $\approx0.43\sigma$. Unweighted squared error is therefore dominated by high-uncertainty queries, while relative errors on small values (dense regions, small $E$) are nearly free (the loss barely affects training where granularity is needed). The binned reliability plots grade every region, so that matters.
- *Why the weight is safe.* For a fixed input, $\mathbb E[w(a-\hat t)^2\mid x_*,D]=w\cdot\mathbb E[(a-\hat t)^2\mid x_*,D]$. Any weight that depends **only on $(x_*,D)$** leaves the minimiser at the conditional mean; it only changes how a finite head trades off errors *across* inputs. A weight depending on $y$ (e.g. $1/\hat t^2$) would bias the heads.
- *Why $e(q_\theta)$.* It is a free, frozen proxy for each query's noise scale computable from $(x_*,D)$. It does not need to be accurate: a poor proxy only reduces the efficiency gain and cannot bias the result.
- *The exponent and floor.*
  - $p=2$ is inverse-variance weighting (variance $\propto$ scale$^2$), the efficient weighted-least-squares choice.
  - $p=0$ is no weighting.
  - $c$ (e.g. the 5th percentile of $e(q_\theta)$ over the prior) stops near-deterministic queries from receiving enormous weights.
- *Protocol.* $p=0$ is the default in **every** run. $p=2$ is a fallback ablation, run only if, after all tests, T2 shows the low-uncertainty bins badly calibrated while the high-uncertainty bins are well calibrated. The weight matters most for $\hat E_{\text{dir}}$, whose range is widest.

> **Epistemic fraction.** Report $\lambda=\hat E_{\text{dir}}/(\hat A+\hat E_{\text{dir}})\in[0,1]$, the share of the forecast's expected score that is reducible (since $\hat A+\hat E_{\text{dir}}$ estimates the expected score of $q_\theta$, §2.2). It is scale-free and comparable across datasets, which resolves the note of §4.6.

## 4. Connecting the heads to the backbone

### The caveat

Pinball pretraining only asks $h_*$ to encode the posterior *predictive* $q_\theta$, a mixture of task laws. The decomposition needs the posterior *over tasks*: $A=\mathbb E_{\tau\mid D}[e(p_\tau)]$ is not a function of the mixture alone (Bergna et al., Prop. 1).

The PFN amortizes the posterior and never had to represent it explicitly. So nothing forced $h_*$ to keep what tells "a mix of sharp laws" apart from "one wide law". It may have kept it (TabICLv2's quantile bands do track heteroscedastic noise, TabICLv2 App. I), but this is not guaranteed.

### The ladder of arms

A **ladder** is a sequence of setups, each giving the heads strictly more access to the backbone's information than the previous one, at a higher cost. We climb only while the next rung brings a real improvement.

It turns "is $h_*$ enough?" into a measurement, and the rung where gains stop tells us where the decomposition information lives. That finding is itself a result about what PFNs keep when they amortize the posterior.

**Comparing rungs needs no ground truth.** On held-out cut-graph tasks,

$$
\mathbb E\big[(\hat A-\hat t_A)^2\big]=\mathbb E\big[(\hat A-A)^2\big]+\text{target-noise variance}.
$$

The noise term is identical across arms, so a lower validation loss *is* a lower true error. T2 grading (slope, intercept, reliability plots) then confirms it.

| Term | Meaning |
|---|---|
| Frozen | Weights not updated during training |
| Residual stream | Each transformer block adds its update to a running vector per row; $r_\ell$ is that vector after block $\ell$, and $h_*$ is the normalized $r_{12}$ at the query row |
| LayerNorm (LN) | Rescales a vector to zero mean and unit variance across its components, then applies learned per-component gains. TabICLv2 is pre-norm, so the stream's magnitude grows with depth and LN puts different depths on one scale |
| Fork | A branch split off the main computation path: a copy that continues separately |

```
R ─► TFicl blocks 1 … 12 (frozen) ─► LN (frozen) ─► h [A1: read from TFicl's last block] ─► quantile decoder (frozen) ─► q_θ ─► e(q_θ)
          │                                         │
          │ r_1 … r_12 ──► [A2: weighted layer mix] ─┤
          │                                         ├─► h_* ─► trunk_A ─► Â
          └─ r_{12−k} ──► [B: trainable copy of     │        └► trunk_E ─► Ê_dir
                           blocks 13−k … 12 + LN] ──┘
```

#### Arm A1 — MLP trunks on $h_*$ (default model)

- **Procedure.** The trunks of §1 read $h_*$. The backbone is frozen and run under `torch.no_grad()`, which saves most activation memory. The PPD is bit-identical to TabICLv2.
- **What it tests.** Whether the split is present in $h_*$, possibly entangled nonlinearly.
- **Moving on.** If its validation loss and T2 reliability are poor, $h_*$ has probably discarded the information.

#### Arm A2 — multi-layer tap

- **Procedure.** Take the query row's residual-stream vectors after several blocks (e.g. $r_4,r_8,r_{12}$, or all 12). Give each its own trainable LN, combine them as $\sum_\ell\alpha_\ell\,\mathrm{LN}_\ell(r_\ell)$ with $\alpha=\mathrm{softmax}(s)$ (learned scalars: positive, summing to 1), and feed the result to the same trunks. The backbone stays frozen and the PPD unchanged.
- **Why it addresses the caveat.** Only the last layer was forced to serve the pinball objective. Intermediate layers may still hold information that later layers merged away because the PPD did not need it, e.g. how strongly nearby context targets disagree before being collapsed into one predictive.
- **What the weights do not show.** The learned $\alpha_\ell$ do **not** say which block holds the decomposition information, for two reasons. (1) The trunk sees $\alpha_\ell$ times $\mathrm{LN}_\ell$'s learned gains, so a small $\alpha_\ell$ can be compensated by large gains: the weights are not identifiable on their own. (2) The residual stream is cumulative ($r_\ell=r_{\ell-1}+$ block $\ell$'s update), so information present at block $\ell$ is usually still present at later blocks, and the mix can draw it from any of them. The $\alpha_\ell$ are logged only as a training diagnostic (e.g. weight piling on $r_{12}$ means A2 is behaving like A1). A2's result is its validation loss against A1's.
- **Cost.** A few extra 512-d vectors per query.

#### Arm B — trainable fork of the last $k$ blocks

- **Procedure.**
  - Copy the last $k\in\{1,2\}$ TFicl blocks and the final LN, initialized with the pretrained weights.
  - The copy receives the same input as the originals, the residual stream after block $12-k$, for **all rows**. Context rows must be processed too, because in every block the query attends to them.
  - The copy's output feeds **only** the heads, and its weights are trained with the UQ loss. The original blocks stay frozen and keep feeding the decoder.
- **Why it beats unfreezing shared layers.** Bergna et al. unfreeze the final two shared layers, which changes the PPD (DecoupledPFN, App. B.2). Here:
  - The quantile path is untouched, so the PPD stays bit-identical by construction.
  - At step 0 the copy outputs exactly $h_*$, so B starts where A1 ends and can only add.
  - $\hat E_{\text{dir}}$'s target stays fixed, so no $\lambda_E=0$ warm-up is needed.
- **Why it can recover posterior information.** The copied blocks re-attend to the context under a new objective. They can learn to compute what the original last layers had no incentive to compute, such as how many context rows are relevant to $x_*$ and how much their targets disagree.
- **Limitation.** It can only recompute from what survives in the stream at block $12-k$; if earlier layers dropped the information, $k$ must grow.
- **Settings.** Learning rate about $10^{-5}$ for the copy versus about $3\times10^{-4}$ for the heads (Bergna et al.'s split, App. B.3). If results are not as expected, we can add an L2 penalty pulling the copy toward its pretrained weights, to avoid overfitting the cut-graph prior.
- **Cost.** About $k/12$ extra TFicl compute, plus backpropagation through $k$ blocks.

**Order.** A1 → A2 → B. Each rung is kept only if it lowers held-out validation loss and improves T2 reliability over the previous one.

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

### The observed-cut prior: what we changed from `graph_scm`

**The original prior.** TabICLv2 was pretrained only on `graph_scm`. Per dataset it samples a random DAG (Cauchy edge model, 2–32 nodes), puts random-point vectors at the root nodes and pushes them through random node functions (MLPs, trees, GPs, discretizations, …). Each node is a vector of LogInt(1, 32) latent dims plus the dims its converters read; features and the target y are read from random nodes by converters (numerical, with optional warping, or categorical). There is no node noise (`graph_noise False`): all aleatoric spread is omitted-variable noise, the dimensions upstream of y that the features do not pin down. A dataset is then post-processed (outlier clamp, standardization) and filtered (graph check, constant columns, ExtraTrees predictability test).

**Why it cannot be used as is.** X and y share upstream randomness U, so an exact replicate of y at a fixed x needs U from the abduction posterior $p(U\mid\Phi(U)=x)$, which these dense nonlinear graphs make intractable (proposal §4.3, CLAUDE.md "abduction rule").

**What we changed.** Each change is the smallest one that makes replicates exact or brings the cut's difficulty back to `graph_scm`'s.

| # | Change | Why |
|---|---|---|
| 1 | **The cut.** X comes from an X-graph with `graph_scm`'s mechanism but no y node. y comes from a second, `graph_scm`-style **target graph** (same DAG sampler and size range, same node functions): $y=\Psi(X_{\text{obs}},U_2)$, $U_1\perp U_2$. | Nothing has to be inferred from X: a replicate is "hold X, redraw $U_2$, rerun Ψ". |
| 2 | **X enters as a parent value.** The target graph's X-input nodes are a random subset of the X-graph's feature-hosting nodes, placed on random DAG positions whose incoming edges are cut. The observed post-converter values of a host's columns are the node's parent value; the node is an ordinary non-root `graph_scm` node (`RandomMultiFunction` of the block, LogInt(1, 32) latent dims, std → importance weights → L2 → node importance), so it is a deterministic function of X. | Randomness enters only at roots, as in `graph_scm`. An earlier version added fresh noise dims to X-input nodes; it made cut tasks far noisier (first gate run: median ExtraTrees $R^2$ 0.44 vs 0.92, 52% rejections). |
| 3 | **Observed values only; the `*_id` write-back is taken out.** X-input values are recomputed from the observed, outlier-clamped columns. For the categorical modes `neighbor_id` and `softmax_id`, whose write-back is the continuous latent (not determined by the observed category), the category's center / embedding is used instead. | y may depend on $U_1$ only through what the model sees; otherwise replicates are not i.i.d. draws from $p(y\mid x)$. |
| 4 | **No noise-ancestor check.** Only the proposal's graph check remains (an X input is an ancestor of y). When no private root is an ancestor of y, the task is exactly deterministic ($A=0$ at every x). | Requiring a private root to reach y forced irreducible noise into every task and removed `graph_scm`'s near-deterministic mass (median ExtraTrees $R^2$ 0.26–0.56 with the check, 0.92–0.94 without). About 60% of accepted stage-1 tasks are exactly deterministic. |
| 5 | **Lower private-root noise scale:** private roots' outputs × 0.1 (`private_root_scale`, i.e. a 10× smaller node importance). | Private roots are $U_2$, the only randomness; X never sees them, so their whole effect on y is irreducible. The scale lowers it. |
| 6 | **Lower private-root dimension:** LogInt(1, 16) dims instead of LogInt(1, 32) (`private_root_dims_range`). | Tree-type node functions split on random points of the data, so they are blind to an input's scale; fewer noise dims also weakens private roots there. With 5 alone the rejection rate stayed at 35%; with 5 + 6 it fell to 25%. |

Two implementation details keep the replicates exact: replicates are drawn as **R separate passes over the same rows**, so the $U_2$-independent part of y is bit-identical across replicates (a single pass over tiled rows let float32 round-off, amplified by tree splits and argmins, put spurious spread of up to 0.2 std(y) on deterministic tasks); and tasks whose target-graph inputs, $y^{(0)}$ or replicates are non-finite are rejected.

**How the cut is checked against `graph_scm`: the difficulty-match gate** (`scripts/00f_difficulty_match.py`, §1.4). It generates matched pairs of cut and unmodified `graph_scm` tasks (same N, $n_{\text{ctx}}$, d; plain i.i.d. split; `graph_scm` with TabICLv2's pretraining config) and requires, per stage: (1) a cut rejection rate in 5–30%; (2) a matching **difficulty profile**: ExtraTrees held-out $R^2$ quantiles q10–q90 within 0.10 of `graph_scm`'s; (3) an **in-domain check**: in 4 equal-count bins of ExtraTrees $R^2$, the median of $R^2(\text{TabICLv2}) - R^2(\text{ExtraTrees})$ agrees between the priors (within 0.05 or overlapping bootstrap 95% CIs; top, saturated bin not gated). Item 3 compares the backbone's advantage over a prior-free model at the same difficulty: a harder slice of the prior keeps it, a slice outside the backbone's domain loses it. The cut also reports an $R^2$ ceiling $1-\mathbb E[\operatorname{Var}(y\mid x)]/\operatorname{Var}(y)$ from its replicates, which splits its difficulty into noise (low ceiling) and complexity (ceiling far above ExtraTrees $R^2$).

**Result for this design (stage 1, 256 pairs; job 45910541):**

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

The design passes the rejection-rate and in-domain checks. The difficulty profile is within 0.10 at every quantile except q25 (−0.156): cut tasks are slightly harder in the lower-middle of the distribution, with matching medians and upper tail. This deviation was **accepted**: TabICLv2's advantage over ExtraTrees at matched difficulty is the same on both priors, so the cut stays inside the backbone's domain, and the large exactly-deterministic share is acceptable for the heads (it is the $A=0$ end they must learn). The zero-mass fraction above was measured before the separate-pass fix; regenerating the same tasks after the fix gives 63% of tasks with $\hat t_A\equiv0$ and no deterministic task with $\hat t_A>0$. Stage 2 and stage 3 results are added when that run finishes (preliminary stage-2 cut rejection rate: 34.7%).

**Rejection rate of the training stream (accepted).** In the full training stream (the pipeline above plus splits and replicates, measured by the step-4 throughput benchmark), the stage-1 rejection rate sits somewhat above the 5–30% bound. This was **accepted** as a documented deviation, not tuned away: the excess is small and was already seen around the calibration above; the rejections come from tabicl's own filters, which define the backbone's prior and are never changed; and its only cost is CPU time per accepted task, already included in the step-4 timings.

### Fixed sizes

Every task has exactly TabICLv2's sizes for the current stage:

$$
n_{\text{ctx}}=\operatorname{round}(f\,N),\qquad m=N-n_{\text{ctx}},
$$

where $N$ is the stage's row count and $f$ the context fraction. Hole and extrapolation queries are carved out of regions whose other rows must be **discarded**, so that the context never sees them. To keep $n_{\text{ctx}}$ and $m$ exact, we generate an **enlarged pool**: the $N$ kept rows plus exactly the rows that will be discarded.

### Pipeline

1. **Sample the task shape** from the current stage's settings: $N$, $f$ and the number of features $d$ (TabICLv2's random feature-count sampling). This fixes $n_{\text{ctx}}$ and $m$.
2. **Fix the query budgets** from the query mixture (§1.1):
   $$
   m_{\text{ext}}=\operatorname{round}(p_{\text{extrap}}\,m),\qquad m_{\text{hole}}=\operatorname{round}(p_{\text{hole}}\,m),\qquad m_{\text{iid}}=m-m_{\text{ext}}-m_{\text{hole}}.
   $$
3. **Draw the region sizes per task**, in rows:
   - tail size $T\sim\text{log-uniform}\,[\,m_{\text{ext}},\ \max(m_{\text{ext}},\,0.2N)\,]$;
   - hole size $k\sim\text{log-uniform}\,[\,m_{\text{hole}},\ \max(m_{\text{hole}},\,0.2N)\,]$.
4. **Generate the X pool.** Run the X-graph once to get
   $$
   P=N+(T-m_{\text{ext}})+(k-m_{\text{hole}})
   $$
   rows.
5. **Draw $y^{(0)}$ for every pool row** by forward-passing the y-graph with fresh private noise. The context's $y^{(0)}$ will be the labels the model sees.
6. **Post-process y exactly as TabICLv2's prior does, fitted once, applied everywhere.** Fit the y transform $g$ on the pool's $y^{(0)}$: robust clamp at mean ± 4σ, then standard scaling (`outlier_removing` → `standard_scaling`, `prior/_graph_scm.py`, `prior/_reg2cls.py`), and apply it to $y^{(0)}$.
7. **Filter the task (§1.4).** If it is rejected, discard everything and restart from step 1 with a fresh task.
8. **Split on X with the mixed rule (§1.1).** This yields exactly $m_{\text{ext}}$ extrapolation queries, $m_{\text{hole}}$ hole queries, $m_{\text{iid}}$ i.i.d. queries, $n_{\text{ctx}}$ context rows, and $P-N$ discarded rows.
9. **Make duplicates (§1.2).** Overwrite a fraction of the query slots with x-values copied from context rows. The sizes do not change.
10. **Draw the replicates.** Query rows get R = 16 i.i.d. draws from the y-graph with fresh private noise, used only to build the targets $\hat t_A,\hat t_E$. The queries’ own $y^{(0)}$ served only $g$ and the previous filtering step, and are not reused as replicates. Apply the same $g$ (same bounds, mean and std) to all R replicates.
11. Run the frozen backbone forward once on the $N$ kept rows, along the forward route of Architecture §1, and the heads.

> **Why this normalization.** It reproduces the backbone's prior and keeps targets in the units the heads predict in (proposal §4.3, "fit/apply").
>
> - **Fit $g$ on the whole pool, before the split.** TabICLv2 fits its y transform on all rows of the dataset, context and queries together. Fitting on the full i.i.d. pool, before any hole, tail or duplicate manipulation, reproduces that exactly, and keeps $g$ independent of our split rule.
> - **Clamp the replicates with the fixed bounds.** TabICLv2's "outlier removal" is a clamp, not a row deletion. It is part of the definition of the task law the backbone was pretrained on. Clamped replicates $g(y^{(r)})$ are i.i.d. draws from the clamped law $g_\#p_\tau$, so $\hat t_A$ is **unbiased** for the aleatoric entropy of that law, which is the law the PPD approximates.
> - **Two rules remain.** Never refit $g$ per replicate (that would make the transform random across draws). Never clip the *targets* $\hat t_A,\hat t_E$ themselves (that biases the estimator, §4.5).
> - **Residual caveat.** At deployment only context labels exist, so any inference-time scaling is context-only. TabICLv2 pretraining carries the same small mismatch, which is negligible for i.i.d. queries. Evaluate T2–T3 with the same pipeline as training.

### Why manipulating the split is allowed

Every split decision depends **only on X** (and on random numbers independent of y). With the y-law generated by its own graph, an X-only selection carries no information about it. So the posterior over the y-law given the manipulated context is unchanged, and the targets keep their meaning as $\mathbb E_{\tau\mid D}[\cdot]$.

Selecting on y, for example dropping context rows with large residuals, would break this. No rule below does that.

### 1.1 Query-type mixture (within every task)

Each task's $m$ queries mix three types. They are built from the pool of $P$ rows in this order, each step removing a region from what is left. **No new rows are created:** every query is an existing pool row, a real draw from the prior. A region's rows are either selected as queries or discarded, so the context never sees that region.

| Type | Default share | Construction (rows left after the step) |
|---|---|---|
| **Extrapolation** | 15% | **1.** Draw a random unit direction $u$ in standardized X space and score all $P$ rows by $s_i=u^\top x_i$.<br>**2.** The $T$ rows with the largest $s_i$ are the **tail**; remove them from the pool.<br>**3.** From those $T$ tail rows, select $m_{\text{ext}}$ uniformly at random as extrapolation queries; discard the other $T-m_{\text{ext}}$.<br>*Left:* $P-T$ rows. |
| **kNN hole** | 15% | **1.** Pick a random centre row among the $P-T$ remaining rows.<br>**2.** The centre and its nearest neighbours, $k$ rows in total (standardized X; a single centre, so one distance computation over the remaining rows, fast even for stage-3 pools), are the **hole**; remove them from the pool.<br>**3.** From those $k$ hole rows, select $m_{\text{hole}}$ uniformly at random as hole queries; discard the other $k-m_{\text{hole}}$.<br>*Left:* $P-T-k$ rows. |
| **i.i.d.** | 70% | **1.** The $P-T-k=N-m_{\text{ext}}-m_{\text{hole}}=n_{\text{ctx}}+m_{\text{iid}}$ remaining rows are an ordinary i.i.d. pool.<br>**2.** Select $m_{\text{iid}}$ of them uniformly at random as i.i.d. queries, exactly TabICLv2's random split.<br>**3.** The rest are the **context**: exactly $n_{\text{ctx}}$ rows.<br>*Left:* nothing. |

**Bookkeeping.** Queries $=m_{\text{ext}}+m_{\text{hole}}+m_{\text{iid}}=m$; context $=n_{\text{ctx}}$; discarded $=(T-m_{\text{ext}})+(k-m_{\text{hole}})=P-N$. The model sees exactly $N=n_{\text{ctx}}+m$ rows.

- **Why randomize T and k separately from the query budgets.** It decouples *how far from support* a query is from *how many* queries there are. Hole and extrapolation queries then cover a range of divergences, not one fixed regime. The upper bounds (0.2N for both the tail and the hole) are defaults and can be changed.
- **Why use counts, not radii.** Defining T and k as row counts keeps region sizes meaningful in any dimension, where distances concentrate as d grows.
- **Cost of the enlarged pool.** At most $0.4N$ extra X rows, each with one extra $y^{(0)}$ draw, which is cheap next to the R replicate passes. In stage 3 ($N$ up to 60K) the pool can reach about 84K rows; lower the region bounds there if X generation or the kNN search becomes the bottleneck.

### 1.2 In-context duplicates

After the split, 5% of query slots (chosen uniformly across all three types) get their x overwritten by the x of a random context row. Only x is copied: the context row’s label is never reused, and the duplicate receives R fresh y draws like any other query.

- This teaches the low-divergence end: at an observed x, E is small but not zero under noise. It mirrors DEUP training the error predictor on in-sample errors and its seen-bit feature (DEUP §3, interactive-learning paragraph).
- The overwrite happens before the replicates are generated, so it costs nothing.
- Because duplicates replace slots, the effective mix is 66.5 / 14.25 / 14.25 / 5.

### 1.3 Mixture as a hyperparameter

```text
query_mix      = [p_iid, p_hole, p_extrap]   # default [0.70, 0.15, 0.15], must sum to 1
dup_fraction   = 0.05                         # share of query slots overwritten with context x (CutGraphConfig)
R              = 16                           # training replicates per query
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
- **Same filter everywhere.** The validation bank and the T2 test bank use the same filter, so every grade is on the backbone's prior.
- **Difficulty-match gate (prior level).** Before training, matched banks of cut and unmodified `graph_scm` tasks are compared (`scripts/00f_difficulty_match.py`); if the cut drifts from `graph_scm`, tune the target graph's private-root noise (`private_root_scale`, `private_root_dims_range`), never the filters. The pass rule, the calibrated configuration and its results are in "The observed-cut prior" above.

## 2. Curriculum, batch, and optimization

| Stage | Steps (default) | Rows N per task | Context fraction | Head max LR | Fork max LR (B) |
|---|---|---|---|---|---|
| 1 | 50,000 | 1,024 | 30–90% | `lr` | `fork_lr` |
| 2 | 4,000 | 400–10,240 (log-uniform) | ≈80% | `lr` / 8 | `fork_lr` / 8 |
| 3 | 1,000 | 400–60,000 (log-uniform) | ≈80% | `lr` / 40 | `fork_lr` / 40 |

- **Step budget.** TabICLv2's 500K : 40K : 10K proportions, scaled by 1/10. The steps per stage are inputs.
- **Learning rate.** `lr` (head) and, in B, `fork_lr` are the swept values (§4), both stage-1 values. Stages 2–3 scale both down by TabICLv2's own ratios (8e-4 → 1e-4 → 2e-5, i.e. /8 and /40), as TabICLv2 scaled its whole model. The ratios are fixed, not swept per stage: a configuration is one stage-1 value per LR, and the same ratios set its stages 2–3. Use a cosine schedule **restarted per stage**, as in TabICLv2's scripts: each stage is a separate launch that loads only the previous stage's weights (`--only_load_model True`, so optimizer and scheduler state reset), then warms up for 1% of its steps to its max LR and decays with a single cosine cycle to 1e-7 (`--scheduler cosine_with_restarts --cosine_num_cycles 1 --warmup_proportion 0.01 --cosine_lr_end 1e-7`, `scripts/train_v2_reg_stage{2,3}.sh`). The fork's schedule (B) ends lower, at **1e-8** in every stage (`--fork_cosine_lr_end 1e-8`): its smallest swept LR reaches 3e-6/40 = 7.5e-8 in stage 3, below the heads' end value of 1e-7 (tabicl's schedule requires the end LR to be below the max LR). The heads keep 1e-7.
- **Optimizer.** AdamW for the heads (all arms) and the fork (B), with weight decay $10^{-5}$, gradient-norm clipping at 5.0 and $\beta=(0.9,0.999)$. In A jobs each head is clipped on its **own** gradient norm (`--clip_per head`): clipping a pair on the joint norm of its two heads would let a large $\hat E_{\text{dir}}$ gradient shrink $\hat A$'s step, coupling heads that share no parameter. These follow DecoupledPFN's defaults (Bergna et al., App. B.3), the closest precedent.
- **Muon is deliberately not inherited from TabICLv2's scripts** (`--muon True` in `scripts/train_v2_reg_stage*.sh`; pass `--muon False` if their flags are reused). Muon helped TabICLv2's pretraining of the full model from scratch (TabICLv2 §4.1, ablation Fig. C.1). It is a pretraining tool, not one for fine-tuning or for training small decoders like the A1/A2 trunks.

> **Conditional ablation: Muon for the fork.** The fork's blocks are copies of weights that were pretrained with Muon. If arm B wins the ladder comparison (§4, procedure step 2–3), run one extra configuration of B's winner: Muon on the fork's 2-D matrices only, AdamW for the heads and all 1-D parameters, same seeds. Its LR grid must be set on Muon's scale. Not run otherwise.
- **Batch.** 32 tasks per step, half of TabICLv2's 64, with the step budget unchanged. This halves the wall time per step: stage 1 is bound by the CPU workers that generate the tasks, not by the GPU (step-4 throughput gate, protocol §3), and stages 2–3 by the backbone forward, both proportional to the tasks per step. The cost is half as many training tasks per run and noisier gradients per step. Micro-batch groups keep TabICLv2's sizes (4 tasks in stage 1, so 8 groups per step; 1 task in stages 2–3, so 32). The backbone runs frozen under `torch.no_grad()`, so no backbone activations are stored. Stage 1 (1,024 rows) fits comfortably on mid-range GPUs. In stages 2–3 (up to 60K rows), reach 32 by **gradient accumulation over micro-batches** (TabICLv2's own mechanism, `micro_batch_size` in `_run.py`).
- **Arm B.** It backpropagates through the k = 2 forked blocks, so it needs gradient checkpointing above about 20K rows (TabICLv2 App. B.2 does the same).
- **Feature counts** follow TabICLv2's prior settings (random per task, up to 100 features).

## 3. Validation

### The bank

- **2,000 tasks**, pre-generated once with a fixed seed.
- The same pipeline, query mixture (including duplicates), and n/d ranges as training.
- **R_val = 128** replicates per query.
- Seeds **disjoint from the T2 test bank** (and from T1, T3, T4). Right now, no seed has been chosen for the T2 test bank, so we can choose one seed randomly.
- **Stratify the bank across stages.** Following the step proportions would give about 91% of 1,024-row tasks, so large-n bins would be nearly empty. To resemble typical dataset size distribution, the bank is split by rows N:

  | Bin | Rows N | Share | Tasks |
  |---|---|---|---|
  | Small | 1,024 (context fraction 30–90%) | 50% | 1,000 |
  | Medium | 1,024–10,240 (log-uniform, ≈80%) | 30% | 600 |
  | Large | 10,240–60,000 (log-uniform, ≈80%) | 20% | 400 |

  ScoringBench subsamples every dataset to at most 3,000 rows (2,400 train + 600 test; ScoringBench App. C.2), and about 70% of TabArena's 51 datasets have fewer than 10K rows (15 have 10K–100K; TabPFN-3 §3.1.1).

> **What the bank showed** (built bank, `VAL_BANK_SEED`; checks of protocol §3 step 6, all passed).
> - **Extrapolation queries carry more epistemic uncertainty than hole queries.** The ordering duplicate < iid < hole, extrapolation held as expected, but hole and extrapolation were not similar: mean $\hat t_E$ is 0.036 (iid), 0.064 (hole) and 0.088 (extrapolation), about 27% higher for extrapolation, and consistently so in every N bin (task-level bootstrap 95% CIs of extrapolation − hole exclude 0 in the small, medium and large bins). Predicting beyond the support of the context is harder than filling a gap inside it. Rules of thumb that expect hole ≈ extrapolation (protocol §8.2) should be read with this in mind.
> - **The PPD's own uncertainty tracks the aleatoric target, not the epistemic one** (a result for the paper). Test: the **e(q) proxy** reference line (protocol §8.1) is the best predictor of a target that sees only the backbone's CRPS entropy $e(q_\theta)$: 50 equal-count bins of $e(q_\theta)$, each predicting its bin's mean target, fitted on half of the bank's tasks and scored on the other half (cross-fitted on even/odd tasks, so no query is scored by bins fitted on its own task). Its **skill** is $1-\text{xMSE}_{\text{proxy}}/\text{xMSE}_{\text{const}}$, the share of the target's variance (beyond the constant baseline) that $e(q_\theta)$ explains, with replicate noise removed by the xMSE. On the pooled bank: **skill 0.61 for $\hat t_A$** (xMSE 0.0132 vs 0.0342 for the constant) and **skill 0.12 for $\hat t_E$** (xMSE 0.0321 vs 0.0364). The noise floors (3.3e-4 for $\hat t_A$, 4.1e-5 for $\hat t_E$) are tiny next to these, so the gap is not target noise. So the backbone's total predictive spread is largely a measure of aleatoric uncertainty and says little about how far its forecast is from the task's true law: the epistemic part must be extracted by a dedicated head, which is the thesis of this project (proposal §1.1). For the paper: report both skills as the baseline the Ê head must beat, and as direct evidence that PPD spread is not an epistemic signal.

### What is logged every N steps

- **Default cadence:** every 5,000 steps in stage 1 (10 evaluations, the last at the stage end), and at the end of stages 2 and 3. A full bank evaluation costs about half an hour of GPU time (dominated by the 400 large tasks; A2's per-layer taps and B's fork cannot be cached), so this keeps validation to roughly 5 h per run. The best-$\hat A$ / best-$\hat E_{\text{dir}}$ checkpoints are therefore chosen at 5,000-step resolution in stage 1. A 64-task mini-bank is evaluated every 250 steps for a finer, cheap curve (protocol §7.4); it is not used for checkpoint selection.
- **Metrics:** unweighted MSE of $\hat A$ against $\hat t_A$, and of $\hat E_{\text{dir}}$ against $\hat t_E$, logged **separately**. What happens in the low-uncertainty regime is read from the per-type and per-bin breakdowns.
- **Breakdowns:** per query type (i.i.d., hole, extrapolation, duplicate), for the total mix, and per n bin (≤512, 512–1K, 1K–4K, 4K–16K, 16K–60K).
- **Reading the curves.** The target-noise term of the MSE is identical for every checkpoint and run, so differences between runs are differences in true error. Because the bank is fixed, every comparison is paired (same tasks), which removes task-to-task noise.

> **Two levels of scoring.**
> - *Within an arm: per head.* Every run reports $\text{MSE}_A$ and $\text{MSE}_E$ separately, and configurations are chosen **per head**. In A1 and A2 the heads share no parameters, so the best $\hat A$ head and the best $\hat E_{\text{dir}}$ head may come from different configurations (and are combined into one model). In B the heads share the fork, so a configuration is chosen as a whole and heads cannot be mixed.
> - *Between arms: mixed score.* To choose the best arm, A1's best head combination, A2's best head combination and B's best configuration are compared on
> $$\text{score}=\frac{\text{MSE}_A}{\text{MSE}_A^{\text{ref}}}+\frac{\text{MSE}_E}{\text{MSE}_E^{\text{ref}}},$$
> where $\text{MSE}_A^{\text{ref}}$ and $\text{MSE}_E^{\text{ref}}$ are the seed means of the default A1 configuration (head LR $3\times10^{-4}$). Each term is that head's error relative to the reference, so a score below 2 means better than default A1 overall. This score is **only** used to choose the arm, never to choose a configuration inside an arm.

### No stopping rule, best-per-head checkpoints

There is **no early stopping**, as in TabICLv2. Each run trains to its fixed budget. The validation curve serves to:

- check that training improves, and
- compare runs and checkpoints: hyperparameters, arms, and seeds (§4).

**Checkpoints kept per run, for all arms** (updated at every validation):

- **latest:** overwritten each time;
- **best $\hat A$ so far** and **best $\hat E_{\text{dir}}$ so far:** each overwritten only when that head's validation MSE improves.

A run's $\hat A$ head is its best-$\hat A$ checkpoint, and its $\hat E_{\text{dir}}$ head its best-$\hat E_{\text{dir}}$ checkpoint. This is checkpoint selection, not early stopping: training always runs to the end.

> **Comment on B.** In A1/A2 the two heads are independent, so selecting each one at its own best checkpoint is exact. In B the heads share the fork, so the best-$\hat A$ and best-$\hat E_{\text{dir}}$ checkpoints are two different states of the fork, and per-head selection may not be the best criterion: the fork can become very good for one head at the cost of the other. Log both heads at both checkpoints so this trade-off is visible.

## 4. Sweeps, seeds, and selection

### Sweep grid

| Arm | Head LR | Fork LR | k | p | Configurations |
|---|---|---|---|---|---|
| A1 | {1e-4, 3e-4, 1e-3} | — | — | 0 (fixed) | 3 |
| A2 | {1e-4, 3e-4, 1e-3} | — | — | 0 (fixed) | 3 |
| B | {1e-4, 3e-4, 1e-3} | {3e-6, 1e-5, 3e-5} | 2 | 0 (fixed) | 9 |

There is no pull-to-pretrained penalty in B. The loss weight is fixed at $p=0$ in every run; $p=2$ is a fallback ablation, run only if T2 shows the low-uncertainty bins badly calibrated after all tests (architecture section, §3).

### Procedure

Every configuration of every arm uses the **same seeds**: seed $s$ fixes both the task stream and the head initialization, so all comparisons are paired.

1. **Round 1.**
   - **A1/A2:** all 6 configurations with seeds {0, 1, 2} (3 jobs, §5). For each arm, pick the best $\hat A$ configuration by seed-mean $\text{MSE}_A$ and the best $\hat E_{\text{dir}}$ configuration by seed-mean $\text{MSE}_E$; together they form the arm's best head combination.
   - **B:** the 9 configurations on seed 0 only. Pick the best configuration as a whole.
2. **Compare arms on seed 0** with the mixed score (§3): A1's and A2's best head combinations against B's best configuration, all on the same seed-0 task stream.
3. **Round 2 (only if B clearly improves on A1 and A2).** Run B's winning configuration on seeds 1 and 2, and compare the arms again on the **3-seed mean** of the mixed score.
4. **Pick the checkpoint**: within the winning arm, the best-validation seed (per head for A1/A2).

> **Comments.**
> - *Arm choice vs reporting.* Picking the arm by seed mean is the right call. Picking the best seed inside it is a mild, acceptable optimism because it uses validation, not test. Report the T1–T4 results for that checkpoint **and** the 3-seed mean ± std, so the reader can see the spread.
> - *What "seed" means.* A seed should change **both** the head initialization and the task stream (§5 explains why this matters for the shared jobs).
> - *Why two rounds for B.* B is the only arm that needs one job per configuration. Running its 9 configurations on seed 0 in parallel with the A jobs avoids waiting for A1/A2's results, and the extra seeds are spent only if B is worth them.

## 5. Job layout and the shared-forward saving

### The saving

In A1 and A2 nothing upstream of the heads trains. For a given batch, the backbone forward (TFcol → TFrow → TFicl → LN → decoder → $q_\theta$, $e(q_\theta)$) produces **the same tensors no matter which head reads them**. That forward is almost all of the compute per step. A head pair is about 1M parameters, a negligible cost next to a 12-block transformer over thousands of rows.

So **one job can run the backbone once per step and feed many independent heads**:

```python
for step in range(total_steps):
    batch = sample_tasks()                                        # X → split → duplicates → y (§1)
    with torch.no_grad():
        resid = backbone.tf_icl_all_layers(batch)                 # r_1 … r_12, all rows
        h_star = backbone.ln(resid[-1])                           # what the decoder sees
        q = backbone.decoder(h_star); e_q = crps_entropy(q)
        tA, tE = targets(q, batch.y_reps)                         # R replicates
    for head in heads:                                            # every A1 and A2 config
        inp = h_star if head.arm == "A1" else head.layer_mix(resid)
        loss = head.loss(inp, tA, tE)                             # each with its own LR (p = 0, init from the job's seed)
        loss.backward(); head.opt.step(); head.opt.zero_grad()
```

- Each head has its **own optimizer and LR schedule**, so the heads stay fully independent: separate parameters, separate gradients.
- A2 additionally needs the 12 residual-stream vectors per query. Tapping all 12 layers once serves every A2 head.

### Layout

| Round | Job | Contents | Count |
|---|---|---|---|
| 1 | **A, seed $s$** | all 3 A1 + 3 A2 configurations (6 head pairs) on one backbone stream, task stream and initialization from seed $s\in\{0,1,2\}$ | 3 |
| 1 | **B, seed 0** | one fork per job, since each fork changes the forward pass | 9 |
| 2 | **B, seeds 1–2** | B's winning configuration, only if B clearly improves on A1/A2 | 2 |

Each A job varies both initialization and data, exactly like B's seeds. A1 and A2 share a job because A1's input $h_*$ is a by-product of the same pass that taps the 12 layers, and B job $s$ uses the same task stream as A job $s$, so arm comparisons are paired too. The sweep and seed stages for A1/A2 collapse into one round: all configurations × seeds train at once.

## 6. Defaults at a glance

| Setting | Default |
|---|---|
| Query mix (i.i.d. / hole / extrapolation) | 0.70 / 0.15 / 0.15 |
| Duplicate fraction | 0.05 of query slots |
| Hole | random centre, random k per task |
| Extrapolation tail ρ | random per task, ≥ query budget |
| R (train) / R_val | 16 / 128 |
| Steps per stage | 50K / 4K / 1K (TabICLv2 proportions ÷ 10) |
| Batch | 32 tasks (half of TabICLv2's 64), with micro-batch accumulation in stages 2–3 |
| Head LR (stage 1); stages 2/3 | swept {1e-4, 3e-4, 1e-3}; ×1/8, ×1/40 |
| Fork LR (B, stage 1); stages 2/3 | swept {3e-6, 1e-5, 3e-5}; ×1/8, ×1/40 |
| Cosine end LR | heads 1e-7; fork (B) 1e-8, every stage |
| Fork depth k (B) | 2 |
| Optimizer | AdamW (heads and fork), weight decay 1e-5, β = (0.9, 0.999); Muon only as a conditional fork ablation if B wins |
| Gradient clipping | 5.0 (per head in A jobs; fork + heads jointly in B) |
| Forward route | one raw training-mode forward (`_icl_predictions`), no `TabICLRegressor` ensemble, everywhere: training, validation, Phase 0 |
| Loss weight exponent | p = 0 fixed; p = 2 only as a fallback ablation |
| Validation | 2,000 tasks (50 / 30 / 20 by N), unweighted MSE per head, every 5,000 steps + end of each stage; mini-bank (64 small tasks) every 250 steps |
| Stopping | none; keep latest + best-$\hat A$ + best-$\hat E_{\text{dir}}$ checkpoints per run |
| Selection | per head within an arm (B: whole configuration) → best arm by mixed score vs default A1 → best seed |
| Task filtering | graph check (on the y-graph), constant columns, ExtraTrees predictability test (p < 0.05); trivial-task test off |

