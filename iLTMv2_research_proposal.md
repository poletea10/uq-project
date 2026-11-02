# Score-Based Uncertainty Decomposition for Tabular Foundation Models

**Research proposal — iLTMv2 / TFM final thesis** · Pol Gàlvez Casasús · revised 29 October 2026

> **Legend.** **TBD** marks a part whose design has not been decided yet. Notes marked *(previous draft)* record what the earlier proposal suggested, for reference only; they are not decisions.

---

## 1. The idea

### 1.1 The problem in one paragraph

A Tabular Foundation Model is a Prior-Data Fitted Network (PFN) specialised to tables: a transformer pretrained on synthetic datasets that, given a labelled table $D$ and a query $x_*$, outputs the posterior predictive distribution (PPD) $q_\theta(y \mid x_*, D)$ in one forward pass (Müller et al., 2021; Qu et al., 2026). The PPD mixes two different kinds of uncertainty: not knowing *which task* generated $D$ (**epistemic**, reducible with more or better context) and the noise *inside* that task (**aleatoric**, irreducible). They call for different actions — collect more data versus accept the limit or change the measurement — and a single total-uncertainty number hides which one is being faced. The split cannot be recovered from the PPD alone (Bergna et al., 2026), but whoever owns the synthetic generator can supervise it directly.

### 1.2 Objectives

1. **Decompose** predictive uncertainty into aleatoric and context-epistemic components, for an interpretable UQ.
2. **Use** the decoupled components to improve predictive performance (MAE, Elo score and/or proper scoring rules).

Both start from **TabICLv2** (Qu et al., 2026): a PFN with a 999-quantile regression head trained with pinball loss, pretrained only on the graph-based SCM prior (`graph_scm`). **Scope: regression scored with CRPS.** The classification instantiation (Brier score) is stated where it differs but is out of scope for the thesis timeline.

### 1.3 How

```
                 ┌──► PPD q_θ           (unchanged TabICLv2 quantile head)
 table D, x* ──► TabICLv2 ──► Â         (aleatoric entropy head)        Phase 1
                 └──► Ê_dir             (epistemic / divergence head)
                                   │
                                   ▼
             pick the context rows that most reduce Ê_dir(x*)           Phase 2
             (instead of the k nearest neighbours)
```

### 1.4 Contributions

| # | Contribution | Phase |
|---|---|---|
| C1 | **Direct Epistemic Uncertainty Prediction (DEUP) for In-Context Learning.** A repurposing of Lahlou et al.’s (2021) method adapted for Tabular Foundation Models, where In-Context Learning is used (i.e. the model works on a prior over tasks and there is no per-dataset training needed). | 1 |
| C2 | **Scoring-rule-general supervision.** A supervised uncertainty decomposition for TFMs that does not change the PPD’s head performance and is not restricted to end-node Gaussian noise, unlike Bergna et al.’s (2026) approach. | 1 |
| C3 | **Exact conditional replicates from the graph prior.** They come from an “observed cut” of TabICLv2’s own prior, which makes the supervision targets exactly unbiased. | 1 |
| C4 | **Epistemic-guided context retrieval.** Selecting context by informativeness for the query rather than by feature-space proximity (cf. Thomas et al., 2024; Ma et al., 2024) | 2 |

### 1.5 Phase map

| Phase | Question | Output |
|---|---|---|
| **0 — Setting the baseline** | Which decomposition do we use, and how well do existing methods recover it against known ground truth? | Baseline test results |
| **1 — Model training** | Can TabICLv2 learn $\hat A$ and $\hat E_{\text{dir}}$ from generator supervision? | iLTMv2 with two UQ heads, evaluated on all Phase 0 tests |
| **2 — Epistemic-guided context** | Does selecting context by $\hat E_{\text{dir}}$ beat k-NN retrieval? | Retrieval method + ablation + benchmark results |

---

## 2. The decomposition (shared by all phases)

### 2.1 Entropy and divergence of a proper score

Let $S(F, y)$ be a proper scoring rule in loss orientation (lower is better; Gneiting & Raftery, 2007). Following Hofman et al. (2024):

$$
e(G) = \mathbb{E}_{Y\sim G}\,[S(G,Y)] \qquad\qquad d_S(F,G) = \mathbb{E}_{Y\sim G}\,[S(F,Y)] - e(G) \;\ge 0
$$

- **Entropy** $e(G)$: the score you still pay *even when your forecast is exactly the truth* $G$ — the irreducible floor.
- **Divergence** $d_S(F,G)$: the *extra* score you pay for forecasting $F$ when the truth is $G$.

### 2.2 Splitting a forecast's expected score

For a query $x_*$ and context $D$, let $\tau$ be the task (everything the generator sampled) and $p_\tau(\cdot\mid x_*)$ its true conditional law. For **any** forecast $q$, the expected score splits exactly into

$$
\underbrace{\mathbb{E}_{\tau\mid D}\,\mathbb{E}_{Y\sim p_\tau}[S(q,Y)]}_{\text{expected score of } q}
\;=\;
\underbrace{\mathbb{E}_{\tau\mid D}\big[e(p_\tau)\big]}_{A:\ \text{aleatoric (irreducible risk)}}
\;+\;
\underbrace{\mathbb{E}_{\tau\mid D}\big[d_S(q,p_\tau)\big]}_{E(q):\ \text{epistemic (excess risk)}} .
$$

- $A$ is the score we would get if we knew the true task. It does not depend on the predictor.
- $E(q)$ is what we pay on top because we do not know the task. It **is** predictor-dependent.
- If $q$ is the exact Bayesian PPD, the left side equals $e(q)$, recovering the familiar form *total = aleatoric + epistemic*. Non-negativity of $E$ follows from concavity of $e$; no Gaussian assumption is needed anywhere.

**In plain words.** Picture the PPD as a weighted mix of task laws. Aleatoric is the spread of the true task law; epistemic is the extra cost of not knowing which law in the mix is the true one.

### 2.3 Instantiations

| Score | Entropy $e(p)$ | Divergence $d_S(p', p)$ |
|---|---|---|
| Log (class.) | $-\sum_k p_k\log p_k$ | $\mathrm{KL}(p\Vert p')$; epistemic term $= I(Y_*;\tau\mid x_*,D)$ |
| Brier (class.) | $1-\sum_k p_k^2$ (Gini impurity) | $\sum_k (p_k-p'_k)^2$ |
| **CRPS (regr., primary)** | $\tfrac12\,\mathbb{E}_F\lvert X-X'\rvert$ | $\mathbb{E}\lvert X-Y\rvert-\tfrac12\mathbb{E}\lvert Y-Y'\rvert-\tfrac12\mathbb{E}\lvert X-X'\rvert$, $X\sim F,\ Y\sim G$ |

**Why CRPS fits TabICLv2.** Substituting $y=Q(\alpha)$ gives $e_{\mathrm{CRPS}}(F)=\int_0^1\alpha(1-\alpha)\,dQ(\alpha)$, a closed-form $O(K)$ sum over the 999 sorted quantiles. So $e(q_\theta)$ is computed directly from the existing head, with no binned output and no convolution. Useful special case: $e_{\mathrm{CRPS}}(\mathcal N(\mu,\sigma^2))=\sigma/\sqrt\pi$.

### 2.4 Unbiased estimators from replicate draws

Given $R\ge2$ i.i.d. draws $y^{(1)},\dots,y^{(R)}$ from $p_\tau(\cdot\mid x_*)$ (how to obtain them: §4.3):

| | Entropy | Unbiased estimator $\hat t_A$ |
|---|---|---|
| CRPS | $\tfrac12\mathbb{E}\lvert Y-Y'\rvert$ | $\dfrac{1}{R(R-1)}\displaystyle\sum_{r<s}\lvert y^{(r)}-y^{(s)}\rvert$ |
| Brier | $1-\sum_k p_k^2$ | $\dfrac{1}{R(R-1)}\displaystyle\sum_{r\neq s}\mathbb 1\{y^{(r)}\neq y^{(s)}\}$ |

The expected score of a forecast is estimated by the plain Monte Carlo average, so the excess-risk target is

$$
\hat t_E \;=\; \frac1R\sum_r S\big(q,\,y^{(r)}\big) \;-\; \hat t_A .
$$

With $R=2$ and CRPS, $\hat t_A=\tfrac12\lvert y^{(1)}-y^{(2)}\rvert$. The log score has no unbiased entropy estimator, which is one more reason to work with CRPS (or Brier).

**Why noisy targets are fine.** A squared-error loss converges to the *conditional mean* of its target. If the target is unbiased, the minimiser is exactly the quantity of interest ($A$ or $E$) however noisy each single target is. More replicates only reduce variance and speed up convergence; they do not move the optimum.

---

## 3. Phase 0 — Setting the baseline

### 3.1 What this phase does

1. Fix the decomposition used throughout the project (§2).
2. **Show** where each model puts aleatoric and epistemic uncertainty, on a 1-D toy set built for the paper's main figure.
3. **Grade** the decomposition against generator-computed ground truth, in distribution (cut-graph test) and out of it (transfer test).
4. **Check usefulness** with the active-learning benchmark of Bergna et al. (2026).
5. Run the baselines on all of the above, for later comparison with iLTMv2.

Grading asks *"is the number right?"*; active learning asks *"does the number lead to good decisions?"*. The literature has so far only answered the second (Bergna et al., 2026; Fortini et al., 2026).

| # | Test | Ground truth | Models | Output |
|---|---|---|---|---|
| T1 | Toy set | Closed form | Heteroscedastic GP, Dec-ICL, iLTMv2 | Figures |
| T2 | Cut-graph test (in distribution) | Exact replicates | Dec-ICL, iLTMv2 | Reliability plots + table |
| T3 | Transfer test (full `graph_scm`) | Exact-match replicates | Dec-ICL, iLTMv2 | Reliability plots + table |
| T4 | Active learning | Test-set performance | Random, TabICLv2, iLTMv2, Tuned-ICL, Dec-ICL | Table + learning curves |

Phase 0 trains nothing; iLTMv2 rows stay visibly empty until Phase 1 delivers the model. The row set and the toy function are fixed before the first run. The execution details (constants, seeds, figure layouts, adapters) live in `phase0_protocol.md`.

### 3.2 What is graded

Every model outputs a predictive distribution $q_m$, as a set of members or as a CDF. From it we read

$$
\hat A = \text{mean member entropy}, \qquad \hat E = e(q_m) - \hat A ,
$$

and grade:

- $\hat A$ against the **true aleatoric entropy** $A=e\big(p_\tau(\cdot\mid x_*)\big)$ of the realized task.
- $\hat E$ against the **realized excess** $E(q_m)=\mathbb{E}_{Y\sim p_\tau}[S(q_m,Y)]-A$: the CRPS that the model's *own* forecast actually pays above the irreducible floor.

The epistemic ground truth is therefore **model-specific** (§2.2): each model is graded on whether it knows its own error, not on distance from a shared Bayes reference. iLTMv2's $\hat E_{\text{dir}}$ is trained to target exactly $E(q_\theta)$ (§4.4).

**What $e_{\text{tot}}-\hat A$ measures.** For the GP and Dec-ICL, $\hat E=e_{\text{tot}}-\hat A$ is exactly the excess score the model *expects* to pay if the truth is one of its own members (a self-predicted excess, non-negative by concavity of $e$). It equals the realized $E(q_m)$ only on average over the tasks consistent with the context, and only if its members are the exact posterior over tasks. A misspecified model misses the bias (DEUP's argument; Lahlou et al., 2023). The gap between self-predicted and realized excess is what the tests measure.

**The standardisation layer.** Every model reduces to a weighted family of predictive CDFs $\{(w_m,F_m)\}$:

$$
\hat A=\sum_m w_m\,e(F_m), \qquad e_{\text{tot}}=e\Big(\sum_m w_mF_m\Big), \qquad \hat E=e_{\text{tot}}-\hat A\ \ge 0 .
$$

One entropy kernel, $e_{\mathrm{CRPS}}(F)=\int F(1-F)\,dt$, serves every model. **Project member-wise:** never convert an already-aggregated variance split into entropy units, since $\mathrm{Var}=\mathbb E[\sigma^2]+\mathrm{Var}[\mu]$ is additive in variance and $\sigma/\sqrt\pi$ is not. The truth code and every adapter's conversion are unit-tested against closed forms before any model is run.

**Grading rule (T2, T3).** Regress realized truth on the estimate, never the reverse: because tasks are drawn from a fixed prior, truth-on-estimate gives the same answer as if the posterior-averaged truth were known (so no costly oracle posterior is needed). Headline metric: the slope (ideal 1), with intercept and mean signed error, each with a 95 % bootstrap CI over whole tasks, plus binned reliability plots coloured by context size $n\in\{16,64,256,1024\}$ ($E$ on log–log axes). $R^2$ and rank correlation are not reported: against realized truth they measure the testbed's noise, not the method.

### 3.3 T1 — Toy set

One **fixed, curated** 1-D function on $[-5,5]$, one fixed context draw, three noise settings:

$$
y = f(x) + \sigma(x)\,\varepsilon,\qquad \varepsilon\sim\mathcal N(0,1).
$$

- **The function.** $f$ is a low-rank GP (28 Gaussian bumps, width 0.3, unit prior s.d.; Rasmussen & Williams, 2006) plus a **gap part**: narrow bumps (width 0.1, s.d. 1.25) that exist only inside the sparse strip $[1,4]$. The dense regions carry no hint of the gap's higher frequency, so a model that infers smoothness from the data it sees is confidently wrong in the gap: posterior variance misses misspecification bias, while excess risk includes it (Lahlou et al., 2023).
- **Curation.** The seed is the first one meeting three pre-registered conditions (bounded amplitude, visible gap structure, non-flat noise strip), chosen on $f$ alone without looking at any model output, then frozen.
- **Context and queries.** The context density is 13× lower inside $[1,4]$ (about 32 gap points at $n=1024$); contexts are nested over $n\in\{16,64,256,1024\}$; 200 evenly spaced queries. All rungs share $f$, the context and the noise draws.
- **Truth** is closed form and exact per query: $A=\sigma(x_*)/\sqrt\pi$, $E(q_m)=\int(F_m-\Phi)^2dt$.

| Toy rung | Noise | Question it asks |
|---|---|---|
| 0 | $\sigma\equiv0$ | Does $\hat E$ rise in the sparse strip with $\hat A\approx0$? |
| 1 | $\sigma\equiv\sigma_0\approx0.25$ | Does gap detection survive noisy surroundings? |
| 2 | $\sigma_0$ outside, $\approx6\sigma_0$ on the *noise strip* $[-3.5,-1.5]$ | Can it *attribute* uncertainty correctly when noise and data density vary independently? |

**What to expect.** With noise, the true gap bump in $E$ is smaller than in rung 0 for every model, and in rung 2 $E$ also rises in the noise strip. So each rung is read as "does $\hat E$ follow $E(q_m)$?", never as "does $\hat E$ match rung 0?". The rung-2 test is whether $\hat A$ rises only in the noise strip and $\hat E$ is higher in the gap than in the noise strip.

**Figures.** One 3×4 figure per (model, toy rung): 9 figures (rows: context and forecast; $A$ vs $\hat A$ with total $e(q_m)$; realized $E(q_m)$ vs $\hat E$ on a log axis; columns: $n$). Plus one 3×3 **paper figure** (rung 2, $n=1024$, one column per model). It is an illustration for the paper, not a grade: one function cannot support calibration claims. Dec-ICL's training prior is close to this toy, so T1 tilts slightly towards Dec-ICL; T2 tilts towards iLTMv2. We state both.

**Backbone check (runs first).** TabICLv2 is run on all rungs and $n$ to confirm the backbone behaves sensibly on this input, with pass criteria fixed beforehand: (1) CRPS outside both strips at most $1.5\times$ the heteroscedastic GP's; (2) on rung 0, total $e(q_\theta)$ in the sparse strip at least $2\times$ that outside; (3) on rung 2, total $e(q_\theta)$ in the noise strip at least $2\times$ that outside. It doubles as the motivating figure: the single total rises in both strips and says nothing about which is which.

> **Toy function $f$ — decided.** Option (a), the curated low-rank GP with extra high-frequency structure in the sparse strip, in the spirit of DEUP (Lahlou et al., 2023, Fig. 1). **Fallback if the backbone check fails:** option (b), a 1-D draw from the cut-graph prior (the target branch with its noise frozen) plus our own Gaussian noise, curated for at least as much variation inside $[1,4]$ as outside, so that $f$ comes from a function class iLTMv2's backbone has seen. It loses the controlled "harder in the gap" structure.

> **Note — optional appendix.** Seed-averaged reliability plots over a distribution of toy functions (the previous protocol's rungs 0–3), cheap to add since the code exists.

### 3.4 T2 — Cut-graph test (in distribution)

Tasks come from the **observed-cut prior** of Phase 1 (§4.3): multi-dimensional, non-Gaussian, heteroscedastic. There is no closed form, but exact replicates of $y$ at fixed $x_*$ give unbiased estimates of $A$ and $E(q_m)$ (§2.4). Provisionally 50 tasks × 5 task seeds (fresh rows, context and hole per seed), 200 queries per context.

- **Context holes.** To test scarce-context behaviour without leaving the prior, a kNN ball around a random anchor (10 % of the pool) is removed from the context, and half the queries are drawn from inside it. Hole queries are real draws from the prior, so "sparse context" is not confounded with "out-of-distribution $x$". The truth decides whether the hole matters for $y$, so a model that merely flags feature-space isolation is graded as wrong.
- **Two levels.** Results are reported across tasks (the grade) and within a task across seeds (descriptive), since even an ideal estimator is only guaranteed to be calibrated on average over tasks.
- **It favours iLTMv2,** because it is iLTMv2's training prior: a tailored in-distribution test, not a fair comparison. It must be rebuilt for each new model prior if the process is repeated. Dec-ICL, which assumes Gaussian end-node noise, is graded as published; part of any gap is that misspecification.

Building T2 is also the first working implementation of the Phase 1 prior.

> **TBD:** generator settings for T2 (number and type of features, node counts). The rejection filter is the one of §4.3.

### 3.5 T3 — Transfer test

Does the decomposition stay right on the **unmodified, fully confounded** `graph_scm`, which the observed cut gives up? Every feature is restricted to be **categorical with low cardinality $k$**, $d$ features per task. All pool rows sharing a query's exact feature vector are then exact i.i.d. draws from $p_\tau(y\mid x_*)$: no distance threshold, no bandwidth. Replicate rows are never context rows.

- **Held-out cells.** 10 % of the eligible feature cells (at least 30 spare rows each) are kept out of the context, the categorical analogue of T2's holes.
- Provisional sweep $k\in\{2,3,5\}$, $d\in\{2,3,4\}$ with $k^d\le256$; graded exactly as T2.
- **Limitation:** continuous features are covered only by extrapolation from this regime.

This doubles as Phase 1's transfer audit (§4.3).

> **TBD:** how to force `graph_scm`'s converters to emit only $k$-level categorical features without otherwise changing the prior, and whether mixed categorical/continuous tasks can be included (they cannot use exact match).

### 3.6 T4 — Active learning

We reuse the active-learning benchmark of Bergna et al. (2026). It does not check whether the decomposition is *correct*, only whether the epistemic part *helps decisions*, which is what Phase 2 relies on. Five public tabular regression datasets (California Housing, Kin8nm, NYC Taxi, Airlines, YearPredictionMSD); initial labelled set $n_0=1024$, unlabelled pool of 5000, test set of 3000, 16–20 seeds with splits shared across methods; acquire 1024 points one at a time, with no training (context retrieval only). Metrics: RMSE, MAE, Gaussian NLL and CRPS on the test set, plus learning curves.

| Method | Acquisition score |
|---|---|
| Random (TabICLv2 predictor) | uniform |
| TabICLv2 | total entropy $e(q_\theta)$ |
| iLTMv2 | $\hat E_{\text{dir}}$ |
| Tuned-ICL | total variance (as in the paper) |
| Dec-ICL | latent / epistemic variance (as in the paper) |

TabICLv2 vs iLTMv2 is our matched pair; Tuned-ICL vs Dec-ICL is theirs. Tuned-ICL and Dec-ICL run on our splits if the authors share the checkpoints, otherwise their reported numbers are used as context only (they were evaluated on unreleased splits, so there is no exact reproduction target).

> **TBD — dataset sources.** Exact OpenML IDs and versions (the paper does not give them).

> **TBD — HPO/BO tests.** Left out for now: active learning plus ground-truth grading are the most informative tests for Phase 2.

### 3.7 Models and their converters

| Model | How it runs | $\hat A$ | Total $e_{\text{tot}}$ |
|---|---|---|---|
| **Heteroscedastic GP** (PyMC) | Two latent GPs ($\mu(x)$ and $\log\sigma(x)$, the latter with a learned constant mean), sparse GP (40 inducing points) + NUTS per dataset; $S$ posterior draws | $\frac1S\sum_s\sigma_s/\sqrt\pi$ | entropy of the $S$-Gaussian mixture |
| **Dec-ICL** (Bergna et al., 2026) | Checkpoint requested from the authors; latent-signal head (bins over $f$) + aleatoric variance $\hat\sigma^2$, convolved | $\hat\sigma/\sqrt\pi$ | $e_{\mathrm{CRPS}}$ of their observation-level predictive |
| **TabICLv2** | Released checkpoint | — | $e(q_\theta)$ from the 999 quantiles only |
| **iLTMv2** | This thesis (Phase 1) | head $\hat A$ | $e(q_\theta)$; epistemic from head $\hat E_{\text{dir}}$ |

In all rows $\hat E=e_{\text{tot}}-\hat A$, except iLTMv2, which predicts its epistemic term directly (§4.4). Dec-ICL is fine-tuned from TabICL (Qu et al., 2025) on the authors' own heteroscedastic prior, so any Dec-ICL vs iLTMv2 gap mixes method, backbone and prior; the thesis says so.

**Which model goes in which test:**

| Model | T1 Toy | T2 Cut-graph | T3 Transfer | T4 AL |
|---|---|---|---|---|
| Heteroscedastic GP | ✓ | — (1-D, too costly) | — | — |
| Dec-ICL | ✓ | ✓ | ✓ | ✓ (or reported numbers) |
| Tuned-ICL | — | — | — | ✓ (or reported numbers) |
| TabICLv2 | backbone check | — | — | ✓ |
| iLTMv2 | ✓ | ✓ | ✓ | ✓ |
| Random (TabICLv2 predictor) | — | — | — | ✓ |

### 3.8 Deliverables

Toy-set figures for the paper (with the TabICLv2 backbone check as motivating figure); T2 and T3 grading of Dec-ICL against generator-computed ground truth (later iLTMv2); active-learning results; and the reusable testbeds and standardisation layer that Phase 1 is evaluated on.

---

## 4. Phase 1 — Model training

### 4.1 What this phase does

1. Establish the **loss function** and **prior generator** that make the decomposition trainable.
2. Design the **architecture** of the $\hat A$ and $\hat E_{\text{dir}}$ heads.
3. **Train** both heads on TabICLv2.
4. **Evaluate** iLTMv2 on every Phase 0 test.

The full design lives in `phase1_design.md`.

### 4.2 Supervision targets

Both targets come from §2.4 and need only $R\ge2$ i.i.d. draws of $y$ at each query from the true task law (training uses $R=16$):

$$
\hat t_A=\frac{1}{R(R-1)}\sum_{r<s}\lvert y^{(r)}-y^{(s)}\rvert, \qquad
\hat t_E=\frac1R\sum_r S\big(q_\theta,y^{(r)}\big)-\hat t_A .
$$

Both are exactly unbiased, so a squared-error loss converges to $A$ and to $E(q_\theta)$ respectively. The PPD is frozen (§4.4), so the replicates are used only by the UQ heads. $e(q_\theta)$, $S(q_\theta,y^{(r)})$ and the Phase 0 grading all use the same CDF built from the quantiles, otherwise $\hat r$ (§4.7) picks up a constant offset.

A lot of inspiration comes from the method and results given in Lahlou et al.’s (2021) paper.

### 4.3 The prior: exact replicates by an observed cut

#### Why the original prior cannot be used directly

In `graph_scm` (Qu et al., 2026), random vectors are sampled at root nodes and propagated through a random DAG. Each feature reads **a subset of a node's dimensions**; the unread dimensions stay unobserved, and *that* is the prior's only source of noise (no node-level Gaussian noise is added). Many node functions are dense maps that blend all input coordinates, so the hidden dimensions that create the noise are entangled with the features. There is no clean way to "freeze $x$ and rerun with fresh noise".

**Formally.** Write $X=\Phi_\theta(U)$ and $y=\Psi_\theta(U,V)$, where $U$ is randomness shared between $X$ and $y$ and $V$ reaches $y$ only. An exact replicate requires first drawing $U$ from the *abduction posterior* $p_\theta(U\mid\Phi_\theta(U)=x)$ (Balke & Pearl, 1994), which for arbitrary nonlinear graphs may be intractable. Freezing the root dimensions that are ancestors of $x$ under-reports the noise; using nearest neighbours as replicates gives a bias that grows with dimension. Both were dropped.

#### The observed cut

Keep the feature generator's mechanism, and make the observed $X$ the **only** frozen-side input to a second, `graph_scm`-style target graph with its own private noise:

$$
X=\Phi_\theta(U_1)\ \ (\text{unmodified }\texttt{graph\_scm}), \qquad y=\Psi_\theta(X,U_2), \qquad U_1\perp U_2 .
$$

Now nothing has to be inferred from $X$: $p_\theta(y\mid X=x)$ is the law of $\Psi_\theta(x,U_2)$ with $U_2$ at its ordinary marginal. **Replication = hold $X$, redraw $U_2$, rerun the target branch.**

```
 U1 ──► [ X-graph: graph_scm mechanism ] ──► X (observed columns)
                    │ feature-hosting nodes
                    ▼
        [ X-input nodes: observed post-converter columns ‖ fresh hidden dims ] ─┐
        [ private noise roots ] ────────────────────────────────────────────────┼─► [ Ψθ: graph_scm-style DAG ] ──► y
                                  ▲                                             │
                     U2 = hidden dims + private roots, redrawn per replicate ───┘
```

The target graph $\Psi_\theta$ copies `graph_scm` as closely as possible:

- **Same DAG sampler, size range and node functions** as the $X$-graph (Cauchy DAG, 2–32 nodes, tabicl's node functions); tabicl's pruning keeps only the ancestors of $y$.
- **$X$-input nodes are the $X$-graph's feature-hosting nodes.** A random subset of them (mirroring `subsample_feature_nodes`) is placed on random DAG positions whose incoming edges are cut. Each carries the **observed post-converter values** of the columns read from its host node, recomputed from the observed (outlier-clamped) columns, so $y$ depends on $U_1$ only through $X$. For the two categorical modes whose write-back is a latent value (`neighbor_id`, `softmax_id`), the observed category's center or embedding is used instead.
- **Hidden dims.** Each $X$-input node also carries LogInt(1, 32) fresh random-point dimensions, mirroring `graph_scm`'s $d_i=\sum_j d_{ij}+\text{LogInt}(1,32)$, where the unread dimensions of a node are the noise. They belong to $U_2$.
- **Private roots.** Every other parentless node draws random points, as `graph_scm`'s roots do ($U_2$).
- $y$ is read from a random non-input node with the usual converter; $y$-warping follows `disallow_y_warping`.

| | |
|---|---|
| **Preserved** | $X$'s generating mechanism (same DAG sampler, node functions, converters, warping and cross-feature correlations). $y\mid x$ stays a non-Gaussian, heteroscedastic pushforward built from the same function classes |
| **Not identical** | $X$'s distribution. (1) The $X$-graph is sampled with no $y$ node; in `graph_scm`, $y$ takes part in node assignment, pruning and the graph check, so the graphs and column placements that occur change. (2) The predictability filter runs on the cut $y$, so accepted $X$ is conditioned on a different acceptance event. (3) `ensure_iid=True`, whereas TabICLv2 was pretrained with `ensure_iid=False`. The difficulty-match gate below checks that the shift is small in practice |
| **Given up** | (1) Latent confounding between $X$ and $y$. (2) Anti-causal features: in `graph_scm` features can be descendants of the $y$ node, while in the cut $X$ is never downstream of $y$. Both are measured by the T3 transfer test |
| **Cost** | One extra pass of the target graph gives a replicate for every row at once (rows are tiled $R$ times), so $R$ replicates cost one pass per dataset |

**Implementation requirements.**

- Set `ensure_iid=True`, so that transforms are not fitted on the rows they transform.
- Cache task-level objects in `RandomNodeFunction._fit` (root point law, feature-importance weights, node importance). Only the private noise may be redrawn.
- Split post-processing into `fit` / `apply` and apply the observed dataset's constants to every replicate. Targets must live in the same units the model predicts in. Concretely, $y$'s transform (outlier clamp at mean ± 4σ, then standard scaling, as in TabICLv2) is fitted once on the full pool's $y$, before any split manipulation, and applied unchanged to every replicate. Clamped replicates are then i.i.d. draws from the clamped task law the PPD approximates, so $\hat t_A$ stays unbiased. Never refit per replicate and never clip the targets themselves.
- Fix `PYTHONHASHSEED` in every sampling job.
- Assert bit-identical $X$ across replicates. This is the only correctness guard the construction needs.

**Task filtering.** TabICLv2 pretrains only on tasks that pass its filters (about 25 % of stage-1 regression tasks are rejected), so our tasks pass the same filters, applied to whole tasks before the split and the replicates (a rejected task is redrawn from scratch):

- *Graph check*, adapted: applied to the **target graph**, resampled until at least one $X$ input is an ancestor of $y$ (the $X$-graph has no $y$ node).
- *Constant columns* removed; a task with no feature left is rejected.
- *Predictability test* as in TabICLv2: an ExtraTrees model must beat the mean-label baseline out-of-bag (p < 0.05).
- *Trivial-task test* off, so that the near-deterministic $A\approx0$ end is learned.
- Non-finite $y$ or replicate values reject the task.

The same filter is used for the validation bank and the T2 test bank, and the rejection rate is logged per stage; if it is much higher than TabICLv2's, tune the target graph's noise scale, not the filter.

**Difficulty-match gate (prior level, before training).** Generate matched banks of cut tasks and unmodified `graph_scm` tasks (same row and feature counts, plain i.i.d. context–query split, no holes or extrapolation queries for this test only). On each task compute the frozen TabICLv2 CRPS on the queries and ExtraTrees held-out $R^2$ on the same split; compare the two distributions (medians, quantiles, two-sample tests) and the filter rejection rate per stage of both priors. If cut tasks are much easier or much harder, **tune the target graph** (hidden dims on $X$-input nodes, number or scale of private roots), **not the filters**, and rerun. The configuration and tolerance used are recorded in the Phase 1 documents (`experiment1_uq_heads/scripts/00f_difficulty_match.py`).

**Known caveat.** The v2 prior's rejection filters discard near-unpredictable datasets, truncating the high-aleatoric tail the head is trained on. The empirical distribution of $\hat t_A$ is logged and reported.

#### Audits

- **Estimator audit (before training).** First, a hand-made prior $x=f(a)$, $y=g(x)+s(x)\,b$, with $b\sim\mathcal N(0,1)$ and $a\perp b$, where $e_{\mathrm{CRPS}}=s(x)/\sqrt\pi$ is known. Check $\hat t_A$ over the deterministic ($s=0$), homoscedastic ($s=\sigma$) and heteroscedastic ($s=e^{\alpha x}$) cases, averaging over a few hundred queries per level. Then run $\hat t_A$ on the observed-cut prior and report the zero-mass fraction, the spread of the non-zero part, and its variation across $x$ within a task. This helps us make sure that the aleatoric entropy estimator works as expected in a MC limit. Phase 0 runs the estimator checks once (on the toy rungs, plus the replicate estimator of $E$), so both phases can cite them.


### 4.4 The heads

$e(q_\theta)$ is not learned; it is computed in closed form from the quantile head (§2.3). Two scalar heads are added on top of the **frozen** TabICLv2 regressor, both reading $h_*$, the representation the quantile decoder reads (the post-final-LayerNorm output of the ICL transformer at the query row, 512-d):

$$
\hat A(x_*,D)=\mathrm{softplus}\big(a_\theta(h_*)\big),\qquad
\hat E_{\text{dir}}(x_*,D)=\mathrm{softplus}\big(b_\theta(h_*)\big).
$$

Each head has its own trunk, mirroring the quantile decoder: `Linear(512→1024) → GELU → Linear(1024→1) → softplus`. Separate trunks keep the noisier $\hat t_E$ gradients from degrading $\hat A$.

- **Why softplus.** It resolves values spanning orders of magnitude (the backbone check shows $E$ from $10^{-4}$ to $10^{0}$) and keeps squared-error gradients from exploding. The non-negativity is correctly specified: $\hat t_A\ge0$ and $\mathbb E[\hat t_E\mid x_*,D]=\mathbb E_{\tau\mid D}[d_S(q_\theta,p_\tau)]\ge0$.
- **No coupling to $e(q_\theta)$, and no gate** *(previous draft: $\hat A=e(q_\theta)\cdot\mathrm{softplus}(z)$ with a soft penalty $\mathrm{ReLU}(\hat A-e(q_\theta))^2$)*. Coupling would (1) contaminate the amortization-gap estimate (§4.7) and (2) tie both heads to a value that does not always follow the true divergence: the backbone check shows $e(q_\theta)$ misses it in sparse-context intervals.
- **Units.** Both heads live in TabICLv2's internal z-normalized $y$ units; multiplying by the context s.d. gives raw units exactly, since CRPS entropy and divergence are positively 1-homogeneous.

**Why $h_*$ alone: the link to DEUP.** DEUP's error predictor needs *stationarizing features* of $(x,\text{dataset})$: a density estimate, a model-variance estimate and a seen-bit (Lahlou et al., 2023, §3). In a PFN the dataset conditioning is built in: $h_*$ is a function of $(x_*,D)$, since the query attends to every context row, so it is a learned proxy for all three (how many relevant rows were attended to; a near-exact match with a context row; the spread of the forecast, of which $e(q_\theta)$ is a deterministic function). Whether $h_*$ actually kept what the split needs is tested in §4.8.

### 4.5 Loss and training

$$
\mathcal L=\overline{\,w\,\big(\hat A-\hat t_A\big)^2\,}+\lambda_E\,\overline{\,w\,\big(\hat E_{\text{dir}}-\hat t_E\big)^2\,},\qquad w=\big(\mathrm{sg}[e(q_\theta)]+c\big)^{-p},
$$

averaged over query rows. The pinball loss is absent because the PPD is frozen, and the heads share no parameters.

- **Stop-gradient** on $q_\theta$ inside $\hat t_E$, so the target is a fixed function of the frozen forecast *(previous draft: so the network cannot lower $\mathcal L_E$ by degrading its own forecast)*.
- **Weight.** Target noise scales with the quantity itself, so unweighted squared error is dominated by high-uncertainty queries. A weight depending **only on $(x_*,D)$** leaves the minimiser at the conditional mean; it only changes how finite capacity trades errors across inputs. $p=0$ (no weighting) is the default in every run; $p=2$ (inverse-variance) is a fallback ablation, run only if T2 shows the low-uncertainty bins badly calibrated while the high-uncertainty ones are fine.
- **No forecast degradation** by construction, so the earlier $\lambda_A,\lambda_E$ sweep against a no-auxiliary-head control *(previous draft)* is dropped; for the frozen-backbone arms the PPD is bit-identical to TabICLv2 and no $\lambda_E=0$ warm-up is needed.
- $\hat E_{\text{dir}}$ is **checkpoint-specific**: its target moves with $q_\theta$, so it must be re-fitted after any backbone change. $\hat A$ has no such dependence.
- Squared error only: no log-space regression, no quantile or absolute-error loss, and **never clip targets at zero**. Each of these breaks the conditional-mean argument of §2.4.

> **TBD — remaining training-recipe questions.** Real-data continued pretraining (RealPFN-style fine-tuning) and its ordering relative to head training. Unfreezing and warm-up are settled by the arms of §4.8.

### 4.6 What a user reads off

| Quantity | Meaning | Units |
|---|---|---|
| $e(q_\theta)$ | How spread out is the forecast? | units of $y$ |
| $\hat A$ | How much of it would remain with infinite data from this task? | units of $y$ |
| $\hat E_{\text{dir}}$ | How much score would I save if I knew the true task law? | score units |
| $\lambda=\hat E_{\text{dir}}/(\hat A+\hat E_{\text{dir}})$ | Epistemic fraction: share of the forecast's expected score that is reducible; scale-free, for comparison across datasets | — |

> **Note on $\lambda$.** Since $\hat A+\hat E_{\text{dir}}$ estimates the expected score of $q_\theta$ (§2.2), $\lambda\in[0,1]$ holds by construction, even when $\hat E_{\text{dir}}$ exceeds $e(q_\theta)$ because the PPD is off *(previous draft: $\lambda=\hat E/e(q_\theta)$, which does not stay in $[0,1]$ for $\hat E_{\text{dir}}$)*.

### 4.7 Amortization error — TBD

Two epistemic estimates are available: the subtraction $e(q_\theta)-\hat A$ and the direct $\hat E_{\text{dir}}$. Their difference

$$
\hat r=\big(e(q_\theta)-\hat A\big)-\hat E_{\text{dir}} \;\longrightarrow\; e(q_\theta)-e(q)-d_S(q_\theta,q)
$$

vanishes when the network's PPD equals the ideal PPD $q$, so it measures how far the amortization is from exact Bayes at that query. It is logged for every iLTMv2 run, and the uncoupled head design (§4.4) keeps it uncontaminated.

> **TBD:** whether $\hat r$ is reported, how it is used (diagnostic, flag, evaluation axis), and whether it is useful at all.

### 4.8 Architecture — the ladder of arms

The heads read $h_*$ by default, but pinball pretraining only asks $h_*$ to encode the posterior *predictive* (a mixture of task laws), whereas the decomposition needs the posterior *over tasks* (Bergna et al., 2026, Prop. 1). Nothing forced $h_*$ to keep what tells "a mix of sharp laws" from "one wide law". A **ladder** of arms gives the heads progressively more access to the backbone's information, at higher cost; we climb only while validation loss and T2 reliability improve. The rung where gains stop is itself a finding about what PFNs keep when they amortize the posterior.

| Arm | What the heads read | Backbone |
|---|---|---|
| **A1** (default) | $h_*$ through the MLP trunks of §4.4. Tests whether the split is present in $h_*$ | Fully frozen; PPD identical to TabICLv2 |
| **A2** | A learned softmax-weighted mix of per-layer, separately normalised residual-stream vectors of the query row (e.g. blocks 4, 8, 12, or all 12). Addresses information that later layers merged away; the learned weights show *which depth* holds the decomposition information | Frozen; PPD unchanged |
| **B** | A trainable copy of the last $k\in\{1,2\}$ ICL-transformer blocks and final LayerNorm, initialised from the pretrained weights, re-attending to the context under the UQ loss and feeding only the heads | Originals stay frozen and keep feeding the decoder, so the PPD is bit-identical and B starts exactly where A1 ends. Unlike Bergna et al., who unfreeze shared layers and change the PPD. Costs about $k/12$ extra compute |

**Comparing rungs needs no ground truth.** On held-out cut-graph tasks, $\mathbb E[(\hat A-\hat t_A)^2]=\mathbb E[(\hat A-A)^2]+\text{target-noise variance}$, and the noise term is identical across arms, so a lower validation loss *is* a lower true error. T2 grading then confirms it.

> **Ablation (only if the first experiments fail).** Append one stop-gradient scalar, the standardised $\log e(q_\theta)$, to the trunk input. It adds no information but hands the trunk DEUP's variance feature ready-made.

### 4.9 Training procedure

Following TabICLv2's meta-training logic: one training example is a whole synthetic task; an infinite stream of fresh tasks, a fixed step budget, **no early stopping**, and a fixed validation bank used only to *compare* runs and checkpoints.

**Task construction.** The split is built on $X$ first and $y$ is drawn last. Each task mixes three query types, built from an enlarged pool whose extra rows are discarded so that the context never sees the carved regions:

| Query type | Share | Construction |
|---|---|---|
| i.i.d. | 70 % | TabICLv2's ordinary random split |
| kNN hole | 15 % | Rows nearest a random centre are removed from the context; queries are drawn from them |
| Extrapolation | 15 % | Rows extreme along a random direction in standardised $X$ space are removed; queries are drawn from them |

Hole and tail sizes are drawn per task (log-uniform, up to $0.2N$), so queries cover a range of divergences. In addition, 5 % of query slots get their $x$ overwritten by a context row's $x$ (a fresh $y$ is drawn), teaching the low-divergence end (cf. DEUP's seen-bit). Every split decision depends only on $X$, so the posterior over the $y$-law given the manipulated context is unchanged and the targets keep their meaning; selecting on $y$ would break this and is never done. The mixture only decides where head capacity is spent, not the minimiser.

**Curriculum** (TabICLv2's 500K : 40K : 10K stages, scaled by 1/10), batch of 64 tasks (gradient accumulation in stages 2–3), AdamW with a cosine schedule restarted per stage:

| Stage | Steps | Rows $N$ per task | Context fraction | Head LR |
|---|---|---|---|---|
| 1 | 50,000 | 1,024 | 30–90 % | `lr` |
| 2 | 4,000 | 400–10,240 | ≈80 % | `lr`/8 |
| 3 | 1,000 | 400–60,000 | ≈80 % | `lr`/40 |

**Validation.** A fixed bank of 2,000 tasks (50 / 30 / 20 % small / medium / large $N$, so large-$n$ bins are not empty), same pipeline and query mixture, $R_{\text{val}}=128$ replicates, seeds disjoint from every test bank. The unweighted MSE of $\hat A$ and of $\hat E_{\text{dir}}$ is logged separately, per query type and per $n$ bin, every 2,500 steps and at each stage end. Three checkpoints per run: latest, best-$\hat A$ and best-$\hat E_{\text{dir}}$ (checkpoint selection, not early stopping). In B the heads share the fork, so both heads are logged at both checkpoints to expose trade-offs.

**Sweeps and selection.** Head LR $\in\{10^{-4},3\cdot10^{-4},10^{-3}\}$ for A1 and A2; for B also fork LR $\in\{3\cdot10^{-6},10^{-5},3\cdot10^{-5}\}$ with $k=2$. All arms use paired seeds $\{0,1,2\}$ (the seed fixes both task stream and head initialisation).

1. Within A1/A2, configurations are chosen **per head** (the heads are independent); within B, as a whole configuration.
2. Arms are compared on a mixed score, the sum of each head's MSE relative to the default A1 configuration (below 2 means better than default A1 overall); it is used only to choose the arm.
3. B runs 9 configurations on seed 0 only; seeds 1–2 are spent on B's winner only if it clearly improves on A1/A2.
4. The checkpoint is the best-validation seed of the winning arm. T1–T4 results are reported for it together with the 3-seed mean ± std.

**Job layout.** In A1/A2 nothing upstream of the heads trains, so one job runs the backbone once per step and feeds all independent A1 and A2 heads (the backbone forward is almost all of the compute). B needs one job per configuration, since each fork changes the forward pass.

### 4.10 Evaluation

iLTMv2 is run on **all Phase 0 tests**.

---

## 5. Phase 2 — Epistemic-guided context retrieval

### 5.1 What this phase does

1. Establish a **new method** for epistemic-guided context retrieval.
2. **Ablation study** over implementations: inference-only, fine-tuned, or built into pretraining.
3. **Test** it on existing retrieval benchmarks, following the protocols of TabDPT (Ma et al., 2024) and LoCalPFN (Thomas et al., 2024).

### 5.2 The problem and the move

**The problem.** A PFN's prediction depends on which rows are in its context. Current retrieval methods pick the $k$ nearest neighbours of the query in feature or embedding space (Thomas et al., 2024; Ma et al., 2024). That is a *proxy* for "reduce uncertainty at the query", and a limited one: neighbours cluster, and clustered context is redundant. If the local structure is a trend, the informative rows are *spread out*, not *near*. A distance criterion cannot express that; an epistemic criterion can.

**The move.** With a supervised epistemic estimate, the real objective becomes available:

$$
S^\star=\arg\min_{S\subseteq D,\ |S|\le B}\ \hat E_{\text{dir}}(x_*,S).
$$

$\hat E_{\text{dir}}$, the excess score of the model's own forecast, is what changing the context actually moves. The general idea is greedy and per query, in the spirit of BALD-style acquisition (Houlsby et al., 2011) but with labels already known:

```
pool D, query x*  →  score each candidate by the drop in Ê_dir(x*) if it were added
                  →  add the best row  →  repeat until the budget B is reached
```

### 5.3 Open design — TBD

> **TBD:** the method itself (search strategy, cost, batched scoring); the training or pretraining pipeline behind each ablation arm; benchmark datasets and metrics; and safeguards against selecting a context on which the model is *confidently wrong* (a lower $\hat E_{\text{dir}}$ without a lower realized score).

### 5.4 Link to Phase 0

The **active-learning** task of the Usefulness test is already context retrieval without training. It is the first, cheapest check of whether $\hat E_{\text{dir}}$ selects useful rows.

---

## 6. Cross-cutting

**Benchmarks in scope:** ScoringBench (Landsgesell et al., 2026), TALENT (Liu et al., 2024), TabArena (Erickson et al., 2025).

**Dependencies.**

```
Phase 0 ──► Phase 1 ──► Phase 2
  │  In-domain test needs the Phase 1 prior (§4.3)
  └─ Active Learning tests previews Phase 2
```

**Removed from scope** (previous draft): a quasi-martingale coherence statistic (to directly estimate the quality of our aleatoric and epistemic entropies), prior-support scoring and directed prior expansion with the coherence statistic, and epistemic-conditioned conformal calibration.

**Open items (consolidated).**

1. T2 generator settings (§3.4).
2. T3: forcing $k$-level categorical features in `graph_scm`; final $(k,d)$ sweep; mixed features (§3.5).
3. Dec-ICL and Tuned-ICL checkpoints (authors contacted), Dec-ICL output format and backbone version (§3.6–3.7).
4. T4 dataset sources (§3.6).
5. Remaining training-recipe question: real-data fine-tuning and its ordering (§4.5).
6. Use of the amortization error $\hat r$ (§4.7).
7. The Phase 2 method, pipeline, metrics and safeguards (§5.3).
8. Kill criteria per phase.

---

## References

- Amini, A., Schwarting, W., Soleimany, A., & Rus, D. (2020). Deep evidential regression. *NeurIPS*. arXiv:1910.02600
- Balke, A., & Pearl, J. (1994). Probabilistic evaluation of counterfactual queries. *AAAI*.
- Bergna, R., Depeweg, S., & Hernández-Lobato, J. M. (2026). Decoupled PFNs: Identifiable epistemic-aleatoric decomposition via structured synthetic priors. arXiv:2605.06413
- CatBoost documentation. Uncertainty. https://catboost.ai/docs/en/references/uncertainty
- Depeweg, S., Hernández-Lobato, J. M., Doshi-Velez, F., & Udluft, S. (2018). Decomposition of uncertainty in Bayesian deep learning for efficient and risk-sensitive learning. *ICML*.
- Erickson, N., et al. (2025). TabArena: A living benchmark for machine learning on tabular data. arXiv:2506.16791
- Fortini, S., Ng, K., Petrone, S., Rousseau, J., & Wei, S. (2026). A principled framework for uncertainty decomposition in TabPFN. arXiv:2602.04596
- Gneiting, T., Balabdaoui, F., & Raftery, A. E. (2007). Probabilistic forecasts, calibration and sharpness. *JRSS-B*, 69(2), 243–268.
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA*, 102(477), 359–378.
- Hofman, P., Sale, Y., & Hüllermeier, E. (2024). Quantifying aleatoric and epistemic uncertainty with proper scoring rules. arXiv:2404.12215
- Houlsby, N., Huszár, F., Ghahramani, Z., & Lengyel, M. (2011). Bayesian active learning for classification and preference learning. arXiv:1112.5745
- Kang, J. (2021). Low-rank method to fasten the inference of Gaussian process. STAT 243 lecture notes.
- Lahlou, S., Jain, M., Nekoei, H., Butoi, V. I., Bertin, P., Rector-Brooks, J., Korablyov, M., & Bengio, Y. (2021, February 16). DEUP: Direct Epistemic Uncertainty Prediction. arXiv.org. https://arxiv.org/abs/2102.08501
- Lahlou, S., Jain, M., Nekoei, H., Butoi, V. I., Bertin, P., Rector-Brooks, J., Korablyov, M., & Bengio, Y. (2023). DEUP: Direct epistemic uncertainty prediction. *TMLR*.
- Landsgesell, J., Knoll, P., & Wenzel, T. (2026). ScoringBench: A benchmark for evaluating tabular foundation models with proper scoring rules. arXiv:2603.29928
- Liu, S., Cai, H., Zhou, Q., & Ye, H. (2024). TALENT: A tabular analytics and learning toolbox. arXiv:2407.04057
- Ma, J., et al. (2024). TabDPT: Scaling tabular foundation models on real data. arXiv:2410.18164
- Malinin, A., Prokhorenkova, L., & Ustimenko, A. (2021). Uncertainty in gradient boosting via ensembles. *ICLR*.
- Müller, S., Hollmann, N., Arango, S. P., Grabocka, J., & Hutter, F. (2021). Transformers can do Bayesian inference. arXiv:2112.10510
- PyMC. Heteroskedastic Gaussian processes. https://www.pymc.io/projects/examples/en/latest/gaussian_processes/GP-Heteroskedastic.html
- Qu, J., Holzmüller, D., Varoquaux, G., & Le Morvan, M. (2025). TabICL: A tabular foundation model for in-context learning on large data. *ICML*.
- Qu, J., Holzmüller, D., Varoquaux, G., & Le Morvan, M. (2026). TabICLv2: A better, faster, scalable, and open tabular foundation model. arXiv:2602.11139
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*, ch. 2. MIT Press.
- Thomas, V., et al. (2024). Retrieval & fine-tuning for in-context tabular models (LoCalPFN). *NeurIPS*. arXiv:2406.05207
