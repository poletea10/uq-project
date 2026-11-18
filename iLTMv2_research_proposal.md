# Score-Based Uncertainty Decomposition for Tabular Foundation Models

**Research proposal — iLTMv2 / TFM final thesis** · Pol Gàlvez Casasús · revised 18 November 2026

> **Legend.** **TBD** marks a part whose design has not been decided yet. Rejected alternatives are stated where they explain a decision; they are not options.

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
| **0 — Setting the baseline** | Which decomposition do we use, where do existing methods put aleatoric and epistemic uncertainty against closed-form truth, and do the estimates rank real errors and guide data selection? | Six tests (T1–T6, §3) and the baseline results on them |
| **1 — Model training** | Can TabICLv2 learn $\hat A$ and $\hat E_{\text{dir}}$ from generator supervision? | iLTMv2 with two UQ heads (arms A and B), its final model chosen with the Phase 0 tests and evaluated on all of them |
| **2 — Epistemic-guided context** | Does selecting context by $\hat E_{\text{dir}}$ beat k-NN retrieval? | Retrieval method + ablation + benchmark results; T6 (§3.9) is its first, inference-only check |

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
2. **Show** where each model puts aleatoric and epistemic uncertainty against closed-form truth: on one curated 1-D function built for the paper's main figure (T1), and across the whole grid of local data density × local noise (T2).
3. **Rank** real errors: whether $\hat A+\hat E$ ranks the forecast's realized loss better than its own spread (T3), and whether $\hat E$ ranks the realized gain from more data (T4).
4. **Check usefulness** for decisions: the active-learning benchmark of Bergna et al. (2026) (T5) and $\hat E$-guided context retrieval for a single query (T6, the prelude to Phase 2).
5. Run the baselines on all of the above, for comparison with iLTMv2.

T1–T2 ask *"is the number right?"*; T3–T4 ask *"does the number rank the errors it claims to predict?"*; T5–T6 ask *"does the number lead to good decisions?"*. The literature has so far only answered the last (Bergna et al., 2026; Fortini et al., 2026).

| # | Test | Question | Ground truth | Models | Output |
|---|---|---|---|---|---|
| T1 | Toy set | Where does each model put $\hat A$ and $\hat E$ on one curated function? | Closed form | Heteroscedastic GP, Dec-ICL, iLTMv2 | Figures |
| T2 | Error heatmap | Are $\hat A$ and $\hat E$ right across local data density and local noise? | Closed form | Heteroscedastic GP, Dec-ICL, iLTMv2 | One heatmap figure |
| T3 | PRR-A (total error) | Do $\hat A+\hat E$ rank the forecast's loss better than its own spread? | Realized CRPS | iLTMv2 (Dec-ICL if available) | Curves + table |
| T4 | PRR-B (reducible error) | Does $\hat E$ say which predictions would improve with more data? | Realized CRPS gain | iLTMv2 (Dec-ICL if available) | Curves + table |
| T5 | Active learning | Does the epistemic estimate select useful data? | Test-set performance | Random, TabICLv2, iLTMv2, Tuned-ICL, Dec-ICL | Table + curves |
| T6 | Context retrieval | Does $\hat E$-guided retrieval beat k-NN for a given query? | Realized CRPS | iLTMv2 | Curves + table |

T1 and T2 are the visual results. T3–T6 are the thorough evaluation on real data. Phase 0 trains nothing; iLTMv2 rows stay visibly empty until Phase 1 delivers the model. The row set, the toy function and every test's grid and constants are fixed before the first run. The execution details (constants, seeds, figure layouts, adapters) live in `phase0_protocol.md`.

### 3.2 What is measured

Every model outputs a predictive distribution $q_m$, as a set of members or as a CDF. From it we read

$$
\hat A = \text{mean member entropy}, \qquad \hat E = e(q_m) - \hat A ,
$$

and compare:

- $\hat A$ against the **true aleatoric entropy** $A=e\big(p_\tau(\cdot\mid x_*)\big)$ of the realized task.
- $\hat E$ against the **realized excess** $E(q_m)=\mathbb{E}_{Y\sim p_\tau}[S(q_m,Y)]-A$: the CRPS that the model's *own* forecast actually pays above the irreducible floor.

The epistemic ground truth is therefore **model-specific** (§2.2): each model is judged on whether it knows its own error, not on distance from a shared Bayes reference. iLTMv2's $\hat E_{\text{dir}}$ is trained to target exactly $E(q_\theta)$ (§4.4), and in every test its epistemic estimate is the head itself; $e(q_\theta)-\hat A$ is not used for iLTMv2.

**Why $e(q_m)-\hat A$ is a fair estimator for the baselines.** One could say that $e(q_m)$ is not the forecast's expected score under the true law, so why should $e(q_m)-\hat A$ match the realized excess? Nobody has the true law at test time, including our head: every model can only *predict* its excess from $(x_*,D)$. A Bayesian-style model holds candidate task laws $F_m$ with weights $w_m$ and forecasts their mixture $q_m=\sum_mw_mF_m$. Averaging the excess over its own belief, and using that the expected score is linear in the law of $Y$,

$$
\sum_m w_m\,\mathbb E_{Y\sim F_m}\big[S(q_m,Y)\big]=e(q_m),
\qquad\text{so}\qquad
\sum_m w_m\,d_S(q_m,F_m)=e(q_m)-\hat A .
$$

So $e(q_m)$ is the model's own prediction of its expected score, $\hat A$ its prediction of the floor, and $e(q_m)-\hat A$ exactly its prediction of its excess (non-negative by concavity of $e$): the standard total-minus-aleatoric decomposition (Hofman et al., 2024; for the log score, the mutual information of BALD, Houlsby et al., 2011; Depeweg et al., 2018). If the members are the exact posterior over tasks, it equals $\mathbb E_{\tau\mid D}[d_S(q,p_\tau)]$, precisely what $\hat E_{\text{dir}}$ converges to under squared error, so both estimators aim at the same quantity in the same units. What differs is how the prediction is formed: the baselines derive it from their own posterior, ours is learned from realized excess (DEUP's idea; Lahlou et al., 2023). A misspecified posterior misses its own bias; that gap is what the tests show, not an artefact of the conversion. The worked example and the limits of the argument are in `phase0_protocol.md`.

**The standardisation layer.** Every baseline reduces to a weighted family of predictive CDFs $\{(w_m,F_m)\}$:

$$
\hat A=\sum_m w_m\,e(F_m), \qquad e_{\text{tot}}=e\Big(\sum_m w_mF_m\Big), \qquad \hat E=e_{\text{tot}}-\hat A\ \ge 0 .
$$

One entropy kernel, $e_{\mathrm{CRPS}}(F)=\int F(1-F)\,dt$, serves every model. **Project member-wise:** never convert an already-aggregated variance split into entropy units, since $\mathrm{Var}=\mathbb E[\sigma^2]+\mathrm{Var}[\mu]$ is additive in variance and $\sigma/\sqrt\pi$ is not. The truth code and every adapter's conversion are unit-tested against closed forms before any model is run.

### 3.3 T1 — Toy set

One **fixed, curated** 1-D function on $[-5,5]$, one fixed context draw, three noise settings:

$$
y = f(x) + \sigma(x)\,\varepsilon,\qquad \varepsilon\sim\mathcal N(0,1).
$$

- **The function.** $f$ is a low-rank GP (28 Gaussian bumps, width 0.3, unit prior s.d.; Rasmussen & Williams, 2006) plus a **gap part**: narrow bumps (width 0.1, s.d. 1.25) that exist only inside the sparse strip $[1,4]$. The dense regions carry no hint of the gap's higher frequency, so a model that infers smoothness from the data it sees is confidently wrong in the gap: posterior variance misses misspecification bias, while excess risk includes it (Lahlou et al., 2023).
- **Curation.** The seed is the first one meeting three pre-registered conditions (bounded amplitude, visible gap structure, non-flat noise strip), chosen on $f$ alone without looking at any model output, then frozen.
- **Context and queries.** The context density is 13× lower inside $[1,4]$; the context is drawn once at $n=4096$ and nested over $n\in\{64,256,1024,4096\}$, so the sparse strip holds about 2, 8, 32 and 128 context points; we predict on 200 evenly spaced queries. All rungs share $f$, the context and the noise draws. The grid starts at 64 because the heads are trained on contexts down to 64 rows (§4.9).
- **Truth** is closed form and exact per query: $A=\sigma(x_*)/\sqrt\pi$, $E(q_m)=\int(F_m-\Phi)^2dt$.

| Toy rung | Noise | Question it asks |
|---|---|---|
| 0 | $\sigma\equiv0$ | Does $\hat E$ rise in the sparse strip with $\hat A\approx0$? |
| 1 | $\sigma\equiv\sigma_0\approx0.25$ | Does gap detection survive noisy surroundings? |
| 2 | $\sigma_0$ outside, $\approx6\sigma_0$ on the *noise strip* $[-3.5,-1.5]$ | Can it *attribute* uncertainty correctly when noise and data density vary independently? |

**What to expect.** Each rung is read as "does $\hat E$ follow $E(q_m)$?", never as "does $\hat E$ match rung 0?", since the model's error $E(q_m)$ is different depending on the complexity of context. The rung-2 test is whether $\hat A$ rises only in the noise strip and $\hat E$ is higher in the gap than in the noise strip. The realized excess of one function is jagged; a correct $\hat E$ is a smooth curve through it (§3.5), so a match in level and shape is the expected good result, not a point-by-point match.

**Figures.** One 3×4 figure per (model, toy rung): 9 figures (rows: context and forecast; $A$ vs $\hat A$ with total $e(q_m)$ dashed; realized $E(q_m)$ vs $\hat E$ on a log axis, with a small legend naming each model's estimator, $e(q_m)-\hat A$ or the $\hat E_{\text{dir}}$ head; columns: $n$). On rung 0, row 2 has per-model y-limits (or a log axis), so that $\hat A\approx0$ is visible. Plus one 3×3 **paper figure** (rung 2, $n=1024$, one column per model); the full 3×4 figures go to the appendix. It is an illustration for the paper, not a grade: one function cannot support calibration claims. The toy function is closer to the heteroscedastic GP's and Dec-ICL's function classes than to iLTMv2's prior, while iLTMv2 is the only model trained on exactly this score-based target (which is the method, not a bias of the test). We state both.

**Backbone check (runs first).** TabICLv2 is run on all rungs and $n$ to confirm the backbone behaves sensibly on this input, with pass criteria fixed beforehand: (1) CRPS outside both strips at most $1.5\times$ the heteroscedastic GP's; (2) on rung 0, total $e(q_\theta)$ in the sparse strip at least $2\times$ that outside; (3) on rung 2, total $e(q_\theta)$ in the noise strip at least $2\times$ that outside. It doubles as the motivating figure: the single total rises in both strips and says nothing about which is which.

> **Toy function $f$ — decided.** Option (a), the curated low-rank GP with extra high-frequency structure in the sparse strip, in the spirit of DEUP (Lahlou et al., 2023, Fig. 1). **Fallback if the backbone check fails:** option (b), a 1-D draw from the cut-graph prior (the target branch with its noise frozen) plus our own Gaussian noise, curated for at least as much variation inside $[1,4]$ as outside, so that $f$ comes from a function class iLTMv2's backbone has seen. It loses the controlled "harder in the gap" structure.

### 3.4 T2 — Error heatmap

T1's front figure (rung 2 + 1,024 context + model comparison) shows two situations on one function: a noisy region with plenty of data, and a sparse region with little noise. T2 fills in the whole grid: for every combination of local data density and local noise, it shows whether each model's $\hat A$ and $\hat E$ are too high, right, or too low. The two strips of T1 would be, therefore, two cells of this grid.

**Construction: a probe strip.** Each toy function (the toy generator's base family on $[-5,5]$, a fresh draw per toy, no gap part) has a fixed background and one strip of width 2 at a random position. The cell sets the strip's local density (0, 2, 8, 32, 128 context points, or the background density) and local noise s.d. (0, 0.1, 0.25, 0.6, 1.5); only queries inside the strip are scored. The surroundings of the strip, the function, the background context and the noise draws are shared by every cell of a toy, so a cell's value is not confounded by what lies around it. 20 toys (the first 10 for the heteroscedastic GP, which needs one NUTS fit per toy and cell).

**Truth.** Closed form per query, as in T1: $A=\sigma(x_*)/\sqrt\pi$ and $E(q_m)=\int(F_m-\Phi_\tau)^2dt$, the model's own excess.

**Cell value.** For each model, quantity and cell,

$$
\text{value}=\log_2\frac{\text{mean estimate}}{\text{mean truth}},
$$

with means over all toys and queries of the cell, clipped at ±3. The mean is taken before the ratio, so the scatter of §3.5 averages out; in the $\sigma=0$ row a fixed floor of 0.01 is added to both means for $\hat A$.

**Figure and reading.** One row per model, two heatmaps per row ($\hat A$ and $\hat E$); signed colour, red = too high, white = correct, blue = too low; each model uses the estimator of §3.2. A model that reads sparse data as noise is red in the left columns of its $\hat A$ panel and blue in the same cells of its $\hat E$ panel. A model that decouples correctly is white in both. The grid, the distributions and the floor are fixed before the first run and not tuned afterwards.

### 3.5 The epistemic scatter and the performance ceiling

No estimator can predict the realized excess of a single task exactly. The same context $D$ is compatible with many tasks, each with a different true law at the query. Tasks are drawn from a fixed prior, so

$$
\mathbb E\big[E(q_m)\mid D,x_*\big]=\bar E(D,x_*)=\mathbb E_{\tau\mid D}\big[d_S(q_m,p_\tau)\big],
$$

and $\bar E$ is the best any estimate, a function of $(D,x_*)$, can do. The realized value is $\bar E$ plus a zero-mean deviation that the context cannot resolve: the **scatter**. If it were predictable from the context, the forecast itself could be corrected with it.

The scatter is of the same order as the quantity itself: for a calibrated Gaussian forecast its coefficient of variation lies between 0.72 (deterministic task) and 1.41 (noise-dominated task), and with a coefficient of variation CV the best possible squared error is $\mathrm{CV}^2/(1+\mathrm{CV}^2)$ times the target's second moment. A perfect estimator therefore has a skill well below 1; on the Phase 1 validation bank the ceiling of the pooled $\hat E$ skill is bracketed between 0.44 and 0.64 (`phase1_design.md`, validation section). Lahlou et al. (2023, Fig. 6c and App. D) meet the same limit and report observed over achievable correlation.

What it means for each test: in T1 a smooth $\hat E$ through a jagged truth is the good result; T2 compares means, where the scatter averages out; in T3–T4 the oracle ordering uses each point's realized loss, which no score can know, so a perfect score has a PRR well below 1, and PRR values compare scores on the same forecast and dataset, not across datasets.

### 3.6 T3 — PRR-A: ranking total error

**Question.** Do the two heads together predict how large the forecast's loss will be, better than the forecast's own spread does? $e(q_\theta)$ is what the forecast expects to pay if it is right about itself; $\hat A+\hat E_{\text{dir}}$ is the learned estimate of what it actually pays (the expected score of $q_\theta$, §2.2). Both score the same forecast and the same errors. The observed loss on real data includes noise, so total uncertainty is the right ranker of it, which is why the sum is used.

**Construction.** The five active-learning datasets and their stored splits (§3.8; per seed a 1,024-row warm-start set, a 5,000-row pool and a 3,000-row test set; no carved splits). Context = warm-start set + pool (6,024 rows); queries = the test set; one raw forward of iLTMv2, the route the heads were trained on. Per test point, the loss is $\ell_i=\mathrm{CRPS}(q_i,y_i)$ and the scores are $s_i^{\text{tot}}=e(q_i)$ and $s_i^{\text{dec}}=\hat A_i+\hat E_i$. Sorting by a score, highest first, the rejection curve after rejecting the first $k$ points is the summed loss of the retained points divided by the total count (rejected points go to an oracle and contribute 0). Random rejection and the oracle ordering (by $\ell_i$) are the references, and

$$
\mathrm{PRR}=\frac{AU_{\text{random}}-AU_{\text{score}}}{AU_{\text{random}}-AU_{\text{oracle}}},
$$

with $AU$ the area under each curve: 1 is the oracle ordering, 0 is random, negative is worse than random.

**Outputs.** One panel per dataset with four curves (oracle, $e(q)$, $\hat A+\hat E$, random), averaged over seeds; a table of both PRRs per dataset (mean ± s.d. over seeds) and their paired difference. If the Dec-ICL checkpoints arrive, its decomposition sums to its total by construction, so on its own forecast it has one score, shown next to Tuned-ICL's total as context.

**Expectation, stated before running.** The two scores differ only where the backbone is wrong about itself, so the gap on ordinary splits may be small.

### 3.7 T4 — PRR-B: ranking reducible error

**Question.** Does $\hat E$ tell which predictions would improve with more data? This is what "epistemic" means operationally, and it gives an epistemic ground truth on real data without replicates. Each test point is predicted with a small context (the first 256 rows of the warm-start set) and with the full one (warm-start set + pool), and the improvement

$$
\Delta_i=\mathrm{CRPS}\big(q_i^{\text{small}},y_i\big)-\mathrm{CRPS}\big(q_i^{\text{full}},y_i\big)
$$

is ranked. In expectation $\Delta_i$ is the excess with the small context minus the excess with the full one; the aleatoric term cancels, so $\Delta$ measures reducible error only.

**Construction.** Same datasets, splits and seeds as T3. Scores from the small-context forward only: $\hat E_i$ against $e(q_i)$, the total-uncertainty baseline. Curve and PRR as in T3, with $\ell_i$ replaced by $\Delta_i$ (oracle: sort by $\Delta_i$). Outputs as in T3. $\Delta_i$ can be negative for single points, and it is a difference of two noisy losses, so absolute PRR values will be low (§3.5); the comparison between the two scores is what counts. If the Dec-ICL checkpoints arrive, the same test runs on its own forecasts with its native $v_{\text{epi}}$ against $v_{\text{tot}}$, and the cross-model comparison is how much each decomposition gains over its own total, as in T5.

### 3.8 T5 — Active learning

We reuse the active-learning benchmark of Bergna et al. (2026). It does not check whether the decomposition is *correct*, only whether the epistemic part *helps decisions*, which is what Phase 2 relies on. Five public tabular regression datasets (California Housing, Kin8nm, NYC Taxi, Airlines, YearPredictionMSD); initial labelled set $n_0=1024$, unlabelled pool of 5000, test set of 3000, 16–20 seeds with splits shared across methods (and reused by T3, T4 and T6); acquire 1024 points one at a time, with no training (context retrieval only). Metrics: RMSE, MAE, Gaussian NLL and CRPS on the test set, plus learning curves.

| Method | Acquisition score |
|---|---|
| Random (TabICLv2 predictor) | uniform |
| TabICLv2 | total entropy $e(q_\theta)$ |
| iLTMv2 | $\hat E_{\text{dir}}$ |
| Tuned-ICL | total variance (as in the paper) |
| Dec-ICL | latent / epistemic variance (as in the paper) |

TabICLv2 vs iLTMv2 is our matched pair; Tuned-ICL vs Dec-ICL is theirs. Tuned-ICL and Dec-ICL run on our splits if the authors share the checkpoints, otherwise their reported numbers are used as context only (they were evaluated on unreleased splits, so there is no exact reproduction target).

> **TBD — dataset sources.** Exact OpenML IDs and versions (the paper does not give them).

> **TBD — HPO/BO tests.** Left out: the ranking tests, active learning and context retrieval are the most informative tests for Phase 2.

### 3.9 T6 — Context retrieval

**Question.** For one query and a fixed starting context: given an extra labelled pool and a budget of $N$ rows, which selection rule gives the best forecast at that query? It compares $\hat E$-guided selection with k-NN and random selection. It is the prelude to Phase 2 (§5.4): the objective of §5.2 solved greedily, at inference only, on a Phase 1 checkpoint with no extra training.

**Set-up.** The datasets and splits of T3 (starting with one dataset, 5 seeds); starting context $S_0$ = the first 256 rows of the warm-start set; the 5,000-row pool with labels known; 200 test queries; budgets $N\in\{16,64,256\}$.

| Method | Rule |
|---|---|
| Start only | $S_0$, no additions (reference) |
| Random | $N$ random pool rows, the same for all queries |
| k-NN | The $N$ pool rows nearest to $x_*$ in standardized $X$ |
| $\hat E$-greedy | Below |
| $e(q)$-greedy | Same procedure with $e(q)$ as the score (control: is the gain due to the epistemic head or to the greedy search?) |
| Whole pool | $S_0$ + all 5,000 rows (reference) |

**$\hat E$-greedy.** 8 rounds, adding $b=N/8$ rows per round. In each round, for each query: build a shortlist of 64 candidates from the remaining pool (the 32 nearest to $x_*$ and 32 drawn uniformly at random); for each candidate $c$, read $\hat E_{\text{dir}}(x_*,S\cup\{c\})$; add the $b$ candidates with the lowest value to $S$. About 500 small forwards per query at any budget. The shortlist is mixed because a nearest-only shortlist would let $\hat E$ do no more than reorder k-NN's picks, contradicting the reason for the test (informative rows may be spread out, not near; §5.2), while a random-only one would almost never contain the nearest rows and would handicap $\hat E$ against k-NN, which sees the whole pool.

**Metric.** Realized $\mathrm{CRPS}(q,y)$ at the query, averaged over queries, per budget, paired across methods; never $\hat E$ itself. Reported alongside: the share of selected rows from the random half of the shortlist (near zero means $\hat E$ behaves like k-NN, a finding in itself), and exhaustive scoring of the whole pool for 20 queries, to show what the shortlist loses.

**Resolution check before building.** Adding one row to 256 may move $\hat E$ by less than the head can resolve. For a few queries, look at the spread of $\hat E(x_*,S\cup\{c\})$ over candidates; if it is flat, score blocks of 8 rows instead of single rows.

**Cautions, stated with the results.** (1) *Selection sees the labels*: the heads' targets are valid because every training split rule looks at $X$ only (§4.9), whereas greedy selection uses each candidate's label through the forward pass, so $\hat E$ can be biased low on the chosen context; this is why the metric is the realized loss. (2) *The whole-pool line may win*: TabICLv2 takes tens of thousands of context rows, so a small budget is artificial at these dataset sizes. (3) The checkpoint must be one trained with the subsampled stage-1 pass (§4.9), which shows the heads how the excess of the same query changes with the amount of context.

### 3.10 Models and their converters

| Model | How it runs | $\hat A$ | Total $e_{\text{tot}}$ |
|---|---|---|---|
| **Heteroscedastic GP** (PyMC) | Two latent GPs ($\mu(x)$ and $\log\sigma(x)$, the latter with a learned constant mean), sparse GP (40 inducing points) + NUTS per dataset; $S$ posterior draws | $\frac1S\sum_s\sigma_s/\sqrt\pi$ | entropy of the $S$-Gaussian mixture |
| **Dec-ICL** (Bergna et al., 2026) | Checkpoint requested from the authors; latent-signal head (bins over $f$) + aleatoric variance $\hat\sigma^2$, convolved | $\hat\sigma/\sqrt\pi$ | $e_{\mathrm{CRPS}}$ of their observation-level predictive |
| **TabICLv2** | Released checkpoint | — | $e(q_\theta)$ from the 999 quantiles only |
| **iLTMv2** | This thesis (Phase 1), the selected checkpoints of §4.10 | head $\hat A$ | $e(q_\theta)$; epistemic from head $\hat E_{\text{dir}}$ |

In all rows $\hat E=e_{\text{tot}}-\hat A$, except iLTMv2, which predicts its epistemic term directly (§4.4). Dec-ICL's native epistemic signal is the variance of its latent head, $v_{\text{tot}}=v_{\text{epi}}+\hat\sigma^2$ (Bergna et al., 2026, App. C.1); a variance cannot be drawn against a CRPS truth, so in T1–T2 its members $\mathcal N(c_j,\hat\sigma^2)$ are projected as above, and its native $v_{\text{epi}}$ is used where only ranking matters (T4, T5). Dec-ICL is fine-tuned from TabICL (Qu et al., 2025) on the authors' own heteroscedastic prior, so any Dec-ICL vs iLTMv2 gap mixes method, backbone and prior; the thesis says so.

**Which model goes in which test:**

| Model | T1 Toy | T2 Heatmap | T3 PRR-A | T4 PRR-B | T5 AL | T6 Retrieval |
|---|---|---|---|---|---|---|
| Heteroscedastic GP | ✓ | ✓ (10 toys) | — | — | — | — |
| Dec-ICL | ✓ | ✓ | if checkpoints | if checkpoints | ✓ (or reported numbers) | — |
| Tuned-ICL | — | — | if checkpoints (total only) | — | ✓ (or reported numbers) | — |
| TabICLv2 | backbone check | — | — | — | ✓ | — |
| iLTMv2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Random (TabICLv2 predictor) | — | — | — | — | ✓ | — |

In T3, T4 and T6 the total-uncertainty baseline $e(q_\theta)$ is read from iLTMv2's own forecast (the frozen TabICLv2 PPD on the heads' forward route), and the random and oracle curves and the random / k-NN selections are references, not models.

### 3.11 Deliverables

Toy-set figures for the paper (with the TabICLv2 backbone check as motivating figure); the error heatmap of every model; the PRR-A and PRR-B curves and tables; active-learning results; the context-retrieval curves; and the reusable testbeds and standardisation layer that Phase 1 is selected and evaluated on.

---

## 4. Phase 1 — Model training

### 4.1 What this phase does

1. Establish the **loss function** and **prior generator** that make the decomposition trainable.
2. Design the **architecture** of the $\hat A$ and $\hat E_{\text{dir}}$ heads: two arms, A and B (§4.8).
3. **Train** both heads on TabICLv2 over a sweep of learning rates and loss exponents (§4.9).
4. **Select** the final model with the validation bank and the Phase 0 tests, and **evaluate** it on every Phase 0 test (§4.10).

The full design lives in `experiment1_uq_heads/phase1_design.md`; how it is built and run lives in `experiment1_uq_heads/phase1_protocol.md`.

### 4.2 Supervision targets

Both targets come from §2.4 and need only $R\ge2$ i.i.d. draws of $y$ at each query from the true task law (training uses $R=16$, validation $R_{\text{val}}=128$):

$$
\hat t_A=\frac{1}{R(R-1)}\sum_{r<s}\lvert y^{(r)}-y^{(s)}\rvert, \qquad
\hat t_E=\frac1R\sum_r S\big(q_\theta,y^{(r)}\big)-\hat t_A .
$$

Both are exactly unbiased, so a squared-error loss converges to $A$ and to $E(q_\theta)$ respectively. The PPD is frozen (§4.4), so the replicates are used only by the UQ heads. $e(q_\theta)$, $S(q_\theta,y^{(r)})$ and the Phase 0 truth all use the same CDF built from the quantiles, otherwise $\hat r$ (§4.7) picks up a constant offset.

A lot of inspiration comes from the method and results given in Lahlou et al.’s (2021) paper.

### 4.3 The prior: exact replicates by an observed cut

#### Why the original prior cannot be used directly

In `graph_scm` (Qu et al., 2026), random vectors are sampled at root nodes and propagated through a random DAG. Each feature reads **a subset of a node's dimensions**; the unread dimensions stay unobserved, and *that* is the prior's only source of noise (no node-level Gaussian noise is added). Many node functions are dense maps that blend all input coordinates, so the hidden dimensions that create the noise are entangled with the features. There is no clean way to "freeze $x$ and rerun with fresh noise".

**Formally.** Write $X=\Phi_\theta(U)$ and $y=\Psi_\theta(U,V)$, where $U$ is randomness shared between $X$ and $y$ and $V$ reaches $y$ only. An exact replicate requires first drawing $U$ from the *abduction posterior* $p_\theta(U\mid\Phi_\theta(U)=x)$ (Balke & Pearl, 1994), which for arbitrary nonlinear graphs may be intractable. Freezing the root dimensions that are ancestors of $x$ under-reports the noise; using nearest neighbours as replicates gives a bias that grows with dimension. Both were rejected.

#### The observed cut

Keep the feature generator's mechanism, and make the observed $X$ the **only** frozen-side input to a second, `graph_scm`-style target graph with its own private noise:

$$
X=\Phi_\theta(U_1)\ \ (\texttt{graph\_scm}\text{ mechanism}), \qquad y=\Psi_\theta(X,U_2), \qquad U_1\perp U_2 .
$$

Then nothing has to be inferred from $X$: $p_\theta(y\mid X=x)$ is the law of $\Psi_\theta(x,U_2)$ with $U_2$ at its ordinary marginal. **Replication = hold $X$, redraw $U_2$, rerun the target branch.**

```
 U1 ──► [ X-graph: graph_scm mechanism ] ──► X (observed columns)
                    │ feature-hosting nodes
                    ▼
        [ X-input nodes: node function of their parent value = observed post-converter columns ] ─┐
                                                          (deterministic in X)                     ├─► [ Ψθ: graph_scm-style DAG ] ──► y
        [ private noise roots ] ── U2: the only randomness, redrawn per replicate ─────────────────┘
```

The target graph $\Psi_\theta$ copies `graph_scm` as closely as possible:

- **Same DAG sampler, size range and node functions** as the $X$-graph (Cauchy DAG, 2–32 nodes, tabicl's node functions); tabicl's pruning keeps only the ancestors of $y$.
- **$X$-input nodes are the $X$-graph's feature-hosting nodes.** A random subset of them (mirroring `subsample_feature_nodes`) is placed on random DAG positions whose incoming edges are cut. The **observed post-converter values** of the columns read from the host node, recomputed from the observed (outlier-clamped) columns, are the node's **parent value**, never part of a root: the node is an ordinary non-root `graph_scm` node (tabicl's `RandomNodeFunction`: a random multi-input function of the block into LogInt(1, 32) latent dims, then standardization, importance weights, L2 normalization and node importance). It is a deterministic function of $X$, so $y$ depends on $U_1$ only through $X$. For the two categorical modes whose write-back is a latent value (`neighbor_id`, `softmax_id`), the observed category's center or embedding is used instead.
- **Private roots.** Every other parentless node draws random points, as `graph_scm`'s roots do. They are $U_2$, the **only** randomness of the target graph: as in `graph_scm`, randomness enters only at roots (node noise off). Their noise is lowered, as calibrated by the difficulty-match gate below: outputs scaled by 0.1 and LogInt(1, 16) dims instead of LogInt(1, 32).
- **No noise-ancestor requirement.** Only the graph check below constrains the target graph. When no private root is an ancestor of $y$, $y$ is an exact deterministic function of $X$ ($A=0$), the cut's counterpart of `graph_scm`'s near-deterministic tasks (about 60% of accepted stage-1 tasks, 62% of the validation bank's tasks). The design, its calibration and its gate results are in `experiment1_uq_heads/phase1_design.md` (task construction, "The observed-cut prior").
- $y$ is read from a random non-input node with the usual converter; $y$-warping follows `disallow_y_warping`.

| | |
|---|---|
| **Preserved** | $X$'s generating mechanism (same DAG sampler, node functions, converters, warping and cross-feature correlations). $y\mid x$ stays a non-Gaussian, heteroscedastic pushforward built from the same function classes |
| **Not identical** | $X$'s distribution. (1) The $X$-graph is sampled with no $y$ node; in `graph_scm`, $y$ takes part in node assignment, pruning and the graph check, so the graphs and column placements that occur differ. (2) The predictability filter runs on the cut $y$, so accepted $X$ is conditioned on a different acceptance event. (3) `ensure_iid=True`, whereas TabICLv2 was pretrained with `ensure_iid=False`. The difficulty-match gate below checks that the shift is small in practice |
| **Given up** | (1) Latent confounding between $X$ and $y$. (2) Anti-causal features: in `graph_scm` features can be descendants of the $y$ node, while in the cut $X$ is never downstream of $y$. No Phase 0 test measures what these cost (§6, risks) |
| **Cost** | One extra pass of the target graph over the query rows gives one replicate for every query at once, so $R$ replicates cost $R$ passes of a small graph per dataset. The passes are kept separate (not one pass over rows tiled $R$ times) so that every replicate runs through identical float32 kernels and the $U_2$-independent part of $y$ is bit-identical across replicates |

**Implementation requirements.**

- Set `ensure_iid=True`, so that transforms are not fitted on the rows they transform.
- Cache task-level objects in `RandomNodeFunction._fit` (root point law, feature-importance weights, node importance). Only the private noise may be redrawn.
- Split post-processing into `fit` / `apply` and apply the observed dataset's constants to every replicate. Targets must live in the same units the model predicts in. Concretely, $y$'s transform (outlier clamp at mean ± 4σ, then standard scaling, as in TabICLv2) is fitted once on the full pool's $y$, before any split manipulation, and applied unchanged to every replicate. Clamped replicates are then i.i.d. draws from the clamped task law the PPD approximates, so $\hat t_A$ stays unbiased. Never refit per replicate and never clip the targets themselves.
- Fix `PYTHONHASHSEED` in every sampling job.
- Assert bit-identical $X$ across replicates. This is the only correctness guard the construction needs.

**Task filtering.** TabICLv2 pretrains only on tasks that pass its filters (about 25 % of stage-1 regression tasks are rejected), so our tasks pass the same filters, applied to whole tasks before the split and the replicates (a rejected task is redrawn from scratch):

- *Graph check*, adapted: applied to the **target graph**, resampled until at least one $X$ input is an ancestor of $y$ (the $X$-graph has no $y$ node).
- Non-finite target-graph inputs reject the task: an $X$-graph latent can be NaN while its observed column is finite (e.g. a category index), and `graph_scm` would reject that dataset too, since the NaN reaches its $y$.
- *Constant columns* removed; a task with no feature left is rejected.
- *Predictability test* as in TabICLv2: an ExtraTrees model must beat the mean-label baseline out-of-bag (p < 0.05).
- *Trivial-task test* off, so that the near-deterministic $A\approx0$ end is learned.
- Non-finite $y$ or replicate values reject the task.

The same filter is used for the training stream and the validation bank, and the rejection rate is logged per stage; if it is much higher than TabICLv2's, tune the target graph's noise scale, not the filter. The stage-1 training stream and the validation bank (30.6%) sit slightly above the gate's 30% bound; this is an accepted, documented deviation (`phase1_design.md`).

**Difficulty-match gate (prior level, before training).** Generate matched banks of cut tasks and unmodified `graph_scm` tasks (same row and feature counts, plain i.i.d. context–query split, no carved regions for this test only). On each task compute the frozen TabICLv2 forecast on the queries (CRPS, and $R^2$ of its median) and ExtraTrees held-out $R^2$ on the same split. Per stage the gate requires: (1) a cut rejection rate in 5–30%; (2) a matching difficulty profile, i.e. ExtraTrees $R^2$ quantiles q10–q90 within 0.10 of `graph_scm`'s; (3) an in-domain check: in equal-count bins of ExtraTrees $R^2$, the median of $R^2(\text{TabICLv2})-R^2(\text{ExtraTrees})$ agrees between the priors (within 0.05 or overlapping bootstrap CIs; top, saturated bin not gated), which tells a harder slice of the prior (same advantage of the backbone) from a slice outside its domain (advantage lost). CRPS quantiles and two-sample tests are reported, not gated. If the gate fails, **tune the target graph's private-root noise** (scale and number of dims), **not the filters**, and rerun. The configuration and tolerance used are recorded in the Phase 1 documents (`experiment1_uq_heads/scripts/00f_difficulty_match.py`).

**Known caveat.** TabICLv2's rejection filters discard near-unpredictable datasets, truncating the high-aleatoric tail the head is trained on. The empirical distribution of $\hat t_A$ is logged and reported.

#### Audits

- **Estimator audit (before training).** First, a hand-made prior $x=f(a)$, $y=g(x)+s(x)\,b$, with $b\sim\mathcal N(0,1)$ and $a\perp b$, where $e_{\mathrm{CRPS}}=s(x)/\sqrt\pi$ is known. Check $\hat t_A$ over the deterministic ($s=0$), homoscedastic ($s=\sigma$) and heteroscedastic ($s=e^{\alpha x}$) cases, averaging over a few hundred queries per level. Then run $\hat t_A$ on the observed-cut prior and report the zero-mass fraction, the spread of the non-zero part, and its variation across $x$ within a task. This helps us make sure that the aleatoric entropy estimator works as expected in a MC limit. Phase 0 runs the estimator checks once (on the toy rungs, plus the replicate estimator of $E$ against a deliberately wrong Gaussian forecast), so both phases can cite them.


### 4.4 The heads

$e(q_\theta)$ is not learned; it is computed in closed form from the quantile head (§2.3). Two scalar heads are added on top of the **frozen** TabICLv2 regressor. In arm A (§4.8) both read $h_*$, the representation the quantile decoder reads (the post-final-LayerNorm output of the ICL transformer at the query row, 512-d); in arm B they read the same quantity recomputed by a trainable fork:

$$
\hat A(x_*,D)=\mathrm{softplus}\big(a_\theta(h_*)\big),\qquad
\hat E_{\text{dir}}(x_*,D)=\mathrm{softplus}\big(b_\theta(h_*)\big).
$$

Each head has its own trunk, mirroring the quantile decoder: `Linear(512→1024) → GELU → Linear(1024→1) → softplus`. Separate trunks keep the noisier $\hat t_E$ gradients from degrading $\hat A$.

- **Why softplus.** It resolves values spanning orders of magnitude (the backbone check shows $E$ from $10^{-4}$ to $10^{0}$) and keeps squared-error gradients from exploding. The non-negativity is correctly specified: $\hat t_A\ge0$ and $\mathbb E[\hat t_E\mid x_*,D]=\mathbb E_{\tau\mid D}[d_S(q_\theta,p_\tau)]\ge0$. Its cost is a weak gradient on small outputs, which the loss exponent of §4.5 addresses.
- **No coupling to $e(q_\theta)$, and no gate.** A coupled form, $\hat A=e(q_\theta)\cdot\mathrm{softplus}(z)$ with a soft penalty $\mathrm{ReLU}(\hat A-e(q_\theta))^2$, is rejected. Coupling would (1) contaminate the amortization-gap estimate (§4.7) and (2) tie both heads to a value that does not always follow the true divergence: the backbone check shows $e(q_\theta)$ misses it in sparse-context intervals.
- **Units.** Both heads live in TabICLv2's internal z-normalized $y$ units; multiplying by the context s.d. gives raw units exactly, since CRPS entropy and divergence are positively 1-homogeneous.

**Why $h_*$ alone: the link to DEUP.** DEUP's error predictor needs *stationarizing features* of $(x,\text{dataset})$: a density estimate, a model-variance estimate and a seen-bit (Lahlou et al., 2023, §3). In a PFN the dataset conditioning is built in: $h_*$ is a function of $(x_*,D)$, since the query attends to every context row, so it is a learned proxy for all three (how many relevant rows were attended to; a near-exact match with a context row; the spread of the forecast, of which $e(q_\theta)$ is a deterministic function). Whether $h_*$ actually kept what the split needs is tested by the comparison of arms A and B (§4.8).

### 4.5 Loss and training

$$
\mathcal L=\overline{\,w\,\big(\hat A-\hat t_A\big)^2\,}+\overline{\,w\,\big(\hat E_{\text{dir}}-\hat t_E\big)^2\,},\qquad w=\big(\mathrm{sg}[e(q_\theta)]+c\big)^{-p},\qquad p\in\{0,1\},
$$

averaged over query rows, with one value of $p$ per head pair. The pinball loss is absent because the PPD is frozen, and the heads share no parameters.

- **Stop-gradient** on $q_\theta$ inside $\hat t_E$ and inside $w$, so the target and the weight are fixed functions of the frozen forecast.
- **Why a weight is safe.** A weight depending **only on $(x_*,D)$** leaves the minimiser at the conditional mean of the target; it only changes how finite capacity trades errors across inputs.
- **$p=0$:** every query counts equally in absolute units, so large-value queries dominate (target noise also scales with the quantity itself).
- **$p=1$:** low-spread queries are up-weighted in proportion to $1/e(q_\theta)$. With softplus, a small output responds to its input in proportion to the output itself, so the squared-error gradient at a query is about $2(\hat A-\hat t)\,\hat A$ and small values are almost never corrected under $p=0$. On the low end the output is about $e(q_\theta)$, so a weight of $1/e(q_\theta)$ gives that factor back. It matters wherever the low end is read: the epistemic fraction $\lambda$ (§4.6), $\hat E$ on a log axis (T1), and $\hat A\approx0$ on toy rung 0.
- **The floor $c$ is fixed at 0.1** (z-normalized $y$ units) and stored in the validation bank's manifest. The largest weight is $1/c=10$, queries with $e(q_\theta)$ well below 0.1 share roughly that weight (about 6 times that of a typical noisy query), and $w$ runs from 10 down to about 1.7. The bank median of $e(q_\theta)$ is not used as $c$: $e(q_\theta)$ over the bank is bimodal (q25 $5\cdot10^{-4}$, median $0.013$, q75 $0.25$; 62% of the bank's tasks are deterministic), the median falls in the gap, and 88% of the queries below it come from deterministic tasks. With $c=0.013$ the largest weight would be 79, about 35 times a typical noisy query's, so the deterministic tasks would dominate the $p=1$ loss.
- **Both values are trained.** The validation metrics cannot choose between them, because each weighting favours the run trained with it; the choice is made with the Phase 0 tests (§4.10). $p=1$ pairs start (output bias) at the weighted mean target, the best constant under their loss.
- **No forecast degradation** by construction: for arm A the PPD is bit-identical to TabICLv2, arm B's fork feeds only the heads, and no $\lambda_E=0$ warm-up is needed.
- $\hat E_{\text{dir}}$ is **checkpoint-specific**: its target moves with $q_\theta$, so it must be re-fitted after any backbone change. $\hat A$ has no such dependence.
- Squared error only: no log-space regression, no quantile or absolute-error loss, and **never clip targets at zero**. Each of these breaks the conditional-mean argument of §2.4.

> **TBD — remaining training-recipe questions.** Real-data continued pretraining (RealPFN-style fine-tuning) and its ordering relative to head training. Unfreezing is settled by the arms of §4.8.

### 4.6 What a user reads off

| Quantity | Meaning | Units |
|---|---|---|
| $e(q_\theta)$ | How spread out is the forecast? | units of $y$ |
| $\hat A$ | How much of it would remain with infinite data from this task? | units of $y$ |
| $\hat E_{\text{dir}}$ | How much score would I save if I knew the true task law? | score units |
| $\lambda=\hat E_{\text{dir}}/(\hat A+\hat E_{\text{dir}})$ | Epistemic fraction: share of the forecast's expected score that is reducible; scale-free, for comparison across datasets | — |

> **Note on $\lambda$.** Since $\hat A+\hat E_{\text{dir}}$ estimates the expected score of $q_\theta$ (§2.2), $\lambda\in[0,1]$ holds by construction, even when $\hat E_{\text{dir}}$ exceeds $e(q_\theta)$ because the PPD is off. The alternative $\lambda=\hat E_{\text{dir}}/e(q_\theta)$ is rejected because it does not stay in $[0,1]$.

### 4.7 Amortization error — TBD

Two epistemic estimates are available: the subtraction $e(q_\theta)-\hat A$ and the direct $\hat E_{\text{dir}}$. Their difference

$$
\hat r=\big(e(q_\theta)-\hat A\big)-\hat E_{\text{dir}} \;\longrightarrow\; e(q_\theta)-e(q)-d_S(q_\theta,q)
$$

vanishes when the network's PPD equals the ideal PPD $q$, so it measures how far the amortization is from exact Bayes at that query. It is logged for every iLTMv2 run, and the uncoupled head design (§4.4) keeps it uncontaminated.

> **TBD:** whether $\hat r$ is reported, how it is used (diagnostic, flag, evaluation axis), and whether it is useful at all.

### 4.8 Architecture — the two arms

Pinball pretraining only asks $h_*$ to encode the posterior *predictive* (a mixture of task laws), whereas the decomposition needs the posterior *over tasks* (Bergna et al., 2026, Prop. 1). Nothing forced $h_*$ to keep what tells "a mix of sharp laws" from "one wide law". Two arms measure how much that matters: A reads $h_*$ as it is, B gives the heads trainable access to the backbone's last blocks, at a higher cost. Whether B's extra access buys anything is itself a finding about what PFNs keep when they amortize the posterior.

| Arm | What the heads read | Backbone |
|---|---|---|
| **A** (default) | $h_*$ through the MLP trunks of §4.4. Tests whether the split is present in $h_*$ | Fully frozen; PPD identical to TabICLv2. Nothing upstream of the heads trains, so one job's backbone pass feeds many head pairs, and the validation bank's $h_*$ is cached once per job |
| **B** | A trainable copy of the last $k=2$ ICL-transformer blocks and final LayerNorm, initialised from the pretrained weights, re-attending to the context (all rows) under the UQ loss and feeding only the heads | Originals stay frozen and keep feeding the decoder, so the PPD is bit-identical and B starts exactly where A ends. Unlike Bergna et al., who unfreeze shared layers and change the PPD. Costs about $k/12$ extra compute in training and at every inference; no penalty pulls the fork towards its pretrained weights |

**Comparing arms needs no ground truth.** On held-out cut-graph tasks, $\mathbb E[(\hat A-\hat t_A)^2]=\mathbb E[(\hat A-A)^2]+\text{target-noise variance}$, and the noise term is identical across arms, so a lower validation loss *is* a lower true error. The final choice between arms is made with the Phase 0 tests (§4.10).

> **Ablation (only if the first experiments fail).** Append one stop-gradient scalar, the standardised $\log e(q_\theta)$, to the trunk input. It adds no information but hands the trunk DEUP's variance feature ready-made.

### 4.9 Training procedure

Following TabICLv2's meta-training logic: one training example is a whole synthetic task; an infinite stream of fresh tasks, a fixed step budget, **no early stopping**, and a fixed validation bank used only to *compare* runs and checkpoints.

**Task construction.** The split is built on $X$ first and $y$ is drawn last. Each task mixes three query types, built from an enlarged pool so that the context sees the carved regions only through the rows deliberately returned to it:

| Query type | Share | Construction |
|---|---|---|
| i.i.d. | 70 % | TabICLv2's ordinary random split |
| kNN hole | 15 % | Rows nearest a random centre are carved out of the context; queries are drawn from them |
| Extrapolation | 15 % | Rows extreme along a random direction in standardised $X$ space are carved out of the context; queries are drawn from them |

Hole and tail sizes are drawn per task (log-uniform, up to $0.2N$), so queries cover a range of divergences. In addition, 5 % of query slots get their $x$ overwritten by a context row's $x$ (a fresh $y$ is drawn), teaching the low-divergence end (cf. DEUP's seen-bit); the effective mix is 66.5 / 14.25 / 14.25 / 5.

**Thinned regions.** A context is rarely either full or empty around a query: regions with a few rows are where "is this spread noise or lack of data?" is hardest, and they are what context retrieval and active learning operate on. So each carved region (the extrapolation tail and the kNN hole) is independently **empty** (none of its rows enters the context) or **thinned** with probability ½ each: a quarter of the tasks have both regions empty, a quarter only the tail thinned, a quarter only the hole thinned, a quarter both thinned. A thinned region returns

$$
r=\max\big(1,\operatorname{round}\big(\rho\,(\text{region size}-\text{its queries})\big)\big),\qquad \rho\sim\text{log-uniform}[0.05,0.2],
$$

of its non-query rows, chosen uniformly, to the context (log-uniform because the effect of density is roughly multiplicative). A region with no non-query row is always empty. All sizes are drawn before generation, so with tail size $T$, hole size $k$ and query budgets $m_{\text{ext}}$, $m_{\text{hole}}$ the pool size

$$
P=N+(T-m_{\text{ext}}-r_{\text{ext}})+(k-m_{\text{hole}}-r_{\text{hole}})
$$

is known in advance, and the i.i.d. remainder supplies $n_{\text{ctx}}-r_{\text{ext}}-r_{\text{hole}}$ context rows, so the context is exactly $n_{\text{ctx}}$ rows and the queries exactly $m$. Queries carry one of six types: `iid`, `hole`, `hole_thinned`, `extrapolation`, `extrapolation_thinned`, `duplicate` (a region with returned rows gives the `_thinned` type).

Every split decision (carving, thinning, duplicates, subsampling) depends only on $X$ and on random numbers independent of $y$, so the posterior over the $y$-law given the manipulated context is unchanged and the targets keep their meaning; selecting on $y$ would break this and is never done. The mixture only decides where head capacity is spent, not the minimiser.

**Subsampled pass in stage 1.** Stage-1 contexts are 30–90% of 1,024 rows, so about 307 rows at least, while T1, T4 and T6 use 64–256. So in stage 1 every task is used twice in its step: the **full pass** (the task as built) and a **subsampled pass**, the same task with its context reduced to $n'$ rows, $n'$ log-uniform in $[64,n_{\text{ctx}}]$, one $n'$ per micro-batch group, rows dropped uniformly at random (which keeps the relative density of thinned regions); queries and replicates are unchanged. $\hat t_A$ is identical in both passes; $\hat t_E$ is recomputed from the subsampled forecast. In the subsampled pass, a duplicate whose source row was dropped takes its source row's type (`iid`, or its region's type), and a thinned region with no returned row left becomes empty. A stage-1 step is 32 tasks and 64 forwards, and the loss averages over the query rows of both passes. Besides covering small contexts, the pass shows the heads the same query under two contexts of the same task, i.e. how the excess changes with the amount of context. Two things to know: below about 307 rows TabICLv2's forecast is off its pretraining range, and $\hat E_{\text{dir}}$ is trained on the frozen forecast as it is, so it should learn the larger excess; and the subsampled pass has a context fraction below 30%, which is harmless because in TabICL a query attends to context rows only (a gate checks that a query's forecast is the same alone or among $m$ queries). Stages 2 and 3 have one pass per task.

**Two task priors.** *Carved* is the design above, used by every sweep run. *i.i.d.* is TabICLv2's plain random split only (no tail, hole, thinning or duplicates; $P=N$), trained only for the winning configurations' ablation (§4.10). Both use the subsampled pass, so the ablation isolates the carving. The i.i.d. prior draws the same task distribution, not the same tasks.

**Curriculum** (TabICLv2's three stages), batch of 32 tasks (half of TabICLv2's 64, to halve the wall time per step; gradient accumulation in stages 2–3), AdamW with a cosine schedule restarted per stage (1% warm-up):

| Stage | Steps | Rows $N$ per task | Context fraction | Head LR | Fork LR (B) |
|---|---|---|---|---|---|
| 1 | 30,000 | 1,024 | 30–90 %, plus the subsampled pass (down to 64 rows) | `lr` | `fork_lr` |
| 2 | 8,000 | 400–10,240 | ≈80 % | `lr`/8 | `fork_lr`/8 |
| 3 | 1,200 | 400–60,000 | ≈80 % | `lr`/40 | `fork_lr`/40 |

Stage 1 has 30,000 steps because in earlier runs more steps did not give a significant improvement. The run totals 39,200 global steps.

**Validation.** A fixed bank of 2,000 tasks, built with the carved prior (thinning and duplicates included), $R_{\text{val}}=128$ replicates, seed disjoint from every test:

| Bin | Rows $N$ | Context fraction | Tasks |
|---|---|---|---|
| Small | Uniform in 256–1,024 | exactly 80 % | 1,000 |
| Medium | 1,024–10,240 (log-uniform) | 79–81 % | 600 |
| Large | 10,240–60,000 (log-uniform) | 79–81 % | 400 |

Every run, carved or i.i.d., is validated on this same bank. The MSE, xMSE (replicate noise removed) and skill of $\hat A$ and of $\hat E_{\text{dir}}$ are logged separately, per query type (the six above) and per context-size slice (`n_le256`, `n_256-512`, `n_512-1K`, `n_1K-4K`, `n_4K-16K`, `n_16K-60K`), under **both weightings** ($p=0$, and $p=1$ with the weighted mean target as constant baseline), whatever the run was trained with, so $p=0$ and $p=1$ runs compare fairly on either. Reference lines (noise floor, constant baseline, and the **e(q) proxy**, the best predictor that sees only $e(q_\theta)$) are computed for both weightings. On the bank the e(q) proxy reaches skill 0.70 on the aleatoric target but only 0.12 on the epistemic one: the forecast's spread tracks the noise, not its own error, which is the gap the heads must fill. The full bank is evaluated at global steps 5,000, 10,000, …, 30,000, 34,000, 38,000, 38,600 and 39,200 (the middle and end of stages 2 and 3), and a 64-task mini-bank (the first 64 small tasks) every 250 steps. Each head pair keeps its latest checkpoint and its best-$\hat A$ and best-$\hat E_{\text{dir}}$ checkpoints, chosen under the pair's own weighting (checkpoint selection, not early stopping). In B the heads share the fork, so both heads are logged at both checkpoints to expose trade-offs.

**Sweep** (seed 0; the seed fixes both task stream and head initialisation). Head LR $\in\{3\cdot10^{-4},10^{-3},3\cdot10^{-3}\}$ for both arms; for B also fork LR $\in\{10^{-5},3\cdot10^{-5},10^{-4}\}$; $p\in\{0,1\}$ for both. All are stage-1 values; stages 2–3 follow the fixed ratios of the table above.

| Arm | Head LR | Fork LR | $p$ | Prior | Configurations |
|---|---|---|---|---|---|
| A | 3e-4, 1e-3, 3e-3 | — | 0, 1 | carved (i.i.d. for the winner) | 6 |
| B | 3e-4, 1e-3, 3e-3 | 1e-5, 3e-5, 1e-4 | 0, 1 | carved (i.i.d. for the winner) | 18 |

**Job layout.** In A nothing upstream of the heads trains, and $p$ only changes a head's loss, so one job per (prior, seed) runs the backbone once per step and feeds all six independent head pairs, each with its own optimizer and schedule (the backbone forward is almost all of the compute). B needs one job per configuration, since the fork changes the forward pass and $p$ changes the fork's gradients: 18 jobs. One job runs stages 1 → 2 → 3 in sequence. The A job runs first; the B jobs follow once the Phase 0 tests are written and their baselines run, since nothing in between depends on them.

### 4.10 Selection and evaluation

1. **Learning rates, from validation.** Within each (arm, $p$, prior) cell, the LR is picked by the pooled validation MSE under that cell's own weighting: per head in A (the heads are independent), as a whole configuration in B (sum of the two heads' MSE, each relative to A's at the same $p$).
2. **Final model, from the Phase 0 tests.** The four carved winners (A and B, $p=0$ and $p=1$) are run on T2–T5, and the best is the final model. $p=1$ is expected to win, since it improves small-value estimates. The final model is then run on T1 and T6.
3. **Seeds and ablation (appendix).** The winning configurations are rerun on seeds 1 and 2 to add error bars. The final model is compared with three counterparts, each differing in one factor (the other $p$, the other arm, the i.i.d. prior), on T2–T5. Learning rates are not ablated.

iLTMv2 is run on **all Phase 0 tests**.

---

## 5. Phase 2 — Epistemic-guided context retrieval

### 5.1 What this phase does

1. Establish a method for epistemic-guided context retrieval.
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

Two Phase 0 tests are already context selection without training, and they are the first, cheapest checks of whether $\hat E_{\text{dir}}$ selects useful rows:

- **T5, active learning** (§3.8): a global choice of which pool point to label next, scored on the whole test set.
- **T6, context retrieval** (§3.9): the per-query objective $S^\star$ of §5.2, solved greedily on a mixed shortlist, at inference only. It is a first version of Phase 2's inference-only arm, and it informs the open design of §5.3: its cost (about 500 small forwards per query) and the resolution check bound the search strategy and the batched scoring; the share of picks from the random half of the shortlist says whether $\hat E$ goes beyond k-NN; the $e(q)$-greedy control separates the head from the search; and the gap between $\hat E$ on the chosen context and the realized loss (selection sees the labels) is a direct measure of the "confidently wrong" risk the safeguards must handle.

---

## 6. Cross-cutting

**Benchmarks in scope:** ScoringBench (Landsgesell et al., 2026), TALENT (Liu et al., 2024), TabArena (Erickson et al., 2025).

**Dependencies.**

```
Phase 0 ──► Phase 1 ──► Phase 2
  │  T2–T5 choose Phase 1's final model (arm, p)
  └─ T5 (active learning) and T6 (retrieval, on a Phase 1 checkpoint) preview Phase 2
```

**Out of scope:** a quasi-martingale coherence statistic (to directly estimate the quality of our aleatoric and epistemic entropies), prior-support scoring and directed prior expansion with the coherence statistic, and epistemic-conditioned conformal calibration.

**Open items (consolidated).**

1. Dec-ICL and Tuned-ICL checkpoints (authors contacted), Dec-ICL output format and backbone version (§3.10).
2. T5 dataset sources (§3.8).
3. Remaining training-recipe question: real-data fine-tuning and its ordering (§4.5).
4. Use of the amortization error $\hat r$ (§4.7).
5. The Phase 2 method, pipeline, metrics and safeguards (§5.3).
6. Kill criteria per phase.

**Risks and doubts** (carried, not decided).

1. **The validation bank barely sees the smallest contexts.** Its smallest context is about 205 rows (256 rows at 80%): only 0.2% of its queries are at $n\le256$, and 64% are in the 16K–60K slice. Training goes down to 64 rows and T1 uses 64, so the effect of the subsampled pass is nearly invisible in validation. Drawing the small tasks from 80 rows up (context 64) would cover it.
2. **Model selection on the test data.** The final model and $p$ are chosen with T2–T5, and the same tests are then reported, which is mildly optimistic. Either say so, or select on a fixed subset (for example T2 plus one dataset) and report on the rest.
3. **Small contexts may be forgotten in stages 2–3.** The subsampled pass exists only in stage 1, and stages 2–3 train on large contexts. Watch `n_le256` and `n_256-512` at global steps 34,000–39,200; if they degrade, mix a share of stage-1-shape tasks into stages 2–3.
4. **The top learning rates.** Head LR 3e-3 may be unstable. Fork LR 1e-4 has no penalty pulling the fork towards its pretrained weights; watch its distance from the initial weights (`fork_dist_from_init`) in the first hours of the B jobs.
5. **No ground-truth test beyond 1-D.** The only "is the number right" evidence in Phase 0 is 1-D with Gaussian noise (T1, T2), and what the observed cut gives up (latent confounding, anti-causal features; §4.3) is not measured by any test. A reliability plot from the validation bank costs nothing and could sit in the appendix.
6. **PRR-A may show a small gap** (§3.6): $\hat A+\hat E$ and $e(q)$ differ only where the backbone is miscalibrated, and on ordinary splits of these five datasets the difference could be within noise.
7. **PRR-B and retrieval depend on how sensitive $\hat E$ is to the context.** Both are constructions with no published reference numbers. In PRR-B the contrast of 256 against 6,024 rows is a choice, and the result may depend on it; in T6 the resolution check (§3.9) decides between single rows and blocks.

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
