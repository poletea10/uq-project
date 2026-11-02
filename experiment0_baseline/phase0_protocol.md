# Phase 0 — Setting the baseline: execution protocol

Phase 0 has four tests and trains nothing. iLTMv2 rows stay visibly empty until Phase 1 delivers the model.

| # | Test | Question | Ground truth | Models | Output |
|---|---|---|---|---|---|
| T1 | **Toy set** (toy rungs 0–2) | *Where* does each model put $\hat A$ and $\hat E$ on one curated 1-D function? | Closed form | Heteroscedastic GP, Dec-ICL, iLTMv2 | One 3×4 context figure per (model, toy rung): 9 figures, plus one 3×3 paper figure |
| T2 | **Cut-graph test** (in distribution) | Are $\hat A$ and $\hat E$ numerically right on the prior iLTMv2 is trained on, including in context holes? | Exact replicates (observed cut) | Dec-ICL, iLTMv2 | Reliability plots + table |
| T3 | **Transfer test** | Do they stay right on the unmodified, fully confounded `graph_scm`? | Exact-match replicates on low-cardinality categorical features | Dec-ICL, iLTMv2 | Reliability plots + table |
| T4 | **Active learning** | Does the epistemic estimate select useful data? | Test-set performance | Random, TabICLv2, iLTMv2, Tuned-ICL, Dec-ICL | Table + learning curves |

T1 is for the paper's main figure: easy to read, no metric table. T2 and T3 carry the numbers. T4 is the usefulness check and the first preview of Phase 2.

> **Changes from the previous protocol, for anyone reading old code** (previous version: `/private/home/ajaras/iLTM-bio/UQ_project/SCRAPPED/v2/experiment0_baseline`).
> - Rungs 0–3 over a *distribution* of low-rank GP functions are replaced by the toy set: **one curated function**, three noise settings, shown as figures rather than graded with metrics. The old seed-averaged reliability scheme survives only as an optional appendix figure (§2.8).
> - Old rung 4 becomes the **cut-graph test** (T2), now with **context-hole queries**.
> - New: the **transfer test** (T3), previously an open item ("rung 5").
> - Baselines reduced to the **heteroscedastic GP** (toy set only) and **Dec-ICL** (Bergna et al., 2026). CatBoost, UD4PFN and DER are dropped, along with their adapters.
> - Tuned-ICL and Dec-ICL are **run** in active learning if the authors share the checkpoints; otherwise their reported numbers are used as context.
> - The TabICLv2 motivating figure is folded into the **backbone check** (§2.7).

---

## 1. Layout, environments and shared constants

```
experiment0_baselines/
├── CLAUDE.md
├── configs/            # protocol.yaml, toy_function.yaml, toy_rung{0,1,2}.yaml,
│                       # cutgraph.yaml, transfer.yaml, al.yaml
├── src/
│   ├── toy.py          # the curated function f, noise settings, context sampler
│   ├── truth.py        # realized A and E: closed form (T1), replicate-based (T2, T3)
│   ├── entropy.py      # e_CRPS kernel, excess kernel, shared t-grid builder
│   ├── adapters/       # base.py (UQOutput), hgp.py, decicl.py, tabiclv2.py, iltmv2.py
│   ├── cutgraph.py     # T2 tasks, pools, context holes, replicates
│   ├── transfer.py     # T3 tasks, categorical pools, held-out cells, exact-match replicates
│   ├── grading.py      # truth-on-estimate OLS, bootstrap, reliability plots (T2, T3)
│   ├── figures.py      # 3×4 context figures (T1, backbone check), 3×3 paper figure (T1)
│   └── active_learning.py
├── scripts/            # 00_curate_toy, 00a_backbone_check, 00b_validate_estimators, 00c_validate_adapters,
│                       # 01_toy (--method, --paper-figure), 02_cutgraph (--method), 03_transfer (--method),
│                       # 04_active_learning (--method --dataset)
└── outputs/{raw,figures,tables}/
```

**Environments.**

- `uq-iltm-bio` for everything by default, including the heteroscedastic GP (PyMC with the numpyro backend).
- Dec-ICL gets whatever environment its checkpoint needs, with the same parquet hand-off if it clashes with `uq-iltm-bio`. Never import a clashing env in-process.

**Recyclable code.** The previous version already has a working heteroscedastic GP adapter, the 3×4 context figures, the entropy kernels, the low-rank GP generator with its rung YAMLs, and the reliability-plot evaluator. Reuse them where this protocol agrees with them; where it does not, this protocol wins.

**Shared constants** (`protocol.yaml`, fixed before the first run, never tuned):

| Constant | Value |
|---|---|
| `n_grid` | 16, 64, 256, 1024 (context sizes; T1, T2, T3) |
| `S` | 500 posterior samples (heteroscedastic GP) |
| `R_rep` | 100 replicate draws of $y$ per query (T2); cap on replicates per query (T3) |
| `n_bins` | 20 equal-count bins for reliability plots (T2, T3) |
| `sigma_floor` | $10^{-6}$. Clip value for every predicted σ (§10, gotcha 3); $\sigma_{\text{floor}}/\sqrt\pi$ is the display floor of log-scale plots |
| `base_seed` | 0. Root of every RNG stream (§9) |

---

## 2. T1 — Toy set

### 2.1 Purpose

One fixed 1-D function, one fixed context draw, three noise settings. Each model gets one figure per setting, showing along $x$ where it puts aleatoric and epistemic uncertainty against the exact truth. It follows the spirit of the toy figures in DEUP (Lahlou et al., 2023, Fig. 1 and App. E) and DecoupledPFN (Bergna et al., 2026, Fig. 1), but uses the context-figure layout of §2.6 rather than DEUP's shaded panels.

It is an illustration, not a grade: a single function cannot support calibration claims. The quantitative grading lives in T2 and T3.

### 2.2 The curated function

Everything lives on `x_range` $=[-5,5]$:

$$
y=f(x)+\sigma(x)\,\varepsilon,\qquad \varepsilon\sim\mathcal N(0,1),\qquad f(x)=f_{\text{base}}(x)+f_{\text{gap}}(x).
$$

**Base part.** The low-rank GP of the previous protocol (Rasmussen & Williams, 2006; Kang, 2021):

$$
f_{\text{base}}(x)=\sum_{j=1}^{J}\theta_j\,b_j(x),\qquad b_j(x)=\exp\!\Big(-\frac{(x-c_j)^2}{2\ell^2}\Big),\qquad \theta\sim\mathcal N(0,s_\theta^2 I_J).
$$

- $J=28$, bump width $\ell=0.3$, centres evenly spaced on `x_range` padded by $2\ell$.
- $s_\theta$ is set so the prior s.d. of $f_{\text{base}}$, averaged over the range, is `f_std` $=1$.

**Gap part: harder inside the sparse strip.** A second set of *narrow* bumps, windowed so that it exists only inside the sparse strip $[1,4]$:

$$
f_{\text{gap}}(x)=h_{[1,4]}(x)\sum_{k=1}^{K}\eta_k\,\tilde b_k(x),\qquad \tilde b_k(x)=\exp\!\Big(-\frac{(x-\tilde c_k)^2}{2\ell_g^2}\Big),\qquad \eta\sim\mathcal N(0,s_\eta^2 I_K),
$$

$$
h_{[a,b]}(x)=\operatorname{logistic}\!\Big(\frac{x-a}{w}\Big)-\operatorname{logistic}\!\Big(\frac{x-b}{w}\Big).
$$

- $K=10$ centres $\tilde c_k$ evenly spaced on $[1,4]$; narrow width $\ell_g=0.1$ (about $\ell/4$); window edge $w=0.05$.
- $s_\eta$ is set so the prior s.d. of $f_{\text{gap}}$ inside the strip is `gap_std` $=1.25$.

**Why this shape.** The dense regions carry no hint of the gap's higher frequency, so a model that infers smoothness from the data it sees will be confidently wrong in the gap. That is DEUP's point: posterior variance misses misspecification bias, while excess risk includes it (Lahlou et al., 2023, §1). Our epistemic truth is exactly that excess risk (proposal §2.2).

No method can *know* the gap wiggles from an empty region. What can differ is how much uncertainty each method keeps there:

- the GP commits to lengthscales fitted on the smooth strips;
- the PFN-style models average over their prior's function classes.

`gap_std` $=1.25\approx5\sigma_0$ keeps the gap misfit above the noise of toy rung 1 (§2.3).

**Curation, pre-registered.** For candidate seeds `toy_seed` $=0,1,\dots$, draw $(\theta,\eta)$ and keep the **first** seed whose $f$ meets all of the following:

1. $\max_x|f(x)|\le4$, so that $f$ and its ordinary-noise 90 % band ($\pm1.645\,\sigma_0\approx\pm0.41$) stay inside the fixed $\pm4.5$ y-limits of row 1 of every figure;
2. the s.d. of $f_{\text{gap}}$ over $[1,4]$ is at least 0.5, so the gap structure is visible;
3. the range of $f$ over the noise strip $[-3.5,-1.5]$ is at least 1, so the noise strip is not a flat region.

Curate on $f$ alone; **never look at any model output while curating.** `00_curate_toy.py` records the chosen seed in `toy_function.yaml`; from then on it is frozen like every other constant.

### 2.3 The three toy rungs

All three use the **same $f$, the same context $x$-positions and the same standard-normal draws $\varepsilon$** (common random numbers), so the rungs differ only in $\sigma(x)$.

| Toy rung | $\sigma(x)$ | Sparse strip | What it shows |
|---|---|---|---|
| 0 — deterministic | $0$ | $[1,4]$ | Does $\hat E$ rise in the gap with $\hat A\approx0$? $A=0$ exactly: a floor test |
| 1 — homoscedastic | $\sigma_0$ everywhere | $[1,4]$ | Does gap detection survive noisy surroundings? |
| 2 — heteroscedastic | $\sigma_0\,e^{\phi_b h_{[-3.5,-1.5]}(x)}$ | $[1,4]$ | Attribution: high $\hat A$ only in the noise strip; $\hat E$ higher in the gap than in the noise strip |

Fixed constants (`toy_rung{k}.yaml`):

- $\sigma_0=e^{-0.7-\log 2}\approx0.25$, the median ordinary noise of the old rung 3;
- $\phi_b=\log 6$, so the noise strip has $\sigma\approx1.5$;
- noise strip $[-3.5,-1.5]$, edge width $w=0.1$.

The noise strip and the sparse strip sit $2.5\approx8\ell$ apart, so the two regions do not bleed into each other.

**What to expect in toy rung 2.** Under the excess-risk definition, $E$ also rises in the noise strip at equal context density (roughly $E\propto\sigma/n_{\text{local}}$). So the noise strip is high-$A$ *and* raised-$E$, not low-$E$. The test is whether $\hat A$ rises only in the noise strip, and whether $\hat E$ rises more in the sparse strip than in the noise strip.

**What to expect in toy rung 1.** The *true* gap bump in $E$ is smaller than in toy rung 0, for every model. A mean error $\delta$ costs about $|\delta|$ in CRPS without noise but only about $\delta^2/(2\sigma\sqrt\pi)$ with noise $\sigma\gg|\delta|$. For a single Gaussian forecast with latent s.d. $s$ and noise $\sigma$, the self-reported split is

$$
\hat A=\frac{\sigma}{\sqrt\pi},\qquad \hat E=\frac{\sqrt{s^2+\sigma^2}-\sigma}{\sqrt\pi}\ \xrightarrow{\ \sigma=0\ }\ \frac{s}{\sqrt\pi}.
$$

So a GP's latent s.d. is proportional to the CRPS epistemic term only in the deterministic case; with noise it is compressed. Read each toy rung as "does $\hat E$ follow $E(q_m)$?", never as "does $\hat E$ match toy rung 0?".

### 2.4 Context and queries

- **Context.** $x$ drawn from a piecewise-uniform density on `x_range`: relative density 1 outside $[1,4]$ and `gap_density` $=0.075$ inside. Drawn once at $n=1024$ from the context stream of §9; each $n\in$ `n_grid` uses the first $n$ points (nested contexts). At $n=16$ the gap is expected to be empty; it holds about 2 points at $n=64$, 8 at $n=256$ and 32 at $n=1024$, where it is still about 13 times sparser than its surroundings.
- **Why this density.** With about 12 points in the gap at $n=1024$, a reviewer could argue that the heteroscedastic GP's "noise" reading of the gap is defensible from the data alone, since 12 scattered points look like noise. With about 32, the wiggles are clearly signal, and "the GP misattributes" becomes unambiguous.
- **Queries.** `n_query` $=200$ points evenly spaced on `x_range`, shared by all toy rungs and models.
- **Truth** (§4.1). Closed form and **exact** for this context: $A=\sigma(x_*)/\sqrt\pi$ and $E(q_m)=\int(F_m-\Phi)^2dt$. There is no realized-task noise, since $f$ and $\sigma$ are known.

### 2.5 Models

| Model | Runs | Notes |
|---|---|---|
| Heteroscedastic GP | ✓ | Adapter §5.2 (recycled, already validated). Priors deliberately not matched to $f$ |
| Dec-ICL | ✓ when the checkpoint arrives | Adapter §5.3. Its training prior (explicit $f$ plus Gaussian noise) is close to this toy, so T1 is, if anything, tilted *towards* Dec-ICL; T2 is tilted towards iLTMv2. State both |
| iLTMv2 | row reserved for Phase 1 | Adapter §5.5 |
| TabICLv2 | backbone check only (§2.7) | Total only |

### 2.6 Figures

**One figure per (model, toy rung), 3 rows × 4 columns: 9 figures.** This is the layout of the previous version's context figures; recycle `figures.py`. Columns are $n\in\{16,64,256,1024\}$. Title: `toy rung k: <method>`.

| Row | Content | Axis |
|---|---|---|
| 1 | Context points; true mean $f$ with true 90 % band $f\pm1.645\,\sigma$; forecast median with forecast 90 % band (5 %–95 % quantiles of $q_m$) | linear |
| 2 | Realized $A$ (blue), $\hat A$ (red) and total $e(q_m)$ (black) | linear |
| 3 | Realized $E(q_m)$ (blue, **this model's own** excess) and $\hat E$ (red) | log, floor $\sigma_{\text{floor}}/\sqrt\pi$; points below it are drawn on the floor |

- Shade the noise strip (orange) and the sparse strip (purple) in every panel.
- Put a rug of the context $x$-positions on rows 2–3.
- **Within a toy rung, share the y-limits of each row across the three models' figures**, so they can be compared side by side.
- For toy rung 0, row 2's truth is the zero line. Report each model's σ clip rate in a corner of row 2 (§10, gotcha 3).
- Row 2 carries $e(q_m)$ so that the split is visible at a glance. For every model whose $\hat E$ is $e_{\text{tot}}-\hat A$ (§4.4), $\hat A\le e(q_m)$, and the vertical gap between the black and red curves *is* $\hat E$.

As reference, you can reuse the code available in `/private/home/ajaras/iLTM-bio/UQ_project/SCRAPPED/v2/experiment0_baseline/src/figures.py`.

**Paper figure.** `01_toy.py --paper-figure` draws one 3×3 figure for toy rung 2 at $n=1024$: one column per model (heteroscedastic GP, Dec-ICL, iLTMv2), with the same three rows as above. It is built from the saved outputs only, once all three models have been run, and shares each row's y-limits across its three columns. It focuses on the one question the paper figure has to answer: with the same context, where does each model put aleatoric and epistemic uncertainty? The per-(model, toy rung) figures stay as the full record across $n$.

### 2.7 Backbone check (run first)

The toy function is far from the observed-cut prior that iLTMv2's heads are trained on. Before building figures around iLTMv2, check that its backbone behaves sensibly on this kind of input. `00a_backbone_check.py` runs TabICLv2 on all three toy rungs and all $n$, and draws the same 3×4 layout:

- row 1 as in §2.6;
- row 2: realized $A$, realized expected score $A+E(q_\theta)$, and TabICLv2's total $e(q_\theta)$;
- row 3: realized $E(q_\theta)$.

This doubles as the motivating figure. On toy rung 2, the single total rises in both strips, and nothing in it says which is which.

**Pass criteria**, fixed before running, at every $n\in\{64,256,1024\}$:

1. **Fit.** On every toy rung, mean CRPS of TabICLv2 on queries outside both strips is at most $1.5\times$ that of the heteroscedastic GP. CRPS here is the exact expected score $A+E(q_m)$ under the known truth.
2. **Locality.** On toy rung 0, mean $e(q_\theta)$ over sparse-strip queries is at least $2\times$ the mean over queries outside both strips.
3. **Noise.** On toy rung 2 at $n\ge256$, mean $e(q_\theta)$ over the noise strip is at least $2\times$ the mean outside both strips.

The check fails if criterion 1 fails at any $n$, or if criterion 2 or 3 fails at more than one $n$.

> **Note — fallback if the backbone check fails: generate $f$ from our own prior.** Replace the hand-built $f$ with a 1-D draw from the observed-cut target branch: one feature $x$ on `x_range`, the target graph $\Psi_\theta(x,U_2)$ with its private noise **frozen at a single draw** $u_2$, so that $f(x)=\Psi_\theta(x,u_2)$ is deterministic. Then add our own Gaussian noise $\sigma(x)$ exactly as in §2.3. The truth stays closed form, and $f$ comes from a function class the backbone has seen. What is lost is the controlled "harder in the gap" structure: curate instead for a draw whose variation inside $[1,4]$ is at least as large as outside (the same pre-registered style as §2.2).

### 2.8 Interpretation guide

| Toy rung | Good behaviour | Typical failure |
|---|---|---|
| 0 | $\hat A\approx0$ without relying on the σ clip; $\hat E$ peaks in $[1,4]$ and shrinks with $n$ elsewhere | $\hat E$ too low in the gap (overconfident lengthscale); $\hat A>0$ absorbing misfit |
| 1 | $\hat A$ flat at $\sigma_0/\sqrt\pi$; $\hat E$ follows the (smaller) true gap bump | $\hat A$ rises in the gap (gap read as noise) |
| 2 | $\hat A$ tracks the noise bump only; $\hat E$ higher in the gap than in the noise strip | $\hat A$ smeared across both strips; $\hat E$ highest in the noise strip |

> **Note — optional appendix figure: seed-averaged reliability on the toy generator.** The previous version graded rungs over a *distribution* of low-rank GP functions: new $(\theta,\phi)$ per seed, `n_seeds` $=20$, truth-on-estimate reliability plots of realized $A$ against $\hat A$ and realized $E(q_m)$ against $\hat E$, coloured by $n$ (the grading of §6, applied to the toy generator). It is not a priority, but it would make an interesting appendix figure: it turns the single-function picture of T1 into a calibration statement over the toy family. It is cheap, because the generator, evaluator and reliability-plot code already exist in the previous version. If added, draw $\eta$ per seed as well, so that every function has its harder gap, and run the same three noise settings.

---

## 3. T2 — Cut-graph test (in distribution)

### 3.1 Tasks

Tasks come from the observed-cut prior of Phase 1 (proposal §4.3):

- $X$ is generated by an $X$-graph with `graph_scm`'s mechanism (no $y$ node).
- $y=\Psi_\theta(X,U_2)$ comes from a second, `graph_scm`-style target graph whose $X$-input nodes are the $X$-graph's feature-hosting nodes (observed post-converter columns plus fresh hidden dims), and whose other roots are private noise roots; hidden dims and private roots are $U_2$.
- Redrawing only $U_2$ gives exact i.i.d. replicates of $y$ at fixed $x_*$; one extra pass of the target branch gives a replicate for every row at once.

A **task** is everything the generator fixes once: the DAG, the node functions, the target graph and its parameters (cached in `RandomNodeFunction._fit`). A **task seed** redraws only the rows, i.e. a fresh pool of $X$ from $U_1$, and with it a fresh context, hole and query set. This gives two levels of repetition: **across tasks**, and **within a task across seeds**.

Provisional constants (`cutgraph.yaml`): `n_tasks` $=50$, `n_task_seeds` $=5$, `n_query` $=200$ (100 hole + 100 ordinary), `hole_frac` $=0.10$.

Note the following:

- **It favours iLTMv2**, because it is iLTMv2's training prior. It is an in-distribution test for iLTMv2, not a fair comparison.
- **It depends on the Phase 1 prior.** Build it after T1. It doubles as the first working implementation of the observed cut.
- **Dec-ICL assumes Gaussian end-node noise** (Bergna et al., 2026), and $y\mid x$ here is non-Gaussian. It is graded as published. Part of any gap is that misspecification, and the thesis says so.

> **Generator settings: see Phase 1.** The target-graph design, its settings and the adapted rejection filter are defined in proposal §4.3 and `experiment1_uq_heads/phase1_design.md` Training §1 and §1.4, and implemented in `experiment1_uq_heads/src/iltm2/prior/` (`_cut_graph_scm.py`, `_filter.py`), which T2 imports as `iltm2.prior`. The final target-graph configuration is the one that passes the difficulty-match gate (`experiment1_uq_heads/scripts/00f_difficulty_match.py`).

### 3.2 Context holes

To test scarce-context behaviour without leaving the prior, carve a hole in the **context** rather than inventing query points. A synthetic $x$ far from the data could be evaluated through $\Psi_\theta$, but it would also be off the $X$ marginal, which confounds "sparse context" with "out-of-distribution $x$". Hole queries are real draws from the prior; only the context around them is missing.

Per (task, seed):

1. Draw a pool $P$ of $|P|=\lceil(n_{\max}+100)/(1-h)\rceil$ rows, with $h=$ `hole_frac` and $n_{\max}=1024$ (1 249 rows). This leaves room for the full context plus 100 spare rows outside the hole, and gives a hole of 125 rows.
2. Standardize each feature over $P$ (categorical features ordinal-encoded, then standardized; an approximation, recorded as such).
3. Pick an anchor uniformly from $P$. The **hole** $H$ is the $\lceil h|P|\rceil$ rows of $P$ nearest to the anchor (Euclidean, standardized), anchor included. That is about 10 % of the $X$ marginal's mass around the anchor, the same region for every $n$.
4. **Context**: drawn from $P\setminus H$ once at $n_{\max}$; each $n$ uses its first $n$ rows (nested).
5. **Hole queries**: 100 rows from $H$. **Ordinary queries**: the 100 spare rows of $P\setminus H$ that never enter the context.

**What the hole does and does not test.** The truth $E(q_m)$ rises in the hole only if the carved region actually matters for $\Psi_\theta$. If the hole lies along features the target ignores, $E$ stays flat, and a model that simply flags feature-space isolation is graded as wrong. That is deliberate: kNN distance is the proxy that Phase 2 argues against.

### 3.3 Truth

From `R_rep` replicates $y^{(1)},\dots,y^{(R)}$ at each query (§4.2). Assert bit-identical $X$ across replicates; it is the only correctness guard the construction needs.

### 3.4 Models and grading

Dec-ICL (§5.3) and iLTMv2 (§5.5). The heteroscedastic GP is excluded: it is 1-D only and too costly.

Grading follows §6. Rows: pooled; per $n$; per query group (hole / ordinary). Two aggregation levels, each reported separately:

- **Across tasks** (the calibration test): pool all tasks and seeds; bootstrap whole tasks.
- **Within task** (descriptive): fit per task over its seeds; report the distribution (median and IQR) of per-task slope and mean signed error.

The two levels mean different things. The argument of §6.1 guarantees that an ideal estimator is calibrated **on average over tasks**, not within each task: a model that cannot identify the task from its context should not be perfectly calibrated on any single one. So the across-task slope is the grade. The within-task spread shows how much a model's estimates adapt to the specific task in front of it, and whether the pooled result hides tasks where it is badly off.

---

## 4. Ground truth

### 4.1 The quantities

For a query $x_*$ in a task $\tau$ with true conditional law $p_\tau(\cdot\mid x_*)$, a model with forecast $q_m$ reports $\hat A$ and $\hat E$ (§5). They are graded against the **realized** truth of that task:

$$
A=e\big(p_\tau(\cdot\mid x_*)\big),\qquad
E(q_m)=\mathbb E_{Y\sim p_\tau}\big[S(q_m,Y)\big]-A=d_S\big(q_m,p_\tau\big).
$$

- $A$ is the score that a forecaster who *knew the task* would still pay: the irreducible floor.
- $E(q_m)$ is the extra score that *this model's own forecast* pays on top of the floor: its excess risk (Hofman et al., 2024).

**Why the model's own excess.** What a user acts on is "how far off is this model's forecast likely to be, beyond the irreducible noise?". Grading against $E(q_m)$ asks whether a model **knows its own error**. For the baselines, $\hat E=e(q_m)-\hat A$ is the model's self-reported spread beyond its aleatoric part, which equals its true excess only if its members behave like a true posterior over tasks (§4.4); the gap between the two is what the tests measure. For iLTMv2, $\hat E_{\text{dir}}$ is trained to target exactly $E(q_\theta)$ (proposal §4.4). $A$ needs no such choice: it is the same for every model.

**T1 (closed form).** $p_\tau=\mathcal N\big(f(x_*),\sigma(x_*)^2\big)$, so with CRPS (Gneiting & Raftery, 2007)

$$
A=\frac{\sigma(x_*)}{\sqrt\pi},\qquad E(q_m)=\int\big(F_m(t)-\Phi_\tau(t)\big)^2dt,
$$

the Cramér distance on the shared $t$-grid (§4.3), with $\Phi_\tau$ a step at $f(x_*)$ on toy rung 0.

### 4.2 Replicate estimators (T2, T3)

From $R$ i.i.d. replicates at $x_*$:

$$
\widehat A=\frac{1}{R(R-1)}\sum_{r<s}\big|y^{(r)}-y^{(s)}\big|,\qquad
\widehat E(q_m)=\frac1R\sum_r\mathrm{CRPS}\big(q_m,y^{(r)}\big)-\widehat A,
$$

with $\mathrm{CRPS}(q_m,y)=\int\big(F_m(t)-\mathbb 1\{t\ge y\}\big)^2dt$. Both are exactly unbiased, so the replicate noise is zero-mean and handled by the grading (§6.1).

**Replicates must be independent of the context.** If a replicate row was also in the context, $q_m$ has already seen it, and $\widehat E$ is biased downwards. On T2 this cannot happen, because replicates are fresh $U_2$ draws. On T3 it can, so it is enforced explicitly (§7.2).

### 4.3 Entropy and excess kernels

$$
e_{\mathrm{CRPS}}(F)=\int F(t)\big(1-F(t)\big)\,dt,\qquad d_S(F,G)=\int\big(F(t)-G(t)\big)^2dt.
$$

```python
def e_crps(cdf, t_grid):
    F = cdf(t_grid)
    return np.trapezoid(F * (1.0 - F), t_grid)

def excess(cdf_model, cdf_true, t_grid):
    d = cdf_model(t_grid) - cdf_true(t_grid)
    return np.trapezoid(d * d, t_grid)

def project(cdfs, weights, t_grid):
    """cdfs: list of callables. Returns (A_hat, e_total, E_hat)."""
    A = sum(w * e_crps(F, t_grid) for w, F in zip(weights, cdfs))
    bar = lambda t: sum(w * F(t) for w, F in zip(weights, cdfs))
    e = e_crps(bar, t_grid)
    assert e - A >= -1e-9, "negative epistemic — projection applied wrongly"
    return A, e, e - A
```

This works for any CDF: Gaussian mixture, binned predictive, point mass, a 999-quantile step function.

**The $t$-grid.** 4096 points spanning $\pm10$ s.d. around the forecast, **widened to also cover the truth**: the true law on T1, all replicates on T2 and T3. One grid construction is shared by every adapter and by `truth.py`. Assert that both CDFs reach their tails inside the grid ($<10^{-4}$ at each end).

**Unit tests.**

- $e_{\mathrm{CRPS}}(\mathcal N(0,\sigma^2))=\sigma/\sqrt\pi$ to $10^{-6}$.
- Agreement with $\tfrac12\mathbb E|Y-Y'|$ by Monte Carlo on a two-component mixture.
- `excess` against the closed-form CRPS of a Gaussian forecast at a point.

**The member-wise rule.** $\hat A$ is the average of member entropies, never a conversion of a pre-aggregated "data uncertainty" scalar. $\mathrm{Var}=\mathbb E[\sigma^2]+\mathrm{Var}[\mu]$ is additive in variance; $\sigma/\sqrt\pi$ is not. A negative $\hat E$ from `project` always means this rule was broken.

### 4.4 What $e_{\text{tot}}-\hat A$ measures

The heteroscedastic GP and Dec-ICL report $\hat E=e_{\text{tot}}-\hat A$ (§5.2, §5.3). This is **not** an approximation of the realized divergence $d_S(q_m,p_\tau)$. It is the model's own posterior expectation of that divergence, and that is an exact identity.

**The identity.** Write the forecast as its mixture of members, $q_m=\sum_m w_mF_m$: posterior draws for the GP, latent bins convolved with the noise model for Dec-ICL. The expected score $\mathbb E_{Y\sim G}[S(q_m,Y)]$ is linear in the law $G$ of $Y$, so

$$
\sum_m w_m\,\mathbb E_{Y\sim F_m}\big[S(q_m,Y)\big]=\mathbb E_{Y\sim q_m}\big[S(q_m,Y)\big]=e(q_m).
$$

With $d_S(q_m,F_m)=\mathbb E_{Y\sim F_m}[S(q_m,Y)]-e(F_m)$ (§2.1 of the proposal), averaging over members gives

$$
\sum_m w_m\,d_S(q_m,F_m)\;=\;e(q_m)-\sum_m w_m\,e(F_m)\;=\;e_{\text{tot}}-\hat A .
$$

So $e_{\text{tot}}-\hat A$ is exactly the excess score the model *expects* to pay if the truth is one of its own members, drawn with its own weights: a **self-predicted excess**. It is non-negative by concavity of $e$, which is what the assertion in `project` checks. This is the decomposition of proposal §2.2, which follows Hofman et al. (2024). For the log score the same identity gives the mutual information $I(Y_*;\tau\mid x_*,D)$, the BALD term (Houlsby et al., 2011; Depeweg et al., 2018).

**When it equals the truth.** If the members are the exact posterior over tasks under the prior that generated the task ($w_m,F_m$ playing the role of $p(\tau\mid D),p_\tau$), the identity becomes

$$
\mathbb E_{\tau\mid D}\big[d_S(q,p_\tau)\big]=e(q)-\mathbb E_{\tau\mid D}\big[e(p_\tau)\big],
$$

so $e_{\text{tot}}-\hat A$ is unbiased for the realized excess **on average over the tasks consistent with the context**, not on each task. That is why T2 and T3 grade truth on estimate across tasks (§6.1), and why T1, with one fixed task, is read descriptively. When the model is misspecified, its members do not cover the truth and the self-predicted excess misses the bias, which is DEUP's argument (Lahlou et al., 2023, §2). The gap between self-predicted and realized excess is exactly what the tests show. iLTMv2's $\hat E_{\text{dir}}$ is instead trained on the realized excess itself (proposal §4.4).

**No general ordering between $e(q_m)$ and the realized divergence.** On a single task, either can be much larger:

- *Sharp and wrong.* As the forecast's spread goes to 0, $e(q_m)\to0$ while $d_S$ tends to the size of the error (on toy rung 0, $d_S\to|\delta|$ for a point forecast off by $\delta$). Here $d_S\gg e(q_m)$.
- *Exact.* $q_m=p_\tau$ gives $d_S=0$ and $e(q_m)=A>0$.
- *Centred but over-dispersed.* $q_m=\mathcal N(f,(1.2\sigma)^2)$ against $p_\tau=\mathcal N(f,\sigma^2)$ gives $e(q_m)-A=0.2\sigma/\sqrt\pi\approx0.11\sigma$ but $d_S=\big(\sqrt{2\cdot2.44/\pi}-1/\sqrt\pi-1.2/\sqrt\pi\big)\sigma\approx0.005\sigma$: near the truth the divergence is second order in the spread error, while $e(q_m)-A$ is first order.

An ordering holds only in expectation, for an exact Bayesian: $\mathbb E_{\tau\mid D}[d_S]=e(q)-\mathbb E_{\tau\mid D}[A]\le e(q)$. On average, an exact Bayesian is never more wrong than its own spread suggests; on a single task it can be.

---

## 5. Adapters

Every adapter returns a weighted family of predictive CDFs, or a closed form equivalent to one, and the kernel does the rest.

```python
@dataclass
class UQOutput:
    A_hat: np.ndarray       # (n_query,) — y-units
    E_hat: np.ndarray
    e_total: np.ndarray
    cdf: Callable           # the forecast q_m, needed for the truth E(q_m)
    fit_seconds: float
    extras: dict            # native outputs, unprojected, for audit
```

Always fill `extras`: when a figure looks wrong you will want the native numbers without re-running. **Each method runs with its own default preprocessing**; whatever it does internally, its adapter returns quantities in $y$ units.

### 5.1 Which model runs where

| Model | T1 toy | T2 cut-graph | T3 transfer | T4 active learning |
|---|---|---|---|---|
| Heteroscedastic GP | ✓ | — | — | — |
| Dec-ICL | ✓ | ✓ | ✓ | ✓ if checkpoint, else reported numbers |
| Tuned-ICL | — | — | — | ✓ if checkpoint, else reported numbers |
| TabICLv2 | backbone check (§2.7) | — | — | ✓ |
| iLTMv2 | row reserved | row reserved | row reserved | row reserved |
| Random (TabICLv2 predictor) | — | — | — | ✓ |

### 5.2 Heteroscedastic GP (PyMC) — recycled

The previous version's adapter is working and validated; reuse it, with a learned constant mean on the log-noise latent (below). It follows the PyMC example "Heteroskedastic Gaussian Processes", section "Sparse Heteroskedastic GP". For reference:

- **Model.** Two latent functions, the mean $\mu(x)$ and the log noise scale $g(x)$, with $y\sim\mathcal N\big(\mu(x),e^{2g(x)}\big)$. $\mu$ has a zero mean function; $g$ has a learned constant mean $m_g\sim\mathcal N(-2,2^2)$ in standardized units (a noise s.d. of about $0.14$ s.d. of $y$ at the prior centre).
- **Approximation.** Each latent is a DTC sparse GP over 40 inducing points evenly spaced on `x_range` (a fixed grid keeps the cost flat in $n$), with whitened inducing values: $u=Lv$, $v\sim\mathcal N(0,I)$, $L=\mathrm{chol}(K_{uu}+10^{-6}I)$, $f(X)=K_{fu}K_{uu}^{-1}u$.
- **Priors.** Generic squared-exponential kernels $\eta^2\,\mathrm{ExpQuad}(\ell)$, deliberately *not* matched to the toy function: $\ell\sim\mathrm{InverseGamma}(\mu_\ell,\sigma_\ell)$ with $(\mu_\ell,\sigma_\ell)$ set from the spread of pairwise context distances (the example's `get_l_prior`); $\eta\sim\mathrm{Gamma}(2,1)$. $y$ is standardized with context statistics before fitting and outputs are mapped back to $y$ units.
- **Inference.** NUTS through numpyro, `target_accept` $=0.95$, 2 chains × (500 tuning + 250 draws), giving $S=500$. Log R-hat, divergences, fit time and the posterior mean of $m_g$ in `extras`; flag fits with R-hat $>1.05$.
- **Prediction.** The example's `conditional`: per posterior draw, mean $K_{su}K_{uu}^{-1}u$ plus a draw from the conditional variance $\mathrm{diag}(K_{ss}-Q_{ss})$. Only marginals at each query enter a member CDF, so the diagonal is exact here.
- **Members.** $\{\mathcal N(\mu_s(x_*),\sigma_s(x_*)^2)\}_{s=1}^S$, $w_s=1/S$, through `project`, so $\hat A=\frac1S\sum_s\sigma_s/\sqrt\pi$.
- **Toy rung 0.** $\sigma\equiv0$ pushes $g$ towards $-\infty$. Clip $\sigma_s$ at `sigma_floor` and report the clip rate.

**Why the log-noise latent has a constant mean.** With zero mean functions for both latents, the GP reported $\hat A\approx420$ in the sparse strip of toy rung 0 at $n=256$ (and about 100 at $n=1024$), on a deterministic function whose whole range is under 8 units. The σ clip rate was 0 %, so the clip did not cause it: the clip only floors σ from below. The forecast band itself left the plot, so this was what the fitted model reported, and nearly all of it was aleatoric ($\hat E\approx3$ there). Two effects stacked:

1. **A genuine one.** The few gap points deviate from the smooth mean, and raising the noise latent there is cheaper in marginal likelihood than shortening the lengthscale everywhere, so the misfit is read as noise. This is a real property of the heteroscedastic GP, also visible on toy rung 2, and worth showing.
2. **An artifact of the parameterization.** With noiseless data, $g$ must sit far below zero in the dense regions, which a zero-mean GP can only do with a large amplitude. In the data-free gap, $g$ then reverts towards 0 with that large variance, and because $\hat A$ averages $\sigma_s=e^{g_s}$ over draws (a lognormal mean), the upper tail of the draws dominates it.

A learned constant mean lets $g$ sit low without a large amplitude, so data-free regions revert to the task's typical noise level. The first effect stays visible without the second. The mean and its prior are generic choices, independent of the toy function.

### 5.3 Dec-ICL (Bergna et al., 2026)

The decoupled TabICL model, obtained from the authors. Until the checkpoint arrives, the adapter is written against the interface below and unit-tested on synthetic outputs (§10, gotcha 2). **Confirm the exact output format with the authors' code before finalizing.**

**Native outputs** (Bergna et al., 2026, §3.1):

- a latent-signal head: probabilities $\pi_j$ over fixed target bins $a_0<\dots<a_K$ with centres $c_j$, a distribution over the noiseless $f(x_*)$;
- an aleatoric head: the observation-noise variance $\hat\sigma^2(x_*)$.

The observation-level predictive is their convolution.

**Conversion.** Members are the latent values $u$ (under whatever within-bin law their code uses), each with law $\mathcal N(u,\hat\sigma^2)$. Every member shares the same $\hat\sigma$, so

$$
\hat A=\frac{\hat\sigma}{\sqrt\pi}\ \ \text{exactly},\qquad e_{\text{tot}}=e_{\mathrm{CRPS}}(q_m),\qquad \hat E=e_{\text{tot}}-\hat A,
$$

where $q_m$ is **their** observation-level predictive, reproduced exactly as their code builds it (point masses at $c_j$, or mass spread within bins). Any spread *within* a latent bin is uncertainty about $f$, so it belongs to $\hat E$, never to $\hat A$. If their aleatoric head outputs a distribution over $\sigma^2$ rather than a point, use $\hat A=\mathbb E[\sigma]/\sqrt\pi$ over it instead.

**Checks and logs.**

- Unit test: our $q_m$, integrated over their observation bins, reproduces their observation-level bin masses to $10^{-6}$.
- On toy rung 0, $\hat\sigma\to0$: clip at `sigma_floor` and report the clip rate. The latent bin width puts a floor under $e_{\text{tot}}$, so log the bin width in $y$ units.
- Log $\pi$, $\hat\sigma$ and their native latent variance in `extras`.

**Backbone confound.** Dec-ICL is fine-tuned from TabICL (Qu et al., 2025) on the authors' own heteroscedastic prior; iLTMv2 is built on TabICLv2 (Qu et al., 2026). Confirm the backbone version with the authors. Any Dec-ICL versus iLTMv2 gap mixes method, backbone and prior, and the thesis says so.

### 5.4 TabICLv2 (Qu et al., 2026)

Total only. `e_total` and `cdf` come from the 999 raw quantiles at levels $k/1000$: the CDF is piecewise-linear through them, 0 below the first and 1 above the last, and its entropy is integrated exactly segment by segment. `A_hat` and `E_hat` are `None`; the evaluator skips them rather than coercing them to zero. Used in the backbone check (§2.7) and in T4.

### 5.5 iLTMv2

Row reserved for Phase 1. $\hat A$ from the aleatoric head and $\hat E=\hat E_{\text{dir}}$ from the epistemic head (proposal §4.4); `e_total` and `cdf` from the quantile head. Also log $e(q_\theta)-\hat A$ in `extras`, which is the amortization gap of proposal §4.7.

---

## 6. Grading (T2, T3)

### 6.1 The rule: regress truth on estimate

The realized truth equals what a model could at best know from its context, plus a zero-mean deviation that the context cannot resolve. Tasks are drawn from a fixed prior, so for any query

$$
\mathbb E\big[T\mid D,x_*\big]=\bar T(D,x_*),\qquad T\in\{A,\,E(q_m)\},
$$

where $\bar T$ is the posterior-averaged truth under that prior. Every estimate $\hat T$ is computed from $(D,x_*)$, plus the method's own randomness, which is independent of the task given $D$. Hence $\mathbb E[T\mid\hat T]=\mathbb E[\bar T\mid\hat T]$.

**Consequence:** regressing realized truth on the estimate gives the same answer as if $\bar T$ were known. Regressing the estimate on realized truth does not: noise in the regressor biases the slope towards zero, which is indistinguishable from the shrinkage we want to detect. **Never regress an estimate on realized truth.** Replicate noise is zero-mean, so the same rule covers it.

This identity holds when averaging over tasks. It is why the across-task rows are the grade and the within-task rows are descriptive (§3.4).

### 6.2 Metrics

For each (test, method, quantity $\in\{A,E\}$), fit per-query OLS $T=a+b\,\hat T$.

| Metric | Ideal | Reading |
|---|---|---|
| **slope $b$** (headline) | 1 | $b>1$: estimates too compressed (e.g. a shrunk $\hat E$); $b<1$: estimates exaggerate differences between queries |
| intercept $a$ | 0 | constant offset |
| mean signed error $\overline{T-\hat T}$ | 0 | $>0$: underestimates on average |

Every metric gets a 95 % bootstrap CI, resampling **whole tasks** for the across-task rows.

- **Rows.** Pooled; per $n$; per query group (T2: hole / ordinary; T3: held-out cell / other); T3 also per $(k,d)$.
- **$R^2$ and rank correlation are not reported.** Against realized truth they are capped by the irreducible deviation and measure the testbed's noise, not the method.
- **Diagnostics.** Log `fit_seconds`, the forward-pass count and the σ clip rate as columns.

**Why sweep $n$, and why also split by it.** $E(q_m)$ typically shrinks as context grows, so sweeping $n$ moves the epistemic truth across its range and gives the pooled regression something to fit. But a method whose bias depends on $n$ gets a pooled slope that describes no context size it was actually run at; the per-$n$ rows show where it breaks. Read them with their CIs; the pooled slope stays the headline.

Aggregate with medians and bootstrap CIs. Never average raw scales across tasks of different $y$-scale.

### 6.3 Figures

Per (test, method), **two reliability plots**:

1. **$A$:** $\hat A$ on the x-axis, realized $A$ on the y-axis, linear axes.
2. **$E$:** $\hat E$ on the x-axis, realized $E(q_m)$ on the y-axis, **log–log display**. Axes are for display only; every metric uses linear values. Values below the display floor are drawn on the floor and counted on the panel.

In each plot:

- per-query points in faint grey;
- one reliability curve per $n$, coloured by $n$: `n_bins` equal-count bins of $\hat T$, bin means with bootstrap CIs, plus a pooled curve in black;
- the $y=x$ line;
- **a second panel per plot split by query group** (hole / ordinary, held-out cell / other), so a model that is calibrated on ordinary queries but not in holes is visible at a glance.

Colouring by $n$ shows the $n$-dependent failures without another figure: an $\hat A$ that drifts with $n$ shows as separated curves in plot 1; a method that puts total uncertainty in both slots shows as plot-1 curves that bend upwards at small $n$.

---

## 7. T3 — Transfer test

### 7.1 Purpose

iLTMv2's heads are trained on the observed cut, which gives up latent confounding between $X$ and $y$ (proposal §4.3). T3 asks whether the heads still report correct $A$ and $E$ on the **unmodified, fully confounded** `graph_scm`, where no closed form or replicate mechanism exists. Dec-ICL is graded the same way, as a reference.

### 7.2 Ground truth by exact match

Restrict every observed feature to be **categorical with low cardinality $k$**, $d$ features per task, so there are $k^d$ feature cells. Rows of the unmodified prior are i.i.d. from the joint law, so all rows sharing a query's exact feature vector are exact i.i.d. draws from $p_\tau(y\mid x_*)$. There is no distance threshold and no bandwidth.

Per (task, seed):

1. Draw one large pool $P$ from the unmodified prior.
2. Group $P$ by exact feature vector. A cell is **eligible** for queries if it holds at least `R_min` $=30$ rows beyond those that could enter the context.
3. **Held-out cells.** Mark 10 % of eligible cells as held out; remove all their rows from the context candidates.
4. **Context**: drawn from the remaining rows once at $n_{\max}$, nested across $n$. If $n_{\max}$ exceeds the rows available, cap `n_grid` for that $(k,d)$ and log it.
5. **Queries**: 100 from held-out cells, 100 from the other eligible cells.
6. **Replicates** for a query: rows of its cell that are **not in the context**, capped at `R_rep`. Truth by §4.2 with that $R$; log $R$ per query.

Query groups are by design: **held-out cell** vs **other**. At small $n$, many "other" cells are unseen by chance; the held-out group guarantees an unseen cell at every $n$.

Provisional sweep (`transfer.yaml`): $k\in\{2,3,5\}$, $d\in\{2,3,4\}$, skipping combinations with $k^d>256$. Pool size is chosen so the expected count per cell is at least $3\,$`R_min`. `n_tasks` and `n_task_seeds` as in T2.

**Limitation.** Continuous-feature tasks are covered only by extrapolation from this regime.

> **TBD.** How to force `graph_scm`'s converters to emit only $k$-level categorical features without otherwise changing the prior; whether mixed categorical/continuous tasks can be included (they cannot use exact match).

### 7.3 Models and grading

Dec-ICL and iLTMv2, graded exactly as T2 (§6), with the same across-task and within-task levels, plus per-$(k,d)$ rows.

---

## 8. T4 — Active learning

Reuses the active-learning protocol of DecoupledPFN (Bergna et al., 2026, §5.3 and App. G). It checks whether the epistemic estimate **selects useful data**, not whether it is numerically right. It is the cheapest preview of Phase 2 (context retrieval without training).

### 8.1 Protocol

| Dataset | Dim | $n_0$ | $n_q$ | Pool | Test | Seeds | Source |
|---|---|---|---|---|---|---|---|
| California Housing | 8 | 1024 | 1024 | 5000 | 3000 | 16 | TBD |
| Kin8nm | 8 | 1024 | 1024 | 5000 | 3000 | 20 | TBD |
| NYC Taxi | 7 | 1024 | 1024 | 5000 | 3000 | 19 | TBD |
| Airlines | 5 | 1024 | 1024 | 5000 | 3000 | 20 | TBD |
| YearPredictionMSD | 90 | 1024 | 1024 | 5000 | 3000 | 20 | TBD |

> **TBD — dataset sources.** Exact OpenML IDs and versions (the paper does not give them).

- **Splits.** For each (dataset, seed), fix the pool/test split and the $n_0$ warm-start indices once, and share them across all methods. Store them in `outputs/raw/`, as the paper does (App. G.5).
- **Loop.** At each step the model, with the current labelled set as context, scores every remaining pool point. The highest-scoring point is labelled and added. 1024 steps, one point at a time, no training.
- **Metrics.** On the test set: RMSE, MAE, Gaussian NLL (from the forecast's mean and variance, as in the paper's Table 6), and CRPS. Report final-step values as mean ± std over seeds, plus learning curves of RMSE and CRPS against the number of labelled points.

### 8.2 Methods

| Method | Predictor | Acquisition score | Status |
|---|---|---|---|
| Random | TabICLv2 | uniform over the pool | run |
| TabICLv2 (total) | TabICLv2 | $e(q_\theta)$, total CRPS entropy | run |
| iLTMv2 | iLTMv2 | $\hat E_{\text{dir}}$ | row reserved for Phase 1 |
| Tuned-ICL | Tuned-ICL | total predictive variance, as in the paper | run if checkpoint, else paper's Table 2 |
| Dec-ICL | Dec-ICL | latent (epistemic) variance, as in the paper | run if checkpoint, else paper's Table 2 |

- **Matched comparisons.** TabICLv2 (total) against iLTMv2: same backbone family and loop, differing only in the uncertainty signal. Tuned-ICL against Dec-ICL is the paper's own matched pair. Comparing the two pairs shows whether our decomposition buys as much as theirs over its own total-uncertainty baseline.
- **If the checkpoints arrive,** Tuned-ICL and Dec-ICL run on **our** shared splits with their native acquisition scores, and the comparison is split for split.
- **If they do not,** their rows carry the paper's numbers, marked "reported", read within their reported spread. Those were evaluated on unreleased splits, and the paper's Random row uses the Dec-ICL backbone, so there is no exact reproduction target.

---

## 9. RNG streams

Every RNG stream hangs off `base_seed`:

- **T1:** function `toy_seed` (frozen by curation); context `[base, toy, 1]`, shared across toy rungs; noise $\varepsilon$ `[base, toy, 2]`, shared across toy rungs; model fit `[base, toy, rung, 3, n]`.
- **T2:** task `[base, cg, task, 0]`; rows of a seed `[base, cg, task, seed, 1]`; anchor `[…, 2]`; context `[…, 3]`; replicates `[…, 4]`; fit `[…, 5, n]`.
- **T3:** task `[base, tr, k, d, task, 0]`; pool `[…, seed, 1]`; held-out cells `[…, 2]`; context and queries `[…, 3]`; fit `[…, 4, n]`.
- **T4:** splits as stored in `outputs/raw/`.

One parquet per (test, method, configuration, $n$, seed) in `outputs/raw/`, reused on re-runs; figures go to `outputs/figures/`, tables to `outputs/tables/`.

---

## 10. Gotchas

1. **The truth code is a model too.** Validate it before running any model. `00b_validate_estimators.py`:
   - **$A$:** on toy rungs 0–2, at 50 queries with $10^5$ replicate pairs, $\tfrac12|y^{(1)}-y^{(2)}|$ averaged per query must lie within 3 Monte Carlo standard errors of $\sigma(x_*)/\sqrt\pi$. This shows exactly 0 on toy rung 0, a flat line on toy rung 1, and the noise bump on toy rung 2.
   - **$E$:** for a fixed, deliberately wrong Gaussian forecast, the replicate estimator of §4.2 with $R=100$, repeated 1000 times, must average to the closed-form $\int(F_m-\Phi_\tau)^2dt$ within 3 standard errors. Also report its per-query s.d. at $R=100$: that is the Monte Carlo noise T2 and T3 will carry.

   This is also the estimator audit Phase 1 calls for, run once here so both phases can cite it.
2. **Test the conversion, not the model.** `00c_validate_adapters.py` unit-tests each adapter's path from native outputs to `UQOutput` with synthetic native outputs and closed-form answers, no model fit:
   - $M$ identical members $\mathcal N(\mu,s^2)$ must give $\hat A=e_{\text{tot}}=s/\sqrt\pi$ and $\hat E=0$;
   - a Gaussian mixture must match the folded-normal closed form;
   - Dec-ICL: a one-hot latent distribution must give $\hat E\approx0$ (up to the bin-width floor) and $\hat A=\hat\sigma/\sqrt\pi$; our $q_m$ must reproduce their observation-level bin masses;
   - use several query rows with distinct parameters, so that an axis mix-up fails.

   Tolerances: $10^{-6}$ relative on $\hat A$ and $e_{\text{tot}}$; $10^{-6}$ of $e_{\text{tot}}$ absolute on $\hat E$.
3. **$\sigma\to0$.** Log-parameterized heads diverge on toy rung 0. Clip every predicted σ at `sigma_floor` and **report the clip rate**: a method that passes toy rung 0 only because of the clip has not passed it.
4. **No tuning on query points.** Baselines may use a validation split carved from the context, never the queries. On T3, replicate rows must never be context rows (§4.2).
5. **Fix the row set and the toy function before the first run.** The iLTMv2 rows stay visibly empty until Phase 1. Adding methods after seeing results, or re-curating $f$ after seeing figures, is how benchmarks become advocacy.
6. **Never regress an estimate on realized truth** (§6.1).

---

## 11. Order of work

| # | Task | Size |
|---|---|---|
| 1 | Recycle `entropy.py` and its unit tests; `truth.py` + `00b` estimator checks | half a day |
| 2 | Toy function + curation script + toy rung configs | half a day |
| 3 | Recycle the heteroscedastic GP adapter; run it on toy rungs 0–2 | half a day |
| 4 | Recycle `figures.py` (3×4 context figures); GP figures | half a day |
| 5 | TabICLv2 adapter + backbone check (§2.7). **Decision point:** keep $f$ or switch to the fallback note | half a day |
| 6 | Dec-ICL adapter against the interface + `00c` unit tests; run when the checkpoint arrives | 1 day |
| 7 | Active-learning harness: splits, loop, metrics; Random + TabICLv2 (total); Tuned-ICL / Dec-ICL if available | 1.5 days |
| 8 | T2: cut-graph tasks, holes, replicates, grading (needs the Phase 1 observed-cut prior) | 2 days |
| 9 | T3: categorical pools, held-out cells, exact-match replicates, grading | 1.5 days |
| 10 | (Optional) appendix seed-averaged reliability on the toy generator (§2.8 note) | half a day |

Items 1–5 already give a reportable result: the toy figures for the heteroscedastic GP, plus the motivating TabICLv2 figure.

---

## 12. Open items

1. T2 generator settings and rejection filter (Phase 1 document).
2. T3: forcing $k$-level categorical features in `graph_scm`; final $(k,d)$ sweep.
3. Dec-ICL and Tuned-ICL checkpoints (authors contacted); Dec-ICL output format and backbone version.
4. Active-learning dataset sources.
5. Phase 0 kill criterion.

---

## References

- Bergna, R., Depeweg, S., & Hernández-Lobato, J. M. (2026). Decoupled PFNs: Identifiable epistemic-aleatoric decomposition via structured synthetic priors. arXiv:2605.06413
- Depeweg, S., Hernández-Lobato, J. M., Doshi-Velez, F., & Udluft, S. (2018). Decomposition of uncertainty in Bayesian deep learning for efficient and risk-sensitive learning. *ICML*.
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA*, 102(477), 359–378.
- Hofman, P., Sale, Y., & Hüllermeier, E. (2024). Quantifying aleatoric and epistemic uncertainty with proper scoring rules. arXiv:2404.12215
- Houlsby, N., Huszár, F., Ghahramani, Z., & Lengyel, M. (2011). Bayesian active learning for classification and preference learning. arXiv:1112.5745
- Kang, J. (2021). Low-rank method to fasten the inference of Gaussian process. STAT 243 lecture notes.
- Lahlou, S., Jain, M., Nekoei, H., Butoi, V. I., Bertin, P., Rector-Brooks, J., Korablyov, M., & Bengio, Y. (2023). DEUP: Direct epistemic uncertainty prediction. *TMLR*.
- Qu, J., Holzmüller, D., Varoquaux, G., & Le Morvan, M. (2025). TabICL: A tabular foundation model for in-context learning on large data. *ICML*.
- Qu, J., Holzmüller, D., Varoquaux, G., & Le Morvan, M. (2026). TabICLv2: A better, faster, scalable, and open tabular foundation model. arXiv:2602.11139
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*, ch. 2. MIT Press.

---

### Appendix — why realized-task truth instead of an oracle

The alternative was to grade against the **posterior-averaged** truth $\bar T=\mathbb E_{\tau\mid D}[\,\cdot\,]$, computed by an oracle: an exact posterior over tasks given each context.

- **What the oracle would buy.** A per-query target with no noise, allowing estimate-on-truth scatter plots and $R^2$.
- **What it costs.** A posterior for every (task, $n$). On the old toy prior, that took hours, needed its own Monte Carlo validation, and forced the noise prior onto a grid. On T2 and T3 it is not available at all.
- **Why the realized truth is enough.** Tasks are drawn from a fixed prior, so the realized truth averages exactly to $\bar T$ given the context (§6.1). Truth-on-estimate grading across tasks therefore gives the same answer as the oracle, at zero cost.
- **What is given up.** Per-query exactness and $R^2$. On T1 nothing is given up: the task is fixed and known, so the truth is exact per query.
