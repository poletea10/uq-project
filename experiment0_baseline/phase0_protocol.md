# Phase 0 — Setting the baseline: execution protocol

Phase 0 has six tests and trains nothing. The iLTMv2 rows use Phase 1's selected checkpoints (§5.5).

| # | Test | Question | Ground truth | Models | Output |
|---|---|---|---|---|---|
| T1 | **Toy set** (toy rungs 0–2) | *Where* does each model put $\hat A$ and $\hat E$ on one curated 1-D function? | Closed form | Heteroscedastic GP, Dec-ICL, iLTMv2 | One 3×4 context figure per (model, toy rung): 9 figures, plus one 3×3 paper figure |
| T2 | **Error heatmap** | Are $\hat A$ and $\hat E$ right across local data density and local noise? | Closed form | Heteroscedastic GP, Dec-ICL, iLTMv2 | One heatmap figure |
| T3 | **PRR-A** (total error) | Do $\hat A+\hat E$ rank the forecast's loss better than its own spread? | Realized CRPS | iLTMv2 (Dec-ICL if available) | Curves + table |
| T4 | **PRR-B** (reducible error) | Does $\hat E$ say which predictions would improve with more data? | Realized CRPS gain | iLTMv2 (Dec-ICL if available) | Curves + table |
| T5 | **Active learning** | Does the epistemic estimate select useful data? | Test-set performance | Random, TabICLv2, iLTMv2, Tuned-ICL, Dec-ICL | Table + learning curves |
| T6 | **Context retrieval** | Does $\hat E$-guided retrieval beat k-NN for a given query? | Realized CRPS | iLTMv2 | Curves + table |

T1 and T2 are the visual results: T1 gives the paper's main figure (easy to read, no metric table), and T2 extends its two situations to a whole grid of local density and local noise. T3–T6 are the thorough evaluation on real data. T3 and T4 ask whether the estimates rank the right errors; T5 and T6 ask whether $\hat E$ selects useful data. T6 is the first preview of Phase 2.

---

## 1. Layout, environments and shared constants

```
experiment0_baseline/
├── CLAUDE.md
├── configs/            # protocol.yaml, toy_function.yaml, toy_rung{0,1,2}.yaml, methods.yaml,
│                       # heatmap.yaml (T2), al.yaml (datasets and splits, T3–T6; loop, T5), retrieval.yaml (T6)
├── src/
│   ├── config.py       # YAML loaders and output paths
│   ├── toy.py          # the curated function f, noise settings, context sampler (T1)
│   ├── heatmap.py      # T2 probe-strip toys: background, strip, cells, shared draws
│   ├── truth.py        # realized A and E(q_m) in closed form (T1, T2); realized CRPS at an observed y (T3–T6)
│   ├── entropy.py      # e_CRPS kernel, excess kernel, shared t-grid builder
│   ├── forecasts.py    # forecast objects q_m (Gaussian mixture, quantile CDF): entropy, excess, CRPS
│   ├── adapters/       # base.py (UQOutput), hgp.py, decicl.py, tabiclv2.py, iltmv2.py
│   ├── evaluate.py     # method registry; per-(rung, n) evaluation of T1
│   ├── datasets.py     # the five real datasets and their stored splits (T3–T6)
│   ├── prr.py          # rejection curves and PRR (T3, T4)
│   ├── active_learning.py   # T5 loop and metrics
│   ├── retrieval.py    # T6 selection rules (random, k-NN, greedy with shortlist)
│   └── figures.py      # 3×4 context figures (T1, backbone check), 3×3 paper figure (T1), heatmap (T2),
│                       # rejection curves (T3, T4), learning and budget curves (T5, T6)
├── scripts/            # 00_curate_toy, 00a_backbone_check, 00b_validate_estimators, 00c_validate_adapters,
│                       # 01_toy (--method, --paper-figure), 02_heatmap (--method), 03_prr_a (--method --dataset),
│                       # 04_prr_b (--method --dataset), 05_active_learning (--method --dataset),
│                       # 06_retrieval (--method --dataset; --resolution-check)
└── outputs/{raw,figures,tables,logs}/
```

**Environments.**

- `uq-iltm-bio` for everything by default, including the heteroscedastic GP (PyMC with the numpyro backend) and iLTMv2, which runs through Phase 1's editable-installed `iltm2` package.
- Dec-ICL gets whatever environment its checkpoint needs, with the same parquet hand-off if it clashes with `uq-iltm-bio`. Never import a clashing env in-process.

**Shared constants** (`protocol.yaml`, fixed before the first run, never tuned):

| Constant | Value |
|---|---|
| `n_grid` | 64, 256, 1024, 4096 (context sizes; T1 and the backbone check) |
| `S` | 500 posterior samples (heteroscedastic GP) |
| `sigma_floor` | $10^{-6}$. Clip value for every predicted σ (§12, gotcha 3); $\sigma_{\text{floor}}/\sqrt\pi$ is the display floor of log-scale plots |
| `base_seed` | 0. Root of every RNG stream (§11) |

**One recorded deviation: `n_grid`.** `n_grid` is the one constant set after a first run. It was $\{16,64,256,1024\}$ when iLTMv2's toy figures were first drawn; it is now $\{64,256,1024,4096\}$. The reason does not depend on which model wins: the smallest context Phase 1 trains on is 64 rows, so $n=16$ is outside every head's training range, and $n=4096$ adds a column where the sparse strip holds about 128 points. The context is drawn once at $n=4096$ and nested; its first 1,024 points equal the earlier draw at $n=1024$ (asserted), so the columns $n\le1024$ do not change. The choice was nevertheless made after seeing iLTMv2's figures, and it is stated here as a deviation from the "fixed before the first run" rule.

---

## 2. T1 — Toy set

### 2.1 Purpose

One fixed 1-D function, one fixed context draw, three noise settings. Each model gets one figure per setting, showing along $x$ where it puts aleatoric and epistemic uncertainty against the exact truth. It follows the spirit of the toy figures in DEUP (Lahlou et al., 2023, Fig. 1 and App. E) and DecoupledPFN (Bergna et al., 2026, Fig. 1), but uses the context-figure layout of §2.6 rather than DEUP's shaded panels.

It is an illustration, not a grade: a single function cannot support calibration claims. T2 turns the same picture into numbers over many toy functions, and T3–T6 carry the evaluation on real data.

### 2.2 The curated function

Everything lives on `x_range` $=[-5,5]$:

$$
y=f(x)+\sigma(x)\,\varepsilon,\qquad \varepsilon\sim\mathcal N(0,1),\qquad f(x)=f_{\text{base}}(x)+f_{\text{gap}}(x).
$$

**Base part.** A low-rank GP (Rasmussen & Williams, 2006; Kang, 2021):

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

- $\sigma_0=e^{-0.7-\log 2}\approx0.25$;
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

- **Context.** $x$ drawn from a piecewise-uniform density on `x_range`: relative density 1 outside $[1,4]$ and `gap_density` $=0.075$ inside. Drawn once at $n=4096$ from the context stream of §11; each $n\in$ `n_grid` uses the first $n$ points (nested contexts). The gap holds about 2 points at $n=64$, 8 at $n=256$, 32 at $n=1024$ and 128 at $n=4096$, where it is still about 13 times sparser than its surroundings.
- **Why this density.** With about 12 points in the gap at $n=1024$, a reviewer could argue that the heteroscedastic GP's "noise" reading of the gap is defensible from the data alone, since 12 scattered points look like noise. With about 32, the wiggles are clearly signal, and "the GP misattributes" becomes unambiguous.
- **Queries.** `n_query` $=200$ points evenly spaced on `x_range`, shared by all toy rungs and models.
- **Truth** (§4.1). Closed form and **exact** for this context: $A=\sigma(x_*)/\sqrt\pi$ and $E(q_m)=\int(F_m-\Phi)^2dt$. There is no realized-task noise, since $f$ and $\sigma$ are known.

### 2.5 Models

| Model | Runs | Notes |
|---|---|---|
| Heteroscedastic GP | ✓ | Adapter §5.2 (validated). Priors deliberately not matched to $f$ |
| Dec-ICL | ✓ when the checkpoint arrives | Adapter §5.3. Its training prior (explicit $f$ plus Gaussian noise) is close to this toy, so T1 and T2 are, if anything, tilted *towards* Dec-ICL and the GP (§2.9). State it |
| iLTMv2 | ✓ | Adapter §5.5: Phase 1's selected arm-A checkpoints (method `iltmv2`, the paper figure's column); arm B's as `iltmv2_b` |
| TabICLv2 | backbone check only (§2.7) | Total only |

### 2.6 Figures

**One figure per (model, toy rung), 3 rows × 4 columns: 9 figures.** Columns are $n\in\{64,256,1024,4096\}$. Title: `toy rung k: <method>`.

| Row | Content | Axis |
|---|---|---|
| 1 | Context points; true mean $f$ with true 90 % band $f\pm1.645\,\sigma$; forecast median with forecast 90 % band (5 %–95 % quantiles of $q_m$) | linear |
| 2 | Realized $A$ (blue), $\hat A$ (red) and total $e(q_m)$ (black, dashed) | linear |
| 3 | Realized $E(q_m)$ (blue, **this model's own** excess) and the model's predicted excess $\hat E$ (red) | log, floor $\sigma_{\text{floor}}/\sqrt\pi$; points below it are drawn on the floor |

- Shade the noise strip (orange) and the sparse strip (purple) in every panel. There is no shading between $\hat A$ and $e(q_m)$.
- Put a rug of the context $x$-positions on rows 2–3.
- Row 3 carries a small legend naming each model's estimator: **$e(q_m)-\hat A$** for the heteroscedastic GP and Dec-ICL, **the $\hat E_{\text{dir}}$ head** for iLTMv2. $e(q_\theta)-\hat A$ is not shown for iLTMv2 (it is logged in `extras`, §5.5).
- **Within a toy rung, share the y-limits of each row across the three models' figures**, so they can be compared side by side. The one exception is row 2 of toy rung 0, which gets per-model y-limits (or a log axis), so that $\hat A\approx0$ stays visible next to a model whose $\hat A$ is large.
- For toy rung 0, row 2's truth is the zero line. Report each model's σ clip rate in a corner of row 2 (§12, gotcha 3).
- Row 2 carries $e(q_m)$ so that the split is visible at a glance. For every model whose $\hat E$ is $e_{\text{tot}}-\hat A$ (§4.4), $\hat A\le e(q_m)$, and the vertical gap between the dashed black and the red curves *is* $\hat E$. For iLTMv2 it is not: its two heads are trained separately (§5.5).

**Paper figure.** `01_toy.py --paper-figure` draws one 3×3 figure for toy rung 2 at $n=1024$: one column per model (heteroscedastic GP, Dec-ICL, iLTMv2), with the same three rows as above. It is built from the saved outputs only, once all three models have been run, and shares each row's y-limits across its three columns. It focuses on the one question the paper figure has to answer: with the same context, where does each model put aleatoric and epistemic uncertainty? The per-(model, toy rung) 3×4 figures go to the appendix, as the full record across $n$.

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

The realized excess of one function is jagged; a correct $\hat E$ is a smooth curve through it (§4.5). A match in level and shape is the expected good result, not a point-by-point match.

### 2.9 Why the comparison is fair

*The objection.* The baselines' $\hat E$ is $e(q_m)-\hat A$, and $e(q_m)$ is not the forecast's expected score under the true law, $\mathbb E_{Y\sim p_\tau}[S(q_m,Y)]$. So why should $e(q_m)-\hat A$ match the realized excess $E(q_m)=\mathbb E_{Y\sim p_\tau}[S(q_m,Y)]-e(p_\tau)$?

*The answer.* Nobody has the true law at test time, including our head. Every model can only **predict** its excess from $(x_*,D)$. The question is how each one forms that prediction.

1. A Bayesian-style model holds a set of candidate task laws $F_m$ with weights $w_m$ (posterior draws for the GP; latent values convolved with the noise model for Dec-ICL). Its forecast is their mixture, $q_m=\sum_m w_mF_m$.
2. The model believes the truth is one of its members, $F_m$ with probability $w_m$. If the truth were $F_m$, the excess would be $\mathbb E_{Y\sim F_m}[S(q_m,Y)]-e(F_m)$.
3. Averaging over its own belief, and using that the expected score is linear in the law of $Y$:

$$
\sum_m w_m\,\mathbb E_{Y\sim F_m}\big[S(q_m,Y)\big]=\mathbb E_{Y\sim q_m}\big[S(q_m,Y)\big]=e(q_m),
\qquad\text{so}\qquad
\sum_m w_m\,d_S(q_m,F_m)=e(q_m)-\hat A .
$$

So $e(q_m)$ is not the forecast's true expected score: it is **the model's own prediction of it**. Likewise $\hat A$ is the model's prediction of the irreducible floor, and $e(q_m)-\hat A$ is exactly the model's prediction of its excess. It is the standard total-minus-aleatoric decomposition (Hofman et al., 2024); for the log score the same identity gives the mutual information used in BALD (Houlsby et al., 2011; Depeweg et al., 2018).

4. *Same target as our head.* If a model's members are the exact posterior over tasks, the identity reads $e(q)-\mathbb E_{\tau\mid D}[e(p_\tau)]=\mathbb E_{\tau\mid D}[d_S(q,p_\tau)]$. The right-hand side is precisely what $\hat E_{\text{dir}}$ converges to under squared error. So both estimators aim at the same quantity, the expected excess given the context, in the same units, and each is graded against its own model's realized excess.
5. *What differs.* The baselines derive the prediction from their own posterior; ours is learned from realized excess (DEUP's idea; Lahlou et al., 2023). When a model's posterior is wrong, its self-predicted excess misses the bias. That gap is the result the figure shows; it is not an artefact of the conversion.

*A worked case.* Members $\mathcal N(\mu_m,\sigma^2)$ with $\mu_m\sim\mathcal N(\bar\mu,s^2)$ give $q=\mathcal N(\bar\mu,s^2+\sigma^2)$, $\hat A=\sigma/\sqrt\pi$ and $e(q)-\hat A=(\sqrt{s^2+\sigma^2}-\sigma)/\sqrt\pi$. Drawing the true mean from $\mathcal N(\bar\mu,s^2)$ and averaging the realized divergence gives the same value.

*Limits, stated in the caption or text:*

- The identity holds on average over tasks consistent with the context. One function cannot show calibration, so T1 is an illustration.
- Dec-ICL's native epistemic signal is the variance of its latent head, with $v_{\text{tot}}=v_{\text{epi}}+\hat\sigma^2$ (Bergna et al., 2026, App. C.1). A variance cannot be drawn against a CRPS truth, so its members $\mathcal N(c_j,\hat\sigma^2)$ are projected as above (§5.3). Its native $v_{\text{epi}}$ is used where only ranking matters (T4, T5).
- iLTMv2 is the only model trained on exactly this target. That is the method, not a bias of the test. In the other direction, the toy function is closer to the GP's and Dec-ICL's function classes than to iLTMv2's prior.

---

## 3. T2 — Error heatmap

### 3.1 What it shows

T1 shows two situations on one function: a noisy region with plenty of data, and a sparse region with little noise. T2 fills in the whole grid. For every combination of local data density and local noise, it shows whether each model's $\hat A$ and $\hat E$ are too high, right, or too low. The two strips of T1 are two cells of this grid.

### 3.2 Construction: a probe strip

Each toy function has a fixed background and one strip. The cell sets the strip's local density and local noise, and only queries inside the strip are scored. The surroundings of the strip are the same for every cell of a toy, so a cell's value is not confounded by what lies around it.

| Element | Setting |
|---|---|
| Function | The toy generator's base family on $[-5,5]$ (§2.2: 28 bumps, width 0.3, unit prior s.d.), fresh draw per toy, no gap part |
| Strip | Width 2, position drawn per toy inside $[-4,4]$, edge width 0.1 |
| Background | Per toy, drawn once and kept for all its cells: context density $d_{\text{bg}}$ log-uniform in [64, 1,024] points per unit of $x$; noise $\sigma_{\text{bg}}$ log-uniform in [0.05, 0.5] |
| x-axis (cell) | Context points inside the strip: 0, 2, 8, 32, 128, and "background" (the strip at density $d_{\text{bg}}$) |
| y-axis (cell) | Noise s.d. inside the strip: 0, 0.1, 0.25, 0.6, 1.5 |
| Queries | 50 evenly spaced points in the strip interior (0.25 away from each edge) |
| Sharing | Within a toy, all 30 cells share the function, the background context and the standard-normal noise draws |
| Toys | 20 (the first 10 for the heteroscedastic GP, which needs one NUTS fit per toy and cell) |

The grid, the distributions and the floor below are fixed before the first run (`heatmap.yaml`) and not tuned afterwards.

### 3.3 Truth and cell value

**Truth.** Closed form per query, as in T1 (§4.1): $A=\sigma(x_*)/\sqrt\pi$ and $E(q_m)=\int(F_m-\Phi_\tau)^2dt$, the model's own excess.

**Cell value.** For each model, quantity and cell:

$$
\text{value}=\log_2\frac{\text{mean estimate}}{\text{mean truth}},
$$

with means over all toys and queries of the cell, clipped at ±3.

- The mean is taken first, the ratio second. A per-query ratio is not used for $\hat E$ because of the scatter of §4.5.
- In the $\sigma=0$ row the true $A$ is 0, so a fixed floor of 0.01 is added to both means for $\hat A$.

Each model uses the estimator of §2.6's legend: $e(q_m)-\hat A$ for the heteroscedastic GP and Dec-ICL, $\hat E_{\text{dir}}$ for iLTMv2.

### 3.4 Figure and how to read it

**Figure.** One row per model, two heatmaps per row ($\hat A$ and $\hat E$). Colour is signed: **red** = the estimate is too high, **white** = correct, **blue** = too low.

**How to read it.** A model that reads sparse data as noise is red in the left columns of its $\hat A$ panel and blue in the same cells of its $\hat E$ panel. A model that decouples correctly is white in both.

---

## 4. Ground truth

### 4.1 The quantities

For a query $x_*$ in a task $\tau$ with true conditional law $p_\tau(\cdot\mid x_*)$, a model with forecast $q_m$ reports $\hat A$ and $\hat E$ (§5). On T1 and T2 they are compared with the **realized** truth of that task:

$$
A=e\big(p_\tau(\cdot\mid x_*)\big),\qquad
E(q_m)=\mathbb E_{Y\sim p_\tau}\big[S(q_m,Y)\big]-A=d_S\big(q_m,p_\tau\big).
$$

- $A$ is the score that a forecaster who *knew the task* would still pay: the irreducible floor.
- $E(q_m)$ is the extra score that *this model's own forecast* pays on top of the floor: its excess risk (Hofman et al., 2024).

**Why the model's own excess.** What a user acts on is "how far off is this model's forecast likely to be, beyond the irreducible noise?". Comparing with $E(q_m)$ asks whether a model **knows its own error**. For the baselines, $\hat E=e(q_m)-\hat A$ is the model's self-reported spread beyond its aleatoric part, which equals its true excess only if its members behave like a true posterior over tasks (§2.9, §4.4); the gap between the two is what the tests measure. For iLTMv2, $\hat E_{\text{dir}}$ is trained to target exactly $E(q_\theta)$ (proposal §4.4). $A$ needs no such choice: it is the same for every model.

**T1, T2 (closed form).** $p_\tau=\mathcal N\big(f(x_*),\sigma(x_*)^2\big)$, so with CRPS (Gneiting & Raftery, 2007)

$$
A=\frac{\sigma(x_*)}{\sqrt\pi},\qquad E(q_m)=\int\big(F_m(t)-\Phi_\tau(t)\big)^2dt,
$$

the Cramér distance on the shared $t$-grid (§4.3), with $\Phi_\tau$ a step at $f(x_*)$ where $\sigma(x_*)=0$ (toy rung 0; the $\sigma=0$ row of T2).

### 4.2 Realized loss on real data (T3–T6)

On real data the task law is unknown, and the only ground truth is the observed label $y$ of each test point. Its realized loss is

$$
\ell=\mathrm{CRPS}(q_m,y)=\int\big(F_m(t)-\mathbb 1\{t\ge y\}\big)^2dt,
$$

with the CDF convention of §5.4. Its expectation over $y\sim p_\tau(\cdot\mid x_*)$ is the expected score $A+E(q_m)$, so a single $\ell$ is an unbiased but noisy draw of it. This is why T3 ranks $\ell$ with total uncertainty, why T4 takes a difference of two losses at the same $y$ to cancel $A$, and why T6 averages $\ell$ over queries. None of them compares an estimate with a single realized value (§4.5).

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

**The $t$-grid.** 4096 points spanning $\pm10$ s.d. around the forecast, **widened to also cover the truth**: the true law on T1 and T2, the observed $y$ on T3–T6. One grid construction is shared by every adapter and by `truth.py`. Assert that both CDFs reach their tails inside the grid ($<10^{-4}$ at each end).

**Unit tests.**

- $e_{\mathrm{CRPS}}(\mathcal N(0,\sigma^2))=\sigma/\sqrt\pi$ to $10^{-6}$.
- Agreement with $\tfrac12\mathbb E|Y-Y'|$ by Monte Carlo on a two-component mixture.
- `excess` against the closed-form CRPS of a Gaussian forecast at a point.

**The member-wise rule.** $\hat A$ is the average of member entropies, never a conversion of a pre-aggregated "data uncertainty" scalar. $\mathrm{Var}=\mathbb E[\sigma^2]+\mathrm{Var}[\mu]$ is additive in variance; $\sigma/\sqrt\pi$ is not. A negative $\hat E$ from `project` always means this rule was broken.

### 4.4 What $e_{\text{tot}}-\hat A$ measures

The heteroscedastic GP and Dec-ICL report $\hat E=e_{\text{tot}}-\hat A$ (§5.2, §5.3). This is **not** an approximation of the realized divergence $d_S(q_m,p_\tau)$. It is the model's own posterior expectation of that divergence, $\sum_m w_m\,d_S(q_m,F_m)$, and that is an exact identity (derivation in §2.9; it is the decomposition of proposal §2.2). It is non-negative by concavity of $e$, which is what the assertion in `project` checks. It is unbiased for the realized excess only **on average over the tasks consistent with the context**, and only when the members are the exact posterior; when the model is misspecified, the self-predicted excess misses the bias (DEUP's argument; Lahlou et al., 2023, §2). iLTMv2's $\hat E_{\text{dir}}$ is instead trained on the realized excess itself (proposal §4.4).

**No general ordering between $e(q_m)$ and the realized divergence.** On a single task, either can be much larger:

- *Sharp and wrong.* As the forecast's spread goes to 0, $e(q_m)\to0$ while $d_S$ tends to the size of the error (on toy rung 0, $d_S\to|\delta|$ for a point forecast off by $\delta$). Here $d_S\gg e(q_m)$.
- *Exact.* $q_m=p_\tau$ gives $d_S=0$ and $e(q_m)=A>0$.
- *Centred but over-dispersed.* $q_m=\mathcal N(f,(1.2\sigma)^2)$ against $p_\tau=\mathcal N(f,\sigma^2)$ gives $e(q_m)-A=0.2\sigma/\sqrt\pi\approx0.11\sigma$ but $d_S=\big(\sqrt{2\cdot2.44/\pi}-1/\sqrt\pi-1.2/\sqrt\pi\big)\sigma\approx0.005\sigma$: near the truth the divergence is second order in the spread error, while $e(q_m)-A$ is first order.

An ordering holds only in expectation, for an exact Bayesian: $\mathbb E_{\tau\mid D}[d_S]=e(q)-\mathbb E_{\tau\mid D}[A]\le e(q)$. On average, an exact Bayesian is never more wrong than its own spread suggests; on a single task it can be.

### 4.5 The epistemic scatter and the performance ceiling

**The statement.** No estimator can predict the realized excess of a single task exactly. It can only predict the average excess over the tasks that are consistent with the context.

**Why.** The same context $D$ is compatible with many tasks, and each gives a different true law at the query. Tasks are drawn from a fixed prior, so for any query

$$
\mathbb E\big[E(q_m)\mid D,x_*\big]=\bar E(D,x_*)=\mathbb E_{\tau\mid D}\big[d_S(q_m,p_\tau)\big].
$$

Every estimate is a function of $(D,x_*)$, so $\bar E$ is the best any estimate can do. The realized value equals $\bar E$ plus a zero-mean deviation that the context cannot resolve: the **scatter**. If the deviation were predictable from the context, the forecast itself could be corrected with it.

**How large it is.** The scatter is of the same order as the quantity itself. For a calibrated Gaussian forecast its coefficient of variation is between 0.72 (deterministic task) and 1.41 (noise-dominated task). With a coefficient of variation CV, the best possible error is $\mathrm{CV}^2/(1+\mathrm{CV}^2)$ times the target's second moment, so a perfect estimator has a skill well below 1. On the Phase 1 validation bank, the ceiling of the pooled $\hat E$ skill is bracketed between 0.44 and 0.64 (`phase1_design.md`, validation section).

**What it means for each test.**

- **T1 (toy set).** The realized excess of one function is jagged; a correct $\hat E$ is a smooth curve through it. A match in level and shape is the expected good result, not a point-by-point match.
- **T2 (heatmap).** Cell values compare means over many toys and queries, where the scatter averages out. This is why the ratio is taken after the mean.
- **T3, T4 (PRR).** The oracle ordering uses each point's realized loss, which no score can know. A perfect score has a PRR well below 1. PRR values compare scores on the same forecast and dataset; they are not comparable across datasets.

**The same point in DEUP.** Lahlou et al. (2023, Fig. 6c and App. D) correlate predicted uncertainty with model residuals and note that the achievable correlation is far below 1. They estimate a best-case upper bound by sampling residuals from each model's own predictive distribution (0.63 ± 0.05 for DEUP against an observed 0.47 ± 0.03) and report the ratio of observed to achievable as their comparison metric. For the same reason, $R^2$ and rank correlation of an estimate against single realized values are not used as grades anywhere in Phase 0.

*Optional reference line for T3.* The same idea applied to PRR: sample $y$ from the model's own forecast and compute the PRR of its score. It shows what PRR the score would get if the forecast were exactly right.

---

## 5. Adapters

Every adapter returns a weighted family of predictive CDFs, or a closed form equivalent to one, and the kernel does the rest.

```python
@dataclass
class UQOutput:
    A_hat: np.ndarray       # (n_query,) — y-units
    E_hat: np.ndarray
    e_total: np.ndarray
    cdf: Callable           # the forecast q_m, needed for the truth E(q_m) and the realized loss
    fit_seconds: float
    extras: dict            # native outputs, unprojected, for audit
```

Always fill `extras`: when a figure looks wrong you will want the native numbers without re-running. **Each method runs with its own default preprocessing**; whatever it does internally, its adapter returns quantities in $y$ units.

### 5.1 Which model runs where

| Model | T1 toy | T2 heatmap | T3 PRR-A | T4 PRR-B | T5 active learning | T6 retrieval |
|---|---|---|---|---|---|---|
| Heteroscedastic GP | ✓ | ✓ (first 10 toys) | — | — | — | — |
| Dec-ICL | ✓ if checkpoint | ✓ if checkpoint | ✓ if checkpoint | ✓ if checkpoint | ✓ if checkpoint, else reported numbers | — |
| Tuned-ICL | — | — | ✓ if checkpoint (total, as context) | — | ✓ if checkpoint, else reported numbers | — |
| TabICLv2 | backbone check (§2.7) | — | — | — | ✓ | — |
| iLTMv2 | ✓ (§5.5) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Random (TabICLv2 predictor) | — | — | — | — | ✓ | — |

### 5.2 Heteroscedastic GP (PyMC)

The adapter (`src/adapters/hgp.py`) is working and validated, with a learned constant mean on the log-noise latent (below). It follows the PyMC example "Heteroskedastic Gaussian Processes", section "Sparse Heteroskedastic GP". For reference:

- **Model.** Two latent functions, the mean $\mu(x)$ and the log noise scale $g(x)$, with $y\sim\mathcal N\big(\mu(x),e^{2g(x)}\big)$. $\mu$ has a zero mean function; $g$ has a learned constant mean $m_g\sim\mathcal N(-2,2^2)$ in standardized units (a noise s.d. of about $0.14$ s.d. of $y$ at the prior centre).
- **Approximation.** Each latent is a DTC sparse GP over 40 inducing points evenly spaced on `x_range` (a fixed grid keeps the cost flat in $n$), with whitened inducing values: $u=Lv$, $v\sim\mathcal N(0,I)$, $L=\mathrm{chol}(K_{uu}+10^{-6}I)$, $f(X)=K_{fu}K_{uu}^{-1}u$.
- **Priors.** Generic squared-exponential kernels $\eta^2\,\mathrm{ExpQuad}(\ell)$, deliberately *not* matched to the toy function: $\ell\sim\mathrm{InverseGamma}(\mu_\ell,\sigma_\ell)$ with $(\mu_\ell,\sigma_\ell)$ set from the spread of pairwise context distances (the example's `get_l_prior`); $\eta\sim\mathrm{Gamma}(2,1)$. $y$ is standardized with context statistics before fitting and outputs are mapped back to $y$ units.
- **Inference.** NUTS through numpyro, `target_accept` $=0.95$, 2 chains × (500 tuning + 250 draws), giving $S=500$. Log R-hat, divergences, fit time and the posterior mean of $m_g$ in `extras`; flag fits with R-hat $>1.05$.
- **Prediction.** The example's `conditional`: per posterior draw, mean $K_{su}K_{uu}^{-1}u$ plus a draw from the conditional variance $\mathrm{diag}(K_{ss}-Q_{ss})$. Only marginals at each query enter a member CDF, so the diagonal is exact here.
- **Members.** $\{\mathcal N(\mu_s(x_*),\sigma_s(x_*)^2)\}_{s=1}^S$, $w_s=1/S$, through `project`, so $\hat A=\frac1S\sum_s\sigma_s/\sqrt\pi$.
- **Toy rung 0** (and the $\sigma=0$ row of T2). $\sigma\equiv0$ pushes $g$ towards $-\infty$. Clip $\sigma_s$ at `sigma_floor` and report the clip rate.

**Why the log-noise latent has a constant mean.** With zero mean functions for both latents, the GP reported $\hat A\approx420$ in the sparse strip of toy rung 0 at $n=256$ (and about 100 at $n=1024$), on a deterministic function whose whole range is under 8 units. The σ clip rate was 0 %, so the clip did not cause it: the clip only floors σ from below. The forecast band itself left the plot, so this was what the fitted model reported, and nearly all of it was aleatoric ($\hat E\approx3$ there). Two effects stacked:

1. **A genuine one.** The few gap points deviate from the smooth mean, and raising the noise latent there is cheaper in marginal likelihood than shortening the lengthscale everywhere, so the misfit is read as noise. This is a real property of the heteroscedastic GP, also visible on toy rung 2, and worth showing.
2. **An artifact of the parameterization.** With noiseless data, $g$ must sit far below zero in the dense regions, which a zero-mean GP can only do with a large amplitude. In the data-free gap, $g$ then reverts towards 0 with that large variance, and because $\hat A$ averages $\sigma_s=e^{g_s}$ over draws (a lognormal mean), the upper tail of the draws dominates it.

A learned constant mean lets $g$ sit low without a large amplitude, so data-free regions revert to the task's typical noise level. The first effect stays visible without the second. The mean and its prior are generic choices, independent of the toy function.

### 5.3 Dec-ICL (Bergna et al., 2026)

The decoupled TabICL model, obtained from the authors. Until the checkpoint arrives, the adapter is written against the interface below and unit-tested on synthetic outputs (§12, gotcha 2). **Confirm the exact output format with the authors' code before finalizing.**

**Native outputs** (Bergna et al., 2026, §3.1):

- a latent-signal head: probabilities $\pi_j$ over fixed target bins $a_0<\dots<a_K$ with centres $c_j$, a distribution over the noiseless $f(x_*)$;
- an aleatoric head: the observation-noise variance $\hat\sigma^2(x_*)$.

The observation-level predictive is their convolution.

**Conversion.** Members are the latent values $u$ (under whatever within-bin law their code uses), each with law $\mathcal N(u,\hat\sigma^2)$. Every member shares the same $\hat\sigma$, so

$$
\hat A=\frac{\hat\sigma}{\sqrt\pi}\ \ \text{exactly},\qquad e_{\text{tot}}=e_{\mathrm{CRPS}}(q_m),\qquad \hat E=e_{\text{tot}}-\hat A,
$$

where $q_m$ is **their** observation-level predictive, reproduced exactly as their code builds it (point masses at $c_j$, or mass spread within bins). Any spread *within* a latent bin is uncertainty about $f$, so it belongs to $\hat E$, never to $\hat A$. If their aleatoric head outputs a distribution over $\sigma^2$ rather than a point, use $\hat A=\mathbb E[\sigma]/\sqrt\pi$ over it instead.

Where only a ranking is needed (T4, T5), Dec-ICL uses its native scores: the latent variance $v_{\text{epi}}$ and the total $v_{\text{tot}}=v_{\text{epi}}+\hat\sigma^2$ (§2.9).

**Checks and logs.**

- Unit test: our $q_m$, integrated over their observation bins, reproduces their observation-level bin masses to $10^{-6}$.
- On toy rung 0, $\hat\sigma\to0$: clip at `sigma_floor` and report the clip rate. The latent bin width puts a floor under $e_{\text{tot}}$, so log the bin width in $y$ units.
- Log $\pi$, $\hat\sigma$ and their native latent variance in `extras`.

**Backbone confound.** Dec-ICL is fine-tuned from TabICL (Qu et al., 2025) on the authors' own heteroscedastic prior; iLTMv2 is built on TabICLv2 (Qu et al., 2026). Confirm the backbone version with the authors. Any Dec-ICL versus iLTMv2 gap mixes method, backbone and prior, and the thesis says so.

### 5.4 TabICLv2 (Qu et al., 2026)

Total only. `e_total` and `cdf` come from the 999 raw quantiles at levels $k/1000$: the CDF is piecewise-linear through them, 0 below the first and 1 above the last, and its entropy is integrated exactly segment by segment. `A_hat` and `E_hat` are `None`; the evaluator skips them rather than coercing them to zero. Used in the backbone check (§2.7) and in T5.

### 5.5 iLTMv2

$\hat A$ from the aleatoric head and $\hat E=\hat E_{\text{dir}}$ from the epistemic head (proposal §4.4); `e_total` and `cdf` from the quantile head. Also log $e(q_\theta)-\hat A$ in `extras`, which is the amortization gap of proposal §4.7.

**Implementation** (`src/adapters/iltmv2.py`). The model is `iltm2.inference.ILTMv2Regressor` (experiment1): `fit` on the context, `predict_uncertainty` on the queries, which returns the 999 raw quantiles, $\hat A$, $\hat E_{\text{dir}}$ and $e(q_\theta)$ in $y$ units.
- **Forward route.** Phase 1's raw training-mode forward of the frozen TabICLv2 (`phase1_design.md`, Architecture, "Forward route"), the route the heads were trained on, not `TabICLRegressor`. X and y are scaled as the training prior scaled them, with constants from the context only (robust clamp at mean ± 4 std, then standardization); quantiles map back affinely, and $\hat A$, $\hat E$, $e(q_\theta)$ scale with the context's y std.
- **Conversion.** $\hat A$ and $\hat E$ are passed through unchanged: the heads predict them, so there is no member family to project. $q_m$ is the CDF piecewise-linear through the 999 quantiles, the convention of §5.4, and `e_total` is its exact entropy.
- **`e_total` is computed, never assembled.** It is $e(q_m)=\int F_m(1-F_m)$ of the forecast itself, integrated exactly from its quantiles, never $\hat A+\hat E$. The two heads are trained separately and nothing ties their sum to $e(q_m)$, so in the figures (row 2: $e(q_m)$ dashed black, $\hat A$ red) $\hat A$ tracking $e(q_m)$ wherever the context is dense, and falling below it only where epistemic uncertainty rises, is a learned behaviour of the aleatoric head, not a construction. For the same reason $e(q_m)-\hat A$ is close to $\hat E$ but not identical to it: $\hat A+\hat E$ estimates the forecast's expected score $A+E(q_m)$, which equals $e(q_m)$ only when $q_m$ is the exact Bayes predictive (the CRPS divergence is symmetric). This is what T3 tests on real data. $e(q_m)-\hat A$ itself is logged in `extras` as the amortization gap (proposal §4.7) and is not drawn. `extras` holds the quantiles, both heads' outputs, $e(q_\theta)$, the amortization gap, the y scale and the checkpoints. Unit test: `00c` (conversion only).
- **Which checkpoints** (`configs/methods.yaml`). `iltmv2`: arm A's checkpoints selected on Phase 1's validation bank; `iltmv2_b`: arm B's selected configuration (fork + both heads). `checkpoint_A` / `checkpoint_E` / `pair` select any other checkpoint of either arm (A, B), for example the candidates of the selection below.
- **Which model is run where.** Phase 1 selects learning rates on its validation bank; the final model (arm and loss exponent $p$) is chosen among the four candidates (arms A and B, $p=0$ and $p=1$) by running them on T2–T5 (`phase1_design.md`, sweep and selection). The final model is then run on T1 and T6. The ablation counterparts of the final model (the other $p$, the other arm, the i.i.d. prior) run on T2–T5.
- **Not the same forecast as the TabICLv2 row.** The TabICLv2 adapter (§5.4, backbone check, T5) runs `TabICLRegressor` with its preprocessing and ensemble, so its $q_m$ differs from iLTMv2's. Compare their forecasts only through this route.

---

## 6. Real data: datasets and splits (T3–T6)

T3–T6 share the five regression datasets of DecoupledPFN's active-learning protocol (Bergna et al., 2026, §5.3 and App. G) and one set of stored splits.

| Dataset | Dim | $n_0$ (warm start) | $n_q$ (acquisitions) | Pool | Test | Seeds | Source |
|---|---|---|---|---|---|---|---|
| California Housing | 8 | 1024 | 1024 | 5000 | 3000 | 16 | TBD |
| Kin8nm | 8 | 1024 | 1024 | 5000 | 3000 | 20 | TBD |
| NYC Taxi | 7 | 1024 | 1024 | 5000 | 3000 | 19 | TBD |
| Airlines | 5 | 1024 | 1024 | 5000 | 3000 | 20 | TBD |
| YearPredictionMSD | 90 | 1024 | 1024 | 5000 | 3000 | 20 | TBD |

> **TBD — dataset sources.** Exact OpenML IDs and versions (the paper does not give them).

- **Splits.** For each (dataset, seed), fix the pool/test split and the $n_0$ warm-start indices once, and share them across all methods and all four tests. Store them in `outputs/raw/`, as the paper does (App. G.5). They are ordinary random splits: no carved regions.
- **Uses.** T3: context = warm-start set + pool (6,024 rows), queries = the test set. T4: the same full context, plus a small context = the first 256 rows of the warm-start set. T5: the warm-start set grows by acquisition from the pool. T6: starting context = the first 256 rows of the warm-start set, the pool as the labelled candidates, queries drawn from the test set.
- **Test points are never context rows**, in any of the four tests.

---

## 7. T3 — PRR-A, ranking total error

### 7.1 What it shows

Whether the two heads together predict how large the forecast's loss will be, better than the forecast's own spread does.

- $e(q_\theta)$ is what the forecast expects to pay if it is right about itself.
- $\hat A+\hat E_{\text{dir}}$ is the learned estimate of what it actually pays (the expected score of $q_\theta$; proposal §2.2, §5.5 above).
- Both score the same forecast and the same errors, so the comparison is clean.

The observed loss on real data includes noise (§4.2), so total uncertainty is the right ranker of it. That is why the sum is used here.

### 7.2 Construction

1. **Data.** The five datasets and their stored splits (§6).
2. **Forecast.** Context = warm-start set + pool (6,024 rows); queries = the test set. One raw forward of iLTMv2, the route the heads were trained on (§5.5).
3. **Loss.** $\ell_i=\mathrm{CRPS}(q_i,y_i)$ per test point, with the CDF convention used everywhere else (§4.2, §5.4).
4. **Scores.** $s_i^{\text{tot}}=e(q_i)$ and $s_i^{\text{dec}}=\hat A_i+\hat E_i$.
5. **Rejection curve.** Sort the test points by a score, highest first. After rejecting the first $k$, the curve value is the summed loss of the retained points divided by the total count $N$. Rejected points contribute 0: they are handed to an oracle that knows the outcome, and a point mass at the true $y$ has a CRPS of exactly 0.
6. **Reference curves.** Random rejection is the straight line from the mean loss to 0. The oracle ordering sorts by $\ell_i$ itself.
7. **PRR.**

$$
\mathrm{PRR}=\frac{AU_{\text{random}}-AU_{\text{score}}}{AU_{\text{random}}-AU_{\text{oracle}}},
$$

with $AU$ the area under each curve. 1 is the oracle ordering, 0 is random, negative is worse than random.

### 7.3 Outputs

- **Figure:** one panel per dataset with four curves (oracle, $e(q)$, $\hat A+\hat E$, random), averaged over seeds.
- **Table:** per dataset, the PRR of $e(q)$ and of $\hat A+\hat E$ (mean ± s.d. over seeds) and their paired difference.

**If the Dec-ICL checkpoints arrive.** Dec-ICL's decomposition sums to its total by construction, so on its own forecast it has one score (its total), shown next to Tuned-ICL's total as context.

**Expectation.** The two scores differ only where the backbone is wrong about itself, so the gap on ordinary splits may be small. This is stated before running.

A perfect score has a PRR well below 1 (§4.5); the optional reference line of §4.5 shows what PRR the score would reach if the forecast were exactly right.

---

## 8. T4 — PRR-B, ranking reducible error

### 8.1 What it shows

Whether $\hat E$ tells which predictions would improve with more data. This is what "epistemic" means operationally, and it gives an epistemic ground truth on real data without replicates.

**The idea.** Predict each test point twice, with a small context and with the full one, and rank the improvement:

$$
\Delta_i=\mathrm{CRPS}\big(q_i^{\text{small}},y_i\big)-\mathrm{CRPS}\big(q_i^{\text{full}},y_i\big).
$$

In expectation, $\Delta_i$ is the excess with the small context minus the excess with the full one. The aleatoric term is the same in both and cancels. So $\Delta$ measures reducible error only.

### 8.2 Construction

1. **Data.** Same datasets, splits and seeds as T3 (§6).
2. **Contexts.** Small = the first 256 rows of the warm-start set. Full = warm-start set + pool (6,024 rows).
3. **Scores.** From the small-context forward only: $\hat E_i$ against $e(q_i)$, the total-uncertainty baseline. Nothing from the full-context forward is used for ranking.
4. **Curve and PRR.** As in T3 with $\ell_i$ replaced by $\Delta_i$. The oracle ordering sorts by $\Delta_i$.

### 8.3 Outputs and reading notes

**Outputs.** As in T3: one panel per dataset (oracle, $e(q)$, $\hat E$, random) and a table with the PRR of both scores per dataset.

**Reading notes.**

- $\Delta_i$ can be negative for single points, so a curve can dip below 0 before returning to it. The PRR formula is unchanged.
- $\Delta_i$ is a difference of two noisy losses, so absolute PRR values will be low (§4.5). The comparison between the two scores is what counts.

**If the Dec-ICL checkpoints arrive.** Same test on its own forecasts: $v_{\text{epi}}$ against $v_{\text{tot}}$. The comparison across models is then how much each decomposition gains over its own total, as in T5.

---

## 9. T5 — Active learning

Reuses the active-learning protocol of DecoupledPFN (Bergna et al., 2026, §5.3 and App. G). It checks whether the epistemic estimate **selects useful data**, not whether it is numerically right. With T6, it is the cheapest preview of Phase 2 (context retrieval without training).

### 9.1 Protocol

- **Data and splits.** §6.
- **Loop.** At each step the model, with the current labelled set as context, scores every remaining pool point. The highest-scoring point is labelled and added. 1024 steps, one point at a time, no training.
- **Metrics.** On the test set: RMSE, MAE, Gaussian NLL (from the forecast's mean and variance, as in the paper's Table 6), and CRPS. Report final-step values as mean ± std over seeds, plus learning curves of RMSE and CRPS against the number of labelled points.

### 9.2 Methods

| Method | Predictor | Acquisition score | Status |
|---|---|---|---|
| Random | TabICLv2 | uniform over the pool | run |
| TabICLv2 (total) | TabICLv2 | $e(q_\theta)$, total CRPS entropy | run |
| iLTMv2 | iLTMv2 | $\hat E_{\text{dir}}$ | run on Phase 1's selected checkpoints (§5.5) |
| Tuned-ICL | Tuned-ICL | total predictive variance, as in the paper | run if checkpoint, else paper's Table 2 |
| Dec-ICL | Dec-ICL | latent (epistemic) variance, as in the paper | run if checkpoint, else paper's Table 2 |

- **Matched comparisons.** TabICLv2 (total) against iLTMv2: same backbone family and loop, differing only in the uncertainty signal. Tuned-ICL against Dec-ICL is the paper's own matched pair. Comparing the two pairs shows whether our decomposition buys as much as theirs over its own total-uncertainty baseline.
- **If the checkpoints arrive,** Tuned-ICL and Dec-ICL run on **our** shared splits with their native acquisition scores, and the comparison is split for split.
- **If they do not,** their rows carry the paper's numbers, marked "reported", read within their reported spread. Those were evaluated on unreleased splits, and the paper's Random row uses the Dec-ICL backbone, so there is no exact reproduction target.

---

## 10. T6 — Context retrieval

### 10.1 What it shows

For one query and a fixed starting context: given an extra labelled pool and a budget of $N$ rows, which selection rule gives the best forecast at that query? It compares $\hat E$-guided selection with k-NN and with random selection. It is a prelude to Phase 2, run on a Phase 1 checkpoint with no extra training.

### 10.2 Set-up

| Element | Setting |
|---|---|
| Data | The five datasets and splits of §6 (start with one dataset, 5 seeds) |
| Starting context $S_0$ | The first 256 rows of the warm-start set |
| Pool | The 5,000-row pool, labels known |
| Queries | 200 points drawn from the test set |
| Budgets $N$ | 16, 64, 256 |

### 10.3 Methods

| Method | Rule |
|---|---|
| Start only | $S_0$, no additions (reference) |
| Random | $N$ random pool rows, the same for all queries |
| k-NN | The $N$ pool rows nearest to $x_*$ in standardized $X$ |
| $\hat E$-greedy | Below |
| $e(q)$-greedy | Same procedure with $e(q)$ as the score (control: is the gain due to the epistemic head or to the greedy search?) |
| Whole pool | $S_0$ + all 5,000 rows (reference) |

**$\hat E$-greedy.** Selection runs in 8 rounds, adding $b=N/8$ rows per round. In each round, for each query:

1. Build a shortlist of 64 candidates from the remaining pool: the 32 nearest to $x_*$ and 32 drawn uniformly at random.
2. For each candidate $c$, run the model with context $S\cup\{c\}$ and read $\hat E_{\text{dir}}(x_*,S\cup\{c\})$.
3. Add the $b$ candidates with the lowest value to $S$.

This costs about 500 small forwards per query at any budget.

**Why a mixed shortlist.** Scoring the whole pool every round is too expensive. A shortlist of nearest neighbours only would let $\hat E$ do no more than reorder k-NN's own picks, which contradicts the reason for the test: informative rows may be spread out, not near (proposal §5.2). A purely random shortlist would almost never contain the nearest rows and would handicap $\hat E$ against k-NN, which sees the whole pool. The mixed shortlist lets $\hat E$ take k-NN's picks when they are best and go elsewhere when they are not.

### 10.4 Metric and reporting

**Metric.** Realized $\mathrm{CRPS}(q,y)$ at the query (§4.2), averaged over queries, per budget, paired across methods. Never $\hat E$ itself.

**Reported alongside.**

- The share of selected rows that came from the random half of the shortlist. Near zero means $\hat E$ behaves like k-NN, which is a finding in itself.
- Exhaustive scoring of the whole pool for 20 queries, to show what the shortlist loses.

**Check before building** (`06_retrieval.py --resolution-check`). Adding one row to 256 may move $\hat E$ by less than the head can resolve. For a few queries, look at the spread of $\hat E(x_*,S\cup\{c\})$ over candidates. If it is flat, score blocks of 8 rows instead of single rows.

**Cautions, stated with the results.**

- **Selection sees the labels.** The heads' targets are valid because every training split rule looks at $X$ only. Greedy selection uses each candidate's label through the forward pass, so $\hat E$ can be biased low on the chosen context. This is why the metric is the realized loss.
- **The whole-pool line may win.** TabICLv2 takes tens of thousands of context rows, so a small budget is artificial at these dataset sizes.
- **Use a checkpoint trained with subsampled contexts.** Phase 1's stage-1 subsampled pass (`phase1_design.md`, task construction) shows the heads the same task under two contexts; it is what teaches the head how $\hat E$ changes with the context.

---

## 11. RNG streams

Every RNG stream hangs off `base_seed`:

- **T1:** function `toy_seed` (frozen by curation); context `[base, toy, 1]`, shared across toy rungs; noise $\varepsilon$ `[base, toy, 2]`, shared across toy rungs; model fit `[base, toy, rung, 3, n]`.
- **T2:** one stream family per toy for the function, the strip position, the background ($d_{\text{bg}}$, $\sigma_{\text{bg}}$, context $x$) and the standard-normal noise draws, all shared by the toy's 30 cells; the in-strip context points of a cell are the first $n$ of one nested draw per toy; model fit per (toy, cell). Stream layout TBD when `heatmap.py` is written.
- **T3–T6:** splits as stored in `outputs/raw/` (§6). T6's query draw, the random halves of the shortlists and the Random method's rows get their own streams per (dataset, seed); layout TBD when `retrieval.py` is written.

One parquet per (test, method, configuration, $n$ or cell, seed) in `outputs/raw/`, reused on re-runs; figures go to `outputs/figures/`, tables to `outputs/tables/`.

---

## 12. Gotchas

1. **The truth code is a model too.** Validate it before running any model. `00b_validate_estimators.py`:
   - **$A$:** on toy rungs 0–2, at 50 queries with $10^5$ replicate pairs, $\tfrac12|y^{(1)}-y^{(2)}|$ averaged per query must lie within 3 Monte Carlo standard errors of $\sigma(x_*)/\sqrt\pi$. This shows exactly 0 on toy rung 0, a flat line on toy rung 1, and the noise bump on toy rung 2.
   - **$E$:** for a fixed, deliberately wrong Gaussian forecast, the average over $R=100$ replicates of $\mathrm{CRPS}(q_m,y^{(r)})$ minus the unbiased replicate entropy $\frac{1}{R(R-1)}\sum_{r<s}|y^{(r)}-y^{(s)}|$, repeated 1000 times, must average to the closed-form $\int(F_m-\Phi_\tau)^2dt$ within 3 standard errors. This checks the closed-form excess (T1, T2) and the CRPS at an observation (§4.2, T3–T6) against each other; it also reports the per-query s.d. at $R=100$.

   The replicate formulas are Phase 1's target estimators (proposal §2); auditing them here, once, lets Phase 1 cite this check instead of repeating it.
2. **Test the conversion, not the model.** `00c_validate_adapters.py` unit-tests each adapter's path from native outputs to `UQOutput` with synthetic native outputs and closed-form answers, no model fit:
   - $M$ identical members $\mathcal N(\mu,s^2)$ must give $\hat A=e_{\text{tot}}=s/\sqrt\pi$ and $\hat E=0$;
   - a Gaussian mixture must match the folded-normal closed form;
   - Dec-ICL: a one-hot latent distribution must give $\hat E\approx0$ (up to the bin-width floor) and $\hat A=\hat\sigma/\sqrt\pi$; our $q_m$ must reproduce their observation-level bin masses;
   - iLTMv2: quantiles and heads pass through unchanged, and `e_total` is the quantile CDF's own entropy, never $\hat A+\hat E$;
   - use several query rows with distinct parameters, so that an axis mix-up fails.

   Tolerances: $10^{-6}$ relative on $\hat A$ and $e_{\text{tot}}$; $10^{-6}$ of $e_{\text{tot}}$ absolute on $\hat E$.
3. **$\sigma\to0$.** Log-parameterized heads diverge on toy rung 0 and in the $\sigma=0$ row of T2. Clip every predicted σ at `sigma_floor` and **report the clip rate**: a method that passes toy rung 0 only because of the clip has not passed it.
4. **No tuning on query points.** Baselines may use a validation split carved from the context, never the queries. On T3–T6, test points never enter the context (§6).
5. **Fix the row set, the toy function and the T2 grid before the first run.** Adding methods after seeing results, re-curating $f$ after seeing figures, or re-tuning the heatmap grid after seeing cells is how benchmarks become advocacy. The one recorded deviation is `n_grid` (§1).
6. **Mean first, ratio second.** A single realized excess or loss carries the scatter of §4.5. T2 takes the ratio of means; T3 and T4 rank realized losses; T6 averages realized losses over queries. Never grade $\hat E$ against one realized value, and never report $R^2$ or rank correlation against realized truth as a grade.

---

## 13. Order of work

| # | Task | Size |
|---|---|---|
| 1 | `entropy.py` and its unit tests; `truth.py` + `00b` estimator checks | half a day |
| 2 | Toy function + curation script + toy rung configs | half a day |
| 3 | Heteroscedastic GP adapter; run it on toy rungs 0–2 | half a day |
| 4 | `figures.py` (3×4 context figures); GP figures | half a day |
| 5 | TabICLv2 adapter + backbone check (§2.7). **Decision point:** keep $f$ or switch to the fallback note | half a day |
| 6 | Toy set at `n_grid` $=\{64,256,1024,4096\}$ (nested-context assert, §1), row-3 estimator legend, per-model row-2 limits on toy rung 0; iLTMv2 adapter + `00c` | TBD |
| 7 | Dec-ICL adapter against the interface + `00c` unit tests; run when the checkpoint arrives | 1 day |
| 8 | T2: probe-strip generator, cells, heatmap figure; heteroscedastic GP on its 10 toys | TBD |
| 9 | Datasets and stored splits (§6); T5 harness: loop, metrics; Random + TabICLv2 (total); Tuned-ICL / Dec-ICL if available | 1.5 days |
| 10 | T3 and T4: PRR scripts on the stored splits | TBD |
| 11 | T6: retrieval script, starting with the resolution check (§10.4) | TBD |

Items 1–5 already give a reportable result: the toy figures for the heteroscedastic GP, plus the motivating TabICLv2 figure. The baselines (heteroscedastic GP, the TabICLv2 rows, Dec-ICL if its checkpoint arrives) are run on T1, T2 and T5 once items 6–11 are written. After Phase 1's selection, the iLTMv2 candidates run on T2–T5, and the final model on T1 and T6 (§5.5).

---

## 14. Open items

1. Dec-ICL and Tuned-ICL checkpoints (authors contacted); Dec-ICL output format and backbone version.
2. Dataset sources of §6.
3. Phase 0 kill criterion.
4. **Model selection on the test data.** The final iLTMv2 model and its $p$ are chosen with T2–T5 (§5.5), and the same tests are then reported. That is mildly optimistic. Either state it with the results, or select on a fixed subset (for example T2 plus one dataset) and report on the rest. TBD.
5. **PRR-A may show a small gap.** $\hat A+\hat E$ and $e(q)$ differ only where the backbone is miscalibrated; on ordinary splits of these five datasets the difference could be within noise (§7.3).
6. **PRR-B and retrieval depend on how sensitive $\hat E$ is to the context.** Both are constructions with no published reference numbers. The PRR-B contrast of 256 against 6,024 rows is one choice, and the result may depend on it. The resolution check (§10.4) is the first look at this sensitivity.
7. **No ground-truth test beyond 1-D.** The only "is the number right" evidence in Phase 0 is 1-D with Gaussian noise (T1, T2); what the observed cut gives up (latent confounding, proposal §4.3) is not measured in Phase 0. A reliability plot from Phase 1's validation bank costs nothing and could sit in the appendix.

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

The alternative was to compare against the **posterior-averaged** truth $\bar T=\mathbb E_{\tau\mid D}[\,\cdot\,]$, $T\in\{A,E(q_m)\}$, computed by an oracle: an exact posterior over tasks given each context.

- **What the oracle would buy.** A per-query target with no scatter (§4.5), allowing per-query comparisons and $R^2$.
- **What it costs.** A posterior for every (toy, $n$) on T1 and every (toy, cell) on T2: hours per posterior, its own Monte Carlo validation, and a noise prior forced onto a grid. On the real data of T3–T6 it is not available at all.
- **Why the realized truth is enough.** Tasks are drawn from a fixed prior and every estimate is a function of $(D,x_*)$, so the realized truth averages exactly to $\bar T$ given the context (§4.5). T2's cell means over many toys therefore give the same answer as the oracle would, at zero cost, and T3–T6 rank or average realized losses, which need no oracle.
- **What is given up.** Per-query exactness and $R^2$. On T1 nothing is given up: the task is fixed and known, so the truth is exact per query.
