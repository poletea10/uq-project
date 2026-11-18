# iLTMv2 transfer guide: first Phase 1 results and the redesign that follows

**Project:** Score-Based Uncertainty Decomposition for Tabular Foundation Models (iLTMv2) · **Written:** 14 November 2026

This guide closes the first round of Phase 1 experiments and defines the second one. Now there's two repositories:

- the **archived repository**: the code, protocols and runs that produced the results of §1, frozen as they are. They sit in `/private/groups/ioannidislab/UQ_project_v1`;
- the **working repository**: a duplicate of it, that we must change as described in §2–§5. Bare in mind that ALL outputs from experiment0_baseline (logs included) have been erased, so we can start from scratch all the evaluator checks and figures. When it comes to the outputs of experiment1_baseline/, I archived them in `/private/groups/ioannidislab/UQ_decoupling_exp/`, so checkpoints can be reused or old logs checked. The new `outputs/` folder in `experiment1_uq_heads/` now symlinks to `/private/groups/ioannidislab/UQ_decoupling_exp_new/`.

The four project documents (`iLTMv2_research_proposal.md`, `phase0_protocol.md`, `phase1_design.md`, `phase1_protocol.md`) are rewritten to describe only the new design, with no before/after notes. This guide is the only place where old and new are compared.

| Section | Content |
|---|---|
| §1 | Analysis of the first results |
| §2 | New Phase 0 test set |
| §3 | New Phase 1 training design |
| §4 | Job set-up for the new sweep |
| §5 | Step-by-step workplan |
| §6 | Legacy checkpoints (text for the end of the new Phase 1 document) |
| §7 | Design liberties taken in this guide, and doubts about the new design |

**Naming.** In §1 the arms keep their old names (A1, A2, B). From §2 on, **A** is the old A1, **B** is unchanged, and A2 no longer exists.

---

## 1. Analysis of the first results

### 1.0 What was run

Two scalar heads on a frozen TabICLv2 regressor: $\hat A$ (target $\hat t_A$, the CRPS entropy of the task law from replicates) and $\hat E_{\text{dir}}$ (target $\hat t_E$, the forecast's mean CRPS minus $\hat t_A$).

| Arm | What the heads read | Runs |
|---|---|---|
| A1 | MLP trunk on $h_*$ | 3 seeds × head LR {1e-4, 3e-4, 1e-3} |
| A2 | Learned softmax mix of 12 residual taps, then the trunk | 3 seeds × head LR {1e-4, 3e-4, 1e-3} |
| B | Trainable copy of the last 2 blocks + LN | Seed 0, head LR × fork LR {3e-6, 1e-5, 3e-5} |

Training used 50,000 / 4,000 / 1,000 steps over three stages, 32 tasks per step, $p=0$, and one context per task. Validation used a fixed bank of 2,000 tasks (3.1M queries, 128 replicates per query).

**Metric.** xMSE is the head's squared error against the realized task with replicate noise removed. Skill is $1-\text{xMSE}/\text{xMSE}_{\text{const}}$, where the constant baseline predicts the bank-wide mean target. The **e(q) proxy** is the best predictor that sees only the backbone's own CRPS entropy $e(q_\theta)$: its skill is what the forecast's spread gives for free.

### 1.1 Headline numbers

Skill at the final checkpoint (step 55,000). A1 and A2 are 3-seed means at head LR 1e-3. B is the selected configuration (head LR 3e-4, fork LR 3e-5, seed 0).

**Aleatoric head $\hat A$**

| Slice | e(q) proxy | A1 | A2 | B |
|---|---|---|---|---|
| Pooled | 0.613 | 0.890 | 0.897 | 0.902 |
| i.i.d. | 0.674 | 0.909 | 0.917 | 0.922 |
| Hole | 0.540 | 0.872 | 0.880 | 0.886 |
| Extrapolation | 0.385 | 0.824 | 0.828 | 0.829 |
| Duplicate | 0.765 | 0.914 | 0.921 | 0.924 |
| n ≤ 512 | 0.009 | 0.718 | 0.730 | 0.753 |
| n 512–1K | 0.306 | 0.799 | 0.801 | 0.829 |
| n 1K–4K | 0.401 | 0.850 | 0.860 | 0.866 |
| n 4K–16K | 0.647 | 0.880 | 0.889 | 0.899 |
| n 16K–60K | 0.678 | 0.915 | 0.921 | 0.921 |

**Epistemic head $\hat E_{\text{dir}}$**

| Slice | e(q) proxy | A1 | A2 | B |
|---|---|---|---|---|
| Pooled | 0.119 | 0.419 | 0.422 | 0.436 |
| i.i.d. | 0.070 | 0.370 | 0.376 | 0.384 |
| Hole | 0.134 | 0.450 | 0.441 | 0.472 |
| Extrapolation | 0.186 | 0.467 | 0.473 | 0.487 |
| Duplicate | −0.159 | 0.501 | 0.528 | 0.513 |
| n ≤ 512 | 0.169 | 0.393 | 0.395 | 0.401 |
| n 512–1K | 0.156 | 0.377 | 0.382 | 0.395 |
| n 1K–4K | 0.145 | 0.403 | 0.408 | 0.407 |
| n 4K–16K | 0.119 | 0.380 | 0.384 | 0.397 |
| n 16K–60K | 0.083 | 0.469 | 0.470 | 0.493 |

n is the number of context rows. Pooled values are query-weighted, and over half of the bank's queries are in the 16K–60K bin.

What the tables say:

- **Both heads beat the e(q) proxy on every slice.** This is the claim of the project: the forecast's spread explains 0.61 of the aleatoric target but only 0.12 of the epistemic one, and the heads raise both. The margin is largest where it matters: $\hat A$ on extrapolation (0.82–0.83 against 0.39) and at small context (0.72–0.75 against 0.01).
- **The three arms are close.** A1 already reaches 0.890 / 0.419. On the paired seed 0, A2 lowers the pooled validation MSE by 5.0% ($\hat A$) and 0.2% ($\hat E$) against A1, and B lowers it by a further 5.2% and 2.5% against A2. So the split is largely readable from $h_*$ alone.
- **Where B helps.** Its gains concentrate in holes for $\hat E$ (+0.03 skill) and in contexts under 1K rows for $\hat A$ (+0.02 to +0.03).
- **Seeds are tight.** Pooled skill varies by 0.001–0.002 across the three seeds, so the small differences above are real, but small.
- **A2's layer weights** end at 0.27–0.33 on layer 12 and 0.14–0.16 on layer 11, and do not move after stage 1. A2 behaves like A1 with a small addition.

**Cost.** One A job trains six head pairs in about 53 GPU-hours (160 hours for all 18 pairs). One B job trains one configuration in about 59 hours, and B also adds two transformer blocks to every inference.

### 1.2 Training dynamics

- **Almost everything is learned in the first 5,000 steps.** By then $\hat A$ has made 97% and $\hat E$ 93% of its total drop below the constant baseline. From step 5,000 to 50,000, xMSE falls a further 15–16% for $\hat A$ and 3% for $\hat E$.
- **The stage-1 plateau after step 35,000 is the schedule.** The cosine learning rate is at 21% of its maximum at step 35,000 and under 10% at 40,000. The plateau therefore does not show that the heads had converged; it shows that they could no longer move.
- **Stage 2 helps, mostly at large context.** Pooled xMSE drops by 6.6–6.9% ($\hat A$) and 1.2–1.6% ($\hat E$) in A1/A2, and by 12% and 2.5% in B.
- **Stage 3 looks like poor value.** It costs 12.5 hours (A) or 18 hours (B) and lowers xMSE by 0.6–1.0% ($\hat A$) and 0.2% ($\hat E$) in A1/A2. This is not conclusive: stage 3 had 1,000 steps and one validation point.
- **Stages 2–3 slightly hurt small contexts.** On the mini-bank (64 tasks of 1,024 rows, the stage-1 shape), $\hat A$ xMSE rises by 5–8% over stages 2–3 in every arm. On the full bank, $\hat A$ skill at n ≤ 512 falls from 0.729 to 0.718 (A1) and from 0.737 to 0.730 (A2).
- **The best learning rates are the top of both grids.** Skill rises with head LR in A1 and A2 on both heads (A1 $\hat A$: 0.880, 0.885, 0.890). In B, at step 54,000 skill rises with fork LR at every head LR (for example $\hat E$ at head LR 3e-4: 0.425, 0.431, 0.434).
- **The fork stops moving after stage 1.** Its distance from the pretrained weights is 9.445 at step 50,000 and 9.455 at step 55,000 (selected configuration).
- **Bank and stream agree.** The stage-1 training xMSE (about 0.042 for $\hat E$, 0.005 for $\hat A$) matches the small-context validation bins, so the heads learned a function of the prior, not of specific tasks.

### 1.3 Where the remaining error sits

#### 1.3.1 The ceiling of the epistemic head

**The statement.** An $\hat E$ skill of 0.42–0.44 looks low next to 1. But 1 is not reachable by any estimator, and the reachable maximum is probably not far above what the heads achieve.

**Why 1 is not reachable: scatter.** Fix one context $D$ and one query $x_*$. Many tasks are consistent with that context. Each has its own true law $p_\tau(\cdot\mid x_*)$, so each gives the forecast a different realized divergence $d_S(q_\theta,p_\tau)$. The head outputs one number for $(x_*,D)$. The best number is the average over those tasks,

$$
\bar E(x_*,D)=\mathbb E_{\tau\mid D}\big[d_S(q_\theta,p_\tau)\big],
$$

which is also what the squared-error loss converges to. The task that was actually drawn still differs from this average. That difference is the **scatter**. It is not a defect of the head: knowing which way the realized value deviates would mean knowing more about the task than the context contains, and then the forecast itself could be improved.

**A concrete case.** Take a deterministic task and a forecast $q=\mathcal N(0,1)$ that is calibrated, so the true value behaves like a draw from $\mathcal N(0,1)$.

| Where the truth lands | Realized divergence |
|---|---|
| At the centre (0) | 0.23 |
| One s.d. away (1) | 0.60 |
| Two s.d. away (2) | 1.45 |
| Average over all landings | 0.56 |

A perfect head says 0.56 every time. The realized values have a standard deviation of 0.40 around it.

**The coefficient of variation.** $\mathrm{CV}$ is the scatter's standard deviation divided by its mean: here $0.40/0.56=0.72$. It says how large the unavoidable scatter is relative to the quantity being predicted.

**From CV to the best possible error.** Write the target as $t=\bar E\,\xi$, where $\xi$ is "realized over expected", with mean 1 and standard deviation CV.

- The perfect head predicts $\bar E$, so its error is $\bar E(\xi-1)$ and its mean squared error is $\mathrm{CV}^2\,\mathbb E[\bar E^2]$.
- The target's second moment is $\mathbb E[t^2]=(1+\mathrm{CV}^2)\,\mathbb E[\bar E^2]$.
- Dividing the two:

$$
\text{xMSE}_{\text{best}}=\frac{\mathrm{CV}^2}{1+\mathrm{CV}^2}\;\mathbb E[t^2].
$$

This assumes one CV for the whole slice. In reality CV varies by query, and the slice value is an average dominated by high-$E$ queries.

**Implied CV of our heads.** Running the formula backwards answers: how large would the scatter have to be for our head to be perfect? Pooled, $\mathbb E[t^2]=0.0364+0.0463^2=0.0386$ and A2's xMSE is 0.0211. The ratio is 0.546, which gives an implied CV of 1.10.

| Slice | A1 | A2 | B |
|---|---|---|---|
| Pooled | 1.10 | 1.10 | 1.07 |
| i.i.d. | 1.23 | 1.22 | 1.20 |
| Hole | 1.03 | 1.05 | 0.99 |
| Extrapolation | 0.99 | 0.98 | 0.96 |
| Duplicate | 1.18 | 1.11 | 1.15 |

The implied CV is an upper bound on the true one: if the true scatter is smaller, the gap is error the head could still remove.

**What the true CV could be.** For an idealised, exactly Bayesian Gaussian forecaster (computed numerically):

| Regime | CV |
|---|---|
| All uncertainty is epistemic (the case above) | 0.72 |
| Epistemic variance 3× the noise variance | 1.07 |
| Epistemic and noise variance equal | 1.21 |
| Noise dominates | 1.41 |

**How near the ceiling we are.** Pooled, B's implied CV of 1.07 caps the true value:

| True CV | Ceiling of pooled $\hat E$ skill |
|---|---|
| 1.07 | 0.44 (B is at the ceiling) |
| 1.00 | 0.47 |
| 0.93 | 0.51 |
| 0.72 | 0.64 (idealised extreme) |

So the ceiling lies between 0.44 and 0.64, and never at 1. Three observations point to its lower half:

- the implied values fall inside the theoretical range;
- their ordering matches theory: i.i.d. queries are the most noise-dominated and have the highest value, extrapolation queries are the most epistemic and have the lowest;
- giving the heads more access to the backbone (A1 → A2 → B) barely moved the skill.

This is a consistency argument under idealised assumptions, not a measurement of the ceiling.

**The same point in DEUP.** Lahlou et al. (2023) meet the same limit when they correlate predicted uncertainty with model residuals on drug-combination data (Fig. 6c and App. D). They estimate a "best-case upper bound" by assuming each predictive $\mathcal N(\hat\mu,\hat\sigma)$ is exactly the target's law, sampling residuals from $\mathcal N(0,\hat\sigma)$, and correlating them with $\hat\sigma$. The bound is far from 1 (0.63 ± 0.05 for DEUP, whose observed correlation is 0.47 ± 0.03), and they report the ratio of observed to achievable (0.75 ± 0.07) as the comparison metric. `phase0_protocol.md` used the same fact when it excluded $R^2$ and rank correlation against realized truth as grades.

**A check that would tighten the bracket.** On the bank's deterministic tasks, replace the idealised Gaussian by the real forecast: for each query, compute the mean and standard deviation of $\mathrm{CRPS}(q_\theta,y)$ for $y$ drawn from $q_\theta$ itself, using the stored quantiles. That gives the true CV on those queries, under the assumption that the backbone is calibrated there.

#### 1.3.2 The low end is poorly resolved

**What the ratio is.** For each query, $e(q_\theta)$ is the forecast's total spread and $A$ is the part of it that is noise. So $A/e(q_\theta)$ is the share of the spread that is truly noise. Its bank average is 0.41. It is low because about 60% of tasks are deterministic, where the share is 0.

**What the heads say.** The same average for $\hat A/e(q_\theta)$ is 1.11 (A1), 0.88 (A2) and 0.87 (B). On a typical query the head claims that nearly all of the spread is noise.

**Why this does not contradict the 1% bias.** The plain mean of $\hat A$ is within 1–2% of the mean target on the pooled bank. That mean is dominated by the few queries with large values, and those are right. The mean of the *ratio* gives every query one vote whatever its scale, so the many small-value queries dominate it. An illustrative query: $e(q)=0.002$, $A=0$, $\hat A=0.002$. The ratio is 1 where it should be 0, but the squared error is $4\cdot10^{-6}$ and the MSE never sees it.

So the ratio says that in near-deterministic tasks $\hat A$ does not go to zero; it stays at about the size of $e(q_\theta)$.

**Why it happens.** With softplus, a small output responds to its input in proportion to the output itself, so the squared-error gradient is about $2(\hat A-t)\,\hat A$.

- Small query ($\hat A=0.002$, $t=0$): gradient about $8\cdot10^{-6}$.
- Large query ($\hat A=0.5$, $t=0.3$): gradient about $0.16$.

With $p=0$ the optimiser almost never corrects small values.

**Evidence that this is optimisation, not missing information:**

- the ratio falls with learning rate (A1: 2.86, 2.00, 1.11 for LR 1e-4, 3e-4, 1e-3);
- it falls over training (A2: 2.02 at step 5,000, 0.85 at step 50,000);
- $\hat E$ shows the same pattern: it over-predicts duplicates, the lowest-divergence queries, by 20–33%;
- on toy rung 0 at $n=1024$, $\hat E$ sits at about $5\cdot10^{-4}$ where the truth is about $2\cdot10^{-4}$.

**Why it matters.** The epistemic fraction $\lambda=\hat E/(\hat A+\hat E)$ is a ratio, the toy figures draw $E$ on a log axis, and toy rung 0 asks for $\hat A\approx0$. All three look at the scale the unweighted MSE ignores.

**Caveat.** A mean of ratios can be driven by a minority of queries with a tiny $e(q_\theta)$. The logs do not show whether the effect is typical or a tail. The median ratio, or the mean $\hat A$ on deterministic tasks only, would settle it.

#### 1.3.3 A mild compression of $\hat E$ across context size

$\hat E$ is 5–10% too low on average for contexts under 1K rows and 3–7% too high at 16K–60K. The error is small.

### 1.4 Toy set

Figures for the selected A2 heads; values are read off the figures and are approximate.

- **$\hat E$ follows the realized excess over three orders of magnitude** at $n\ge64$ on all rungs, and closely at $n=1024$.
- **The attribution test of rung 2 passes at $n=1024$.** $\hat A$ rises in the noise strip (about 0.95 against 0.84 true), and $\hat E$ is higher in the sparse strip (about 0.3) than in the noise strip (about 0.06).
- **$\hat A$ leaks in the sparse strip at $n=1024$:** about 0.3 on rung 1 and 0.65 on rung 2, against a truth of 0.14. The heteroscedastic GP reports 1.65 there on rung 2. So iLTMv2 is far better, but this is its most visible flaw. Part of the leak may be the correct answer of the prior: with few points, noise and signal are hard to tell apart.
- **$\hat E$ over-predicts in the noise strip** at $n=1024$ on rung 2 (about 0.06 against 0.01).
- **$n=16$ and $n=64$ are poor.** $\hat A$ is near zero at $n=16$, and $\hat E$ misses the extrapolation beyond $x=4$. The smallest context the heads ever trained on is about 307 rows (30% of 1,024), so three of the four toy columns were outside the training range.
- **Rung 0, row 2 is unreadable:** the shared y-limit reaches about 140 because of the GP, so $\hat A\approx0$ cannot be seen.

### 1.5 Conclusions

1. **The results are encouraging, and they flatter in four ways.** $\hat A$ clears the protocol's 0.7 rule of thumb and $\hat E$ sits inside its 0.3–0.6 band. But validation is in-distribution; over half of the queries are in the easiest (largest-context) bin; unweighted MSE is dominated by a heavy tail ($\hat t_E$'s standard deviation is four times its mean); and the duplicate skill is inflated by the bank-wide baseline (0.40–0.43 against a duplicate-only baseline).
2. **$\hat E$ is probably near what any head can reach on this bank** (§1.3.1). Architecture changes have little pooled skill left to buy.
3. **The arms differ little.** A2's gain over A1 is small and is mostly a re-weighting towards the last layers. B's gain is real but modest, costs a job per configuration and extra inference, and its best fork LR was the top of the grid, so B was under-tuned.
4. **The visible weaknesses are elsewhere:** small contexts (never trained below about 307 rows), the low-uncertainty end (§1.3.2), and regions that are thin but not empty (the $\hat A$ leak of §1.4; training only ever built fully emptied holes).
5. **The schedule can be shortened in stage 1.** Whether stages 2–3 deserve more steps is open: each had a single validation point.

---

## 2. New Phase 0 test set

Phase 0 becomes six tests. The cut-graph test and the transfer test leave Phase 0 (§2.5).

| # | Test | Question | Ground truth | Models | Output |
|---|---|---|---|---|---|
| T1 | Toy set | Where does each model put $\hat A$ and $\hat E$ on one curated function? | Closed form | Heteroscedastic GP, Dec-ICL, iLTMv2 | Figures |
| T2 | Error heatmap | Are $\hat A$ and $\hat E$ right across local data density and local noise? | Closed form | Heteroscedastic GP, Dec-ICL, iLTMv2 | One heatmap figure |
| T3 | PRR-A (total error) | Do $\hat A+\hat E$ rank the forecast's loss better than its own spread? | Realized CRPS | iLTMv2 (Dec-ICL if available) | Curves + table |
| T4 | PRR-B (reducible error) | Does $\hat E$ say which predictions would improve with more data? | Realized CRPS gain | iLTMv2 (Dec-ICL if available) | Curves + table |
| T5 | Active learning | Does the epistemic estimate select useful data? | Test-set performance | Random, TabICLv2, iLTMv2, Tuned-ICL, Dec-ICL | Table + curves |
| T6 | Context retrieval | Does $\hat E$-guided retrieval beat k-NN for a given query? | Realized CRPS | iLTMv2 | Curves + table |

T1 and T2 are the visual results. T3–T6 are the thorough evaluation on real data. T5 is unchanged from the current protocol.

### 2.1 Toy set: context sizes

- `n_grid` becomes **64, 256, 1024, 4096**. $n=16$ is dropped and never trained on.
- Same curated function, same rungs, same queries.
- The context is drawn once at $n=4096$ and nested. Assert that its first 1,024 points equal the current context, so the existing columns do not change.
- The sparse strip then holds about 2, 8, 32 and 128 points.
- The paper figure stays toy rung 2 at $n=1024$, one column per model. The full 3×4 figures go to the appendix.
- Row 2 gets per-model y-limits (or a log axis) on rung 0, so that $\hat A\approx0$ is visible.

`n_grid` was a constant fixed before the first run. The change is made because of the training range, which does not depend on which model wins; it is recorded here as a deviation from that rule.

### 2.2 Toy set: figure design and why the comparison is fair

**Design.** The current layout is kept:

| Row | Content |
|---|---|
| 1 | Context, true mean and 90% band, forecast median and 90% band |
| 2 | True $A$, $\hat A$, and total $e(q_m)$ as a dashed line |
| 3 | Realized excess $E(q_m)$ and the model's predicted excess $\hat E$, log axis |

- No shading between $\hat A$ and $e(q_m)$.
- Row 3 carries a small legend naming each model's estimator: **$e(q_m)-\hat A$** for the heteroscedastic GP and Dec-ICL, **the $\hat E_{\text{dir}}$ head** for iLTMv2.
- $e(q_\theta)-\hat A$ is not shown for iLTMv2.

**Why the comparison is fair** (text for `phase0_protocol.md`).

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
- Dec-ICL's native epistemic signal is the variance of its latent head, with $v_{\text{tot}}=v_{\text{epi}}+\hat\sigma^2$ (Bergna et al., 2026, App. C.1). A variance cannot be drawn against a CRPS truth, so its members $\mathcal N(c_j,\hat\sigma^2)$ are projected as above. Its native $v_{\text{epi}}$ is used where only ranking matters (T4, T5).
- iLTMv2 is the only model trained on exactly this target. That is the method, not a bias of the test. In the other direction, the toy function is closer to the GP's and Dec-ICL's function classes than to iLTMv2's prior.

### 2.3 T2: error heatmap

**What it shows.** T1 shows two situations on one function: a noisy region with plenty of data, and a sparse region with little noise. T2 fills in the whole grid. For every combination of local data density and local noise, it shows whether each model's $\hat A$ and $\hat E$ are too high, right, or too low. The two strips of T1 are two cells of this grid.

**Construction: a probe strip.** Each toy function has a fixed background and one strip. The cell sets the strip's local density and local noise, and only queries inside the strip are scored. The surroundings of the strip are the same for every cell of a toy, so a cell's value is not confounded by what lies around it.

| Element | Setting |
|---|---|
| Function | The toy generator's base family on $[-5,5]$ (28 bumps, width 0.3, unit prior s.d.), fresh draw per toy, no gap part |
| Strip | Width 2, position drawn per toy inside $[-4,4]$, edge width 0.1 |
| Background | Per toy, drawn once and kept for all its cells: context density $d_{\text{bg}}$ log-uniform in [64, 1,024] points per unit of $x$; noise $\sigma_{\text{bg}}$ log-uniform in [0.05, 0.5] |
| x-axis (cell) | Context points inside the strip: 0, 2, 8, 32, 128, and "background" (the strip at density $d_{\text{bg}}$) |
| y-axis (cell) | Noise s.d. inside the strip: 0, 0.1, 0.25, 0.6, 1.5 |
| Queries | 50 evenly spaced points in the strip interior (0.25 away from each edge) |
| Sharing | Within a toy, all 30 cells share the function, the background context and the standard-normal noise draws |
| Toys | 20 (the first 10 for the heteroscedastic GP, which needs one NUTS fit per toy and cell) |

**Truth.** Closed form per query, as in T1: $A=\sigma(x_*)/\sqrt\pi$ and $E(q_m)=\int(F_m-\Phi_\tau)^2dt$, the model's own excess.

**Cell value.** For each model, quantity and cell:

$$
\text{value}=\log_2\frac{\text{mean estimate}}{\text{mean truth}},
$$

with means over all toys and queries of the cell, clipped at ±3.

- The mean is taken first, the ratio second. A per-query ratio is not used for $\hat E$ because of the scatter of §2.4.
- In the $\sigma=0$ row the true $A$ is 0, so a fixed floor of 0.01 is added to both means for $\hat A$.

**Figure.** One row per model, two heatmaps per row ($\hat A$ and $\hat E$). Colour is signed: **red** = the estimate is too high, **white** = correct, **blue** = too low. Each model uses the estimator of §2.2's legend.

**How to read it.** A model that reads sparse data as noise is red in the left columns of its $\hat A$ panel and blue in the same cells of its $\hat E$ panel. A model that decouples correctly is white in both.

The grid, the distributions and the floor are fixed before the first run and not tuned afterwards.

### 2.4 The epistemic scatter and the performance ceiling

Text for `phase0_protocol.md`, placed after the T1 and T2 descriptions.

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

**The same point in DEUP.** Lahlou et al. (2023, Fig. 6c and App. D) correlate predicted uncertainty with model residuals and note that the achievable correlation is far below 1. They estimate a best-case upper bound by sampling residuals from each model's own predictive distribution (0.63 ± 0.05 for DEUP against an observed 0.47 ± 0.03) and report the ratio of observed to achievable as their comparison metric.

*Optional reference line for T3.* The same idea applied to PRR: sample $y$ from the model's own forecast and compute the PRR of its score. It shows what PRR the score would get if the forecast were exactly right.

### 2.5 Tests removed from Phase 0

- **Cut-graph test.** Not a Phase 0 test any more. The cut-graph validation bank belongs to Phase 1 and is used only to track and compare training runs (§3.4).
- **Transfer test.** Removed from the design. Its description stays in the archived repository.

The new `phase0_protocol.md` does not mention either. With them go the truth-on-estimate grading section, the replicate estimators for Phase 0, and the `cutgraph.py`, `transfer.py` and `grading.py` modules.

### 2.6 T3: PRR-A, ranking total error

**What it shows.** Whether the two heads together predict how large the forecast's loss will be, better than the forecast's own spread does.

- $e(q_\theta)$ is what the forecast expects to pay if it is right about itself.
- $\hat A+\hat E_{\text{dir}}$ is the learned estimate of what it actually pays (the expected score of $q_\theta$; proposal §2.2).
- Both score the same forecast and the same errors, so the comparison is clean.

The observed loss on real data includes noise, so total uncertainty is the right ranker of it. That is why the sum is used here.

**Construction.**

1. **Data.** The five active-learning datasets and their stored splits (California Housing, Kin8nm, NYC Taxi, Airlines, YearPredictionMSD; per seed a 1,024-row warm-start set, a 5,000-row pool and a 3,000-row test set). No carved splits.
2. **Forecast.** Context = warm-start set + pool (6,024 rows); queries = the test set. One raw forward of iLTMv2, the route the heads were trained on.
3. **Loss.** $\ell_i=\mathrm{CRPS}(q_i,y_i)$ per test point, with the CDF convention used everywhere else.
4. **Scores.** $s_i^{\text{tot}}=e(q_i)$ and $s_i^{\text{dec}}=\hat A_i+\hat E_i$.
5. **Rejection curve.** Sort the test points by a score, highest first. After rejecting the first $k$, the curve value is the summed loss of the retained points divided by the total count $N$. Rejected points contribute 0: they are handed to an oracle that knows the outcome, and a point mass at the true $y$ has a CRPS of exactly 0.
6. **Reference curves.** Random rejection is the straight line from the mean loss to 0. The oracle ordering sorts by $\ell_i$ itself.
7. **PRR.**

$$
\mathrm{PRR}=\frac{AU_{\text{random}}-AU_{\text{score}}}{AU_{\text{random}}-AU_{\text{oracle}}},
$$

with $AU$ the area under each curve. 1 is the oracle ordering, 0 is random, negative is worse than random.

**Outputs.**

- **Figure:** one panel per dataset with four curves (oracle, $e(q)$, $\hat A+\hat E$, random), averaged over seeds.
- **Table:** per dataset, the PRR of $e(q)$ and of $\hat A+\hat E$ (mean ± s.d. over seeds) and their paired difference.

**If the Dec-ICL checkpoints arrive.** Dec-ICL's decomposition sums to its total by construction, so on its own forecast it has one score (its total), shown next to Tuned-ICL's total as context.

**Expectation.** The two scores differ only where the backbone is wrong about itself, so the gap on ordinary splits may be small. This is stated before running.

### 2.7 T4: PRR-B, ranking reducible error

**What it shows.** Whether $\hat E$ tells which predictions would improve with more data. This is what "epistemic" means operationally, and it gives an epistemic ground truth on real data without replicates.

**The idea.** Predict each test point twice, with a small context and with the full one, and rank the improvement:

$$
\Delta_i=\mathrm{CRPS}\big(q_i^{\text{small}},y_i\big)-\mathrm{CRPS}\big(q_i^{\text{full}},y_i\big).
$$

In expectation, $\Delta_i$ is the excess with the small context minus the excess with the full one. The aleatoric term is the same in both and cancels. So $\Delta$ measures reducible error only.

**Construction.**

1. **Data.** Same datasets, splits and seeds as T3.
2. **Contexts.** Small = the first 256 rows of the warm-start set. Full = warm-start set + pool (6,024 rows).
3. **Scores.** From the small-context forward only: $\hat E_i$ against $e(q_i)$, the total-uncertainty baseline. Nothing from the full-context forward is used for ranking.
4. **Curve and PRR.** As in T3 with $\ell_i$ replaced by $\Delta_i$. The oracle ordering sorts by $\Delta_i$.

**Outputs.** As in T3: one panel per dataset (oracle, $e(q)$, $\hat E$, random) and a table with the PRR of both scores per dataset.

**Reading notes.**

- $\Delta_i$ can be negative for single points, so a curve can dip below 0 before returning to it. The PRR formula is unchanged.
- $\Delta_i$ is a difference of two noisy losses, so absolute PRR values will be low (§2.4). The comparison between the two scores is what counts.

**If the Dec-ICL checkpoints arrive.** Same test on its own forecasts: $v_{\text{epi}}$ against $v_{\text{tot}}$. The comparison across models is then how much each decomposition gains over its own total, as in T5.

### 2.8 T6: context retrieval

**What it shows.** For one query and a fixed starting context: given an extra labelled pool and a budget of $N$ rows, which selection rule gives the best forecast at that query? It compares $\hat E$-guided selection with k-NN and with random selection. It is a prelude to Phase 2, run on a Phase 1 checkpoint with no extra training.

**Set-up.**

| Element | Setting |
|---|---|
| Data | The five datasets and splits of T3 (start with one dataset, 5 seeds) |
| Starting context $S_0$ | The first 256 rows of the warm-start set |
| Pool | The 5,000-row pool, labels known |
| Queries | 200 points drawn from the test set |
| Budgets $N$ | 16, 64, 256 |

**Methods.**

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

**Metric.** Realized $\mathrm{CRPS}(q,y)$ at the query, averaged over queries, per budget, paired across methods. Never $\hat E$ itself.

**Reported alongside.**

- The share of selected rows that came from the random half of the shortlist. Near zero means $\hat E$ behaves like k-NN, which is a finding in itself.
- Exhaustive scoring of the whole pool for 20 queries, to show what the shortlist loses.

**Check before building.** Adding one row to 256 may move $\hat E$ by less than the head can resolve. For a few queries, look at the spread of $\hat E(x_*,S\cup\{c\})$ over candidates. If it is flat, score blocks of 8 rows instead of single rows.

**Cautions, stated with the results.**

- **Selection sees the labels.** The heads' targets are valid because every training split rule looks at $X$ only. Greedy selection uses each candidate's label through the forward pass, so $\hat E$ can be biased low on the chosen context. This is why the metric is the realized loss.
- **The whole-pool line may win.** TabICLv2 takes tens of thousands of context rows, so a small budget is artificial at these dataset sizes.
- **Use the new checkpoint.** Training on subsampled contexts of the same task (§3.9) is what teaches the head how $\hat E$ changes with the context.

---

## 3. New Phase 1 training design

Each item states the decision and the reason. The new `phase1_design.md` and `phase1_protocol.md` describe the result as the design, without reference to the old one.

| Item | Old | New |
|---|---|---|
| Steps per stage | 50,000 / 4,000 / 1,000 | 30,000 / 8,000 / 1,200 |
| Validations | Every 5,000 steps in stage 1, stage ends | Plus the middle of stages 2 and 3 |
| Arms | A1, A2, B | A, B |
| Seeds | 0, 1, 2 (A); 0 (B) | 0 (and 1+2 for best configs) |
| Head LR | {1e-4, 3e-4, 1e-3} | {3e-4, 1e-3, 3e-3} |
| Fork LR (B) | {3e-6, 1e-5, 3e-5} | {1e-5, 3e-5, 1e-4} |
| Loss exponent $p$ | 0 | {0, 1} |
| Carved regions | Fully emptied | Half emptied, half thinned |
| Prior variants | Carved | Carved and i.i.d.-only (latter only used for ablation in best config) |
| Contexts per task (stage 1) | One | Full + one subsampled (down to 64 rows) |
| Validation bank | Small tasks of 1,024 rows, 30–90% context; unweighted metrics | Small tasks of 256–1,024 rows, 80% context; thinned regions; metrics for $p=0$ and $p=1$ |

### 3.1 Steps per stage

**30,000 / 8,000 / 1,200** (40,000 global steps). The learning-rate schedule keeps its form: cosine restarted per stage, 1% warm-up, maximum LR divided by 1, 8 and 40.

Sentence for the new Phase 1 document: *"Stage 1 has 30,000 steps because in earlier runs more steps did not give a significant improvement."*

### 3.2 Validation cadence

| Stage | Global steps of the full-bank validations |
|---|---|
| 1 | 5,000, 10,000, 15,000, 20,000, 25,000, 30,000 |
| 2 | 34,000, 38,000 |
| 3 | 38,600, 39,200 |

Ten validations per run, as before. The mini-bank stays at every 250 steps.

### 3.3 Arms

Only **A** (MLP trunks on $h_*$; the old A1) and **B** (trainable fork of the last 2 blocks + LN). The multi-layer tap is removed from the design and from the code, and the new documents do not mention it.

A practical consequence: with no residual taps, the bank's $h_*$ can be cached once per A job, so an A validation costs only the head MLPs.

### 3.4 Validation bank

A new bank of 2,000 tasks replaces the old one. It follows the training pipeline, including thinned regions (§3.7).

| Bin | Rows $N$ | Context fraction | Share | Tasks |
|---|---|---|---|---|
| Small | Uniform in 256–1,024 | 80% | 50% | 1,000 |
| Medium | 1,024–10,240 (log-uniform) | ≈80% | 30% | 600 |
| Large | 10,240–60,000 (log-uniform) | ≈80% | 20% | 400 |

- Query mixture, duplicates and thinning follow the training distribution of the carved prior.
- $R_{\text{val}}=128$; a new seed, disjoint from every test.
- Every run, carved or not, is validated on this same bank.

**Query-type slices:** `iid`, `hole`, `hole_thinned`, `extrapolation`, `extrapolation_thinned`, `duplicate`. A region with zero returned rows is `hole` / `extrapolation`; a region with returned rows is the `_thinned` type.

**Context-size slices:** `n_le256`, `n_256-512`, `n_512-1K`, `n_1K-4K`, `n_4K-16K`, `n_16K-60K`.

**Metrics.** Every run reports both weightings, whatever it was trained with:

- unweighted ($p=0$): MSE, xMSE, skill, as before;
- weighted ($p=1$): the same three with each query's term multiplied by $w=(e(q_\theta)+c)^{-1}$. The weighted skill uses the best constant under that weighting (the weighted mean target) as its baseline.

The reference lines (noise floor, constant baseline, e(q) proxy) are recomputed on the new bank for both weightings. The constant $c=0.1$ (§3.6) is stored in its manifest.

### 3.5 Learning-rate grids

| | Stage-1 values |
|---|---|
| Head LR (A and B) | 3e-4, 1e-3, 3e-3 |
| Fork LR (B) | 1e-5, 3e-5, 1e-4 |

The best value was the top of both old grids (clearly for the fork, by a small margin for the heads), so both grids move up to find where the gain stops.

### 3.6 Loss exponent

$$
\mathcal L=\overline{\,w\,(\hat A-\hat t_A)^2\,}+\overline{\,w\,(\hat E_{\text{dir}}-\hat t_E)^2\,},\qquad w=\big(\mathrm{sg}[e(q_\theta)]+c\big)^{-p},\qquad p\in\{0,1\}.
$$

- **$p=0$:** every query counts equally in absolute units, so large-value queries dominate.
- **$p=1$:** low-spread queries are up-weighted in proportion to $1/e(q_\theta)$.
- **Why $p=1$ should help.** §1.3.2 showed that with softplus the gradient on small values is weaker by one factor of the output's scale. On the low end the output is about $e(q_\theta)$, so a weight of $1/e(q_\theta)$ gives that factor back.
- **The floor $c$.** Without it, a query with $e(q_\theta)$ near 0 would get a near-infinite weight; the largest weight is $1/c$. **$c$ is fixed at 0.1** (z-normalized $y$ units), so queries with $e(q_\theta)$ well below 0.1 share roughly the same weight, at most about 6 times that of a typical noisy query, and the rest are down-weighted smoothly: $w$ runs from 10 down to about 1.7. The bank median was the first candidate, and it was rejected once the bank was built: $e(q_\theta)$ is bimodal (q25 $=5\cdot10^{-4}$, median $0.013$, q75 $0.25$), the median falls in the gap between the deterministic tasks (62% of the bank) and the noisy ones, and 88% of the queries below it come from deterministic tasks. With $c=0.013$ the largest weight would be 79, about 35 times a typical noisy query's, so the deterministic tasks would dominate the $p=1$ loss.
- **The weight is safe.** It depends on $(x_*,D)$ only, so the minimiser stays the conditional mean of the target.

Both values are trained. The validation metrics cannot choose between them, because each weighting favours the run trained with it. The choice is made with T2–T6 (§3.10).

### 3.7 Thinned regions

Text for `phase1_design.md`, as part of the task construction.

**Why.** A context is rarely either full or empty around a query. Regions with a few rows are where the question "is this spread noise or lack of data?" is hardest, and they are what context retrieval and active learning operate on. So the carved regions cover the whole range from empty to thin.

**Rule.** For each carved region (the extrapolation tail and the kNN hole, drawn independently):

- with probability 0.5 the region is **empty**: no row of it enters the context;
- otherwise it is **thinned**: a share $\rho$, log-uniform in [0.05, 0.2], of its non-query rows is returned to the context.

Log-uniform because the effect of density is roughly multiplicative: halving it matters about as much at 20% as at 5%.

**Sizes.** With tail size $T$, hole size $k$ and query budgets $m_{\text{ext}}$, $m_{\text{hole}}$:

$$
r_{\text{ext}}=\operatorname{round}\big(\rho_{\text{ext}}(T-m_{\text{ext}})\big),\qquad
r_{\text{hole}}=\operatorname{round}\big(\rho_{\text{hole}}(k-m_{\text{hole}})\big),
$$

$$
P=N+(T-m_{\text{ext}}-r_{\text{ext}})+(k-m_{\text{hole}}-r_{\text{hole}}).
$$

- Returned rows are chosen uniformly among the region's non-query rows and join the context.
- The i.i.d. remainder supplies $n_{\text{ctx}}-r_{\text{ext}}-r_{\text{hole}}$ context rows, so the context is exactly $n_{\text{ctx}}$ and the queries exactly $m$.
- All sizes are drawn before generation, so $P$ is known in advance.
- $r\le0.04N$ per region, far below the smallest context.
- A thinned region returns at least one row ($r=\max(1,\operatorname{round}(\cdot))$), so each region is empty or thinned with probability ½ each, independently of the other: a quarter of the tasks have both regions empty, a quarter only the tail thinned, a quarter only the hole thinned, a quarter both thinned. Only a region with no non-query row ($T=m_{\text{ext}}$ or $k=m_{\text{hole}}$) is always empty.

**Still an X-only rule.** Which rows are returned depends on $X$ and on random numbers independent of $y$, so the targets keep their meaning.

**Logging.** Query types as in §3.4. Each region's realized $\rho$ is stored per task.

### 3.8 Carved and non-carved priors

The best configuration is trained twice:

- **carved:** the full pipeline (extrapolation, hole, thinning, duplicates). Trained during the sweep;
- **i.i.d.:** the same tasks with TabICLv2's plain random split only ($P=N$; no tail, hole, thinning or duplicates). Trained only for the best configuration.

The carved prior is the design. The i.i.d. runs exist for the ablation of §3.10. Both use context subsampling (§3.9), so the ablation isolates the carving.

### 3.9 Subsampled contexts in stage 1

**Why.** The smallest context in stage 1 is about 307 rows, while the toy set, T4 and T6 use 64–256. Below 307 the heads have had no supervision.

**Rule.** In stage 1, every task is used twice in its step:

1. **Full pass:** the task as built (context fraction 30–90% of 1,024 rows).
2. **Subsampled pass:** the same task with its context reduced to $n'$ rows, $n'$ log-uniform in $[64,n_{\text{ctx}}]$. The rows are dropped uniformly at random. Queries and replicates are unchanged.

Details:

- One carving per task, as before.
- $\hat t_A$ is identical in both passes. $\hat t_E$ is recomputed, because the forecast changes.
- One $n'$ per micro-batch group, so a group still has one `(seq_len, train_size)`.
- A duplicate query whose source row was dropped is relabelled to its underlying type.
- Uniform dropping keeps the relative density of thinned regions.
- A step is 32 tasks and 64 forwards. The loss averages over the query rows of both passes.
- Stages 2 and 3 are unchanged (one pass per task).

**What it also buys.** The same query is seen under two contexts of the same task, which shows the heads directly how the excess changes with the amount of context.

**Two things to know.**

- TabICLv2 itself was pretrained with at least about 307 context rows. Below that its forecast is off its pretraining range. $\hat E_{\text{dir}}$ is trained on the frozen forecast as it is, so it should learn the larger excess.
- The subsampled pass has a context fraction below 30%. In TabICL a query attends to context rows only, so its forecast should not depend on how many other queries there are. A unit test confirms it: same context, 1 query against $m$ queries, identical quantiles.

### 3.10 The sweep and the selection

**Sweep (seed 0 only).**

| Arm | Head LR | Fork LR | $p$ | Prior | Configurations |
|---|---|---|---|---|---|
| A | 3e-4, 1e-3, 3e-3 | — | 0, 1 | carved (i.i.d. for winning config) | 6 (+1 without carving) |
| B | 3e-4, 1e-3, 3e-3 | 1e-5, 3e-5, 1e-4 | 0, 1 | carved (i.i.d. for winning config) | 18 (+1 without carving) |

Every run logs weighted and unweighted MSE, xMSE and skill (§3.4).

Once winning configurations for each arm are fixed, a 3 seed sweep is done to add error bars to the ablation.

**Selection.**

1. **Learning rates, from validation.** Within each (arm, $p$, prior) cell, pick the LR by the pooled validation MSE under that cell's own weighting. In A per head; in B the configuration as a whole (sum of the two heads' MSE, each relative to A's at the same $p$).
2. **Final model, from the tests.** The four carved winners (A and B, $p=0$ and $p=1$) are run on T2, T3, T4 and T5, and the best is chosen. $p=1$ is expected to win, since it improves small-value estimates.
3. **Ablation (appendix).** The final model against three counterparts, each differing in one factor: the other $p$, the other arm, and the i.i.d. prior. All on the heatmap, the PRR tests and active learning. Learning rates are not ablated.

---

## 4. Job set-up

| Jobs | Count | Contents |
|---|---|---|
| `A_carved_seed0` | 1 | One frozen backbone stream; 6 independent head pairs (3 head LR × 2 $p$), each with its own optimizer and schedule |
| `B_h<H>_f<F>_p<P>_carved_seed0` | 18 | One fork per job (3 head LR × 3 fork LR × 2 $p$) |

Afterwards, various jobs will be created for each of the winning configs, to do the ablation study (including the 3 seeds).

- **Why A needs one jobs.** Nothing upstream of the heads trains, so all head pairs can share one backbone pass. $p$ only changes a head's loss, so both values ride on the same pass.
- **Why B needs 18.** The fork changes the forward pass, and $p$ changes the fork's gradients.
- **Stages.** One SLURM job per run, stages 1 → 2 → 3 in sequence, one wandb run, as now.
- **Head pair names in A jobs:** `A_lr<LR>_p<P>` (for example `A_lr1e-3_p1`).
- **Storage.** New runs go to `/private/groups/ioannidislab/UQ_decoupling_exp_new/`, so nothing collides with the archived runs (the old `A_seed0` would otherwise be overwritten).

**Rough cost,** from the old median step times and before counting the second stage-1 forward:

| | Stage 1 | Stage 2 | Stage 3 | Validation | Per job (with overhead) | Sweep |
|---|---|---|---|---|---|---|
| A | 13 h | 13 h | 13 h | under 1 h (cached $h_*$) | about 45–50 h | about 45-50 h |
| B | 12 h | 14 h | 19 h | 4 h | about 55-60 h | about 1,000 h |

The old B sweep cost about 530 GPU-hours, just bare in mind that this is way more intense than the previous sweep.

---

## 5. Workplan

Training code comes first, so the jobs can run while Phase 0 is coded. Just redo debugging code checks when code is actually changed, if not take it as working (you can check the old logs in the archived repo).

**Code policy.** The current code works and is recycled wherever it fits. Where keeping the old structure would make the result convoluted, the piece is rewritten. The working repository should read as if it had been written for the new design: no A2 code paths, no cut-graph or transfer test modules, no flags that only the old design used.

### Step 0 — Archive

Record the repository's last commit hash here and in the new `CLAUDE.md`. The actual repo duplication and archiving has already been done, as I told you.

**Done.** The archived repository `/private/groups/ioannidislab/UQ_project_v1` is at commit `c1ed503d0a269051724ee61485862b96472eee45` ("First experiments finished. Wandb metric extractor added"). The hash is recorded in the root `CLAUDE.md`, in `experiment1_uq_heads/CLAUDE.md` and in the legacy-checkpoint text of §6.

### Step 1 — Phase 1 code (blocking)

| File | Change |
|---|---|
| `prior/_split.py` | Thinning (returned rows); an i.i.d.-only mode; uniform context subsampling; six query-type labels; relabel duplicates whose source was dropped |
| `prior/_dataset.py` | Pool size with returned rows; stage step counts 30k/8k/1.2k; the subsampled pass in stage 1; the small-task shape of the bank |
| `prior/_bank.py`, `01_make_val_bank.py`, `01b_check_val_bank.py` | New bank; reference lines for both weightings; $c$ in the manifest |
| `model/_heads.py`, `model/_backbone.py` | Remove the layer mix and the 12 residual taps; keep the hook that feeds the fork |
| `train/_loss.py` | $p$ per head pair; $c$ read from the manifest |
| `train/_metrics.py`, `train/_validate.py` | Weighted and unweighted metrics; new slices |
| `train/_config.py` | Steps, grids, validation cadence in stages 2–3, `--prior {carved,iid}`, `--p` |
| `train/frozen.py` | Head pairs = LR × $p$; two passes per task in stage 1; cached $h_*$ for validation |
| `train/fork.py` | `--p`, `--prior` |
| `scripts/train_A.sh`, `train_B.sh` | New arguments and names |
| All `scripts/*.sh` | `#SBATCH --output` / `--error` to `/private/groups/ioannidislab/UQ_decoupling_exp_new/slurm_logs/` |
| `train/_config.py` (`ROOT`), `prior/_bank.py` (`BANK_DIR`), `scripts/04_smoke_A.sh` (`ROOT`), `scripts/00e_test_heads.py` (`SMOKE_BANK`), `scripts/05_export_runs.py` (default output) | Root moved to `/private/groups/ioannidislab/UQ_decoupling_exp_new/` |
| Usage lines in `01_make_val_bank`, `01b_check_val_bank`, `04b_check_smoke`, `05_export_runs` | Example paths updated to the new root |

**Output paths.** After the duplication, every experiment1 script still points at the archived folder `/private/groups/ioannidislab/UQ_decoupling_exp/`. Jobs launched as they are would write into the v1 archive, and `A_seed0`-style run names would overwrite archived runs. All writes must go to `/private/groups/ioannidislab/UQ_decoupling_exp_new/`, which `experiment1_uq_heads/outputs` symlinks to. Create its `slurm_logs/` subfolder before the first job: SLURM does not create the log directory, and a job without one fails with no log. There is one exception. `inference.py` (`RUNS`, `BEST`) loads the v1 selected checkpoints, and those stay in the archived folder (§6). Keep that path for the legacy checkpoints, and point the v2 checkpoints at the new root.

### Step 2 — Gates before launch

1. **Split tests (`00c`):** exact counts with thinning; the split never looks at $y$; subsampling keeps queries and replicates; duplicate relabelling.
2. **Split figures (`03`):** a thinned hole and a thinned tail look right.
3. **Query-independence test:** same context, 1 query against $m$ queries, identical quantiles.
4. **Head tests (`00e`):** the 6 pairs of an A job are independent; with $p=0$ the weight is exactly 1.
5. **Throughput (`02`):** stage 1 with two forwards per task.
6. **Bank:** build `val_bank_v2`; check the target ordering (duplicate < iid < thinned < empty regions is the expectation); read the quantiles of $e(q_\theta)$ and the weights' range under $c$. **Decided:** $c$ fixed at 0.1 after reading the built bank's quantiles (§3.6).
7. **Smoke run (`04`):** short stages at real sizes, resume, logging of the new slices.

### Step 3 — Launch the A job

1. The A job (`A_carved_seed0`). **Launched.**
2. In the first hours, compare the training xMSE with the mini-bank by context-size bin. The mini-bank's shape no longer equals the stage-1 shape, so the old direct comparison does not apply.

The B jobs are not launched here: they wait until the Phase 0 tests are written (Step 5).

### Step 4 — While the A job runs

1. **Rewrite the four documents** as the current design only:
   - `iLTMv2_research_proposal.md`: phase map, Phase 0 test table (§2), loss exponent, arms, training procedure, the link between T6 and Phase 2, open items;
   - `phase0_protocol.md`: T1–T6, with the fairness text (§2.2) and the scatter text (§2.4);
   - `phase1_design.md`: thinning and subsampling as part of the task construction, arms A and B, validation bank, sweep and selection, the ceiling bracket;
   - `phase1_protocol.md`: layout, gates, launch plan, and the legacy block of §6 at the end.
2. **Phase 0 code:**
   - toy set: new `n_grid`, legend, per-model limits on rung 0;
   - heatmap generator and figure (T2);
   - PRR-A and PRR-B scripts on the active-learning splits (T3, T4);
   - retrieval script, starting with the resolution check (T6);
   - remove `cutgraph.py`, `transfer.py`, `grading.py`;
   - active-learning harness (T5), unchanged.
3. Run the baselines (heteroscedastic GP, TabICLv2 rows, Dec-ICL if the checkpoints arrive) on T1, T2 and T5.

### Step 5 — Launch the B jobs (after the Phase 0 tests are done)

1. The 18 B jobs (§4), once the Phase 0 tests of Step 4 are written and their baselines run. B was deferred to prioritise Phase 0; nothing in Steps 3–4 depends on it.
2. In the first hours, the same training-vs-mini-bank check as in Step 3, plus `fork_dist_from_init` at the higher fork LRs (§7.2, doubt 4).

### Step 6 — Select and evaluate

1. Pick learning rates per cell from validation (§3.10, step 1).
2. Run the four carved winners on T2–T5 and choose the final model.
3. Run the final model on T1 and T6.
4. Run the ablation counterparts on T2–T5.
5. Add the new best checkpoints to the Phase 1 document.

---

## 6. Legacy checkpoints

Text for the end of the new `phase1_protocol.md`.

> ### Legacy checkpoints (first training round)
>
> Checkpoints of the first training round are kept on the group disk. Their training code and full protocol are in the archived repository (`/private/groups/ioannidislab/UQ_project_v1`, commit `c1ed503d0a269051724ee61485862b96472eee45`). Paths are under `/private/groups/ioannidislab/UQ_decoupling_exp/`.
>
> | Arm (old name) | Head | Checkpoint | Head LR / fork LR, seed | Pooled val MSE | Val skill (all / iid / hole / extrapolation / duplicate) |
> |---|---|---|---|---|---|
> | A1 (now A) | $\hat A$ | `runs/A_seed1/stage3/best/A1_lr1e-3_A.pt` | 1e-3, seed 1 | 0.004041 | 0.891 / 0.910 / 0.872 / 0.825 / 0.916 |
> | A1 (now A) | $\hat E_{\text{dir}}$ | `runs/A_seed0/stage3/best/A1_lr1e-3_E.pt` | 1e-3, seed 0 | 0.02117 | 0.420 / 0.370 / 0.447 / 0.474 / 0.496 |
> | A2 (removed) | $\hat A$ | `runs/A_seed1/stage3/best/A2_lr1e-3_A.pt` | 1e-3, seed 1 | 0.003812 | 0.898 / 0.917 / 0.880 / 0.829 / 0.922 |
> | A2 (removed) | $\hat E_{\text{dir}}$ | `runs/A_seed1/stage3/best/A2_lr1e-3_E.pt` | 1e-3, seed 1 | 0.02108 | 0.423 / 0.375 / 0.443 / 0.476 / 0.526 |
> | B | $\hat A$ | `runs/B_h3e-4_f3e-5_seed0/stage3/best/B_A.pt` | 3e-4 / 3e-5, seed 0 | 0.003678 | 0.902 / 0.922 / 0.886 / 0.829 / 0.924 |
> | B | $\hat E_{\text{dir}}$ | `runs/B_h3e-4_f3e-5_seed0/stage3/best/B_E.pt` | 3e-4 / 3e-5, seed 0 | 0.02058 | 0.436 / 0.384 / 0.472 / 0.487 / 0.513 |
>
> All are the end of stage 3 (global step 55,000). Skill and MSE are on the first validation bank and are not comparable with numbers on the current one.
>
> **Shared training configuration.** 50,000 / 4,000 / 1,000 steps; 32 tasks per step; AdamW (weight decay 1e-5, clipping 5.0); cosine schedule restarted per stage, LR divided by 1 / 8 / 40; $p=0$; $R=16$.
>
> **How they differ from the current training.**
>
> - **Data.** Holes and tails were always fully emptied (no thinned regions). One context per task: 30–90% of 1,024 rows in stage 1, so no context below about 307 rows.
> - **Batch.** One forward per task (no subsampled pass).
> - **Loss.** Unweighted only.
> - **Arms.** A1 is the current arm A. A2 (softmax mix of 12 residual taps) has no counterpart in the current code and needs the archived repository to load.
> - **Validation.** First bank: small tasks of 1,024 rows at 30–90% context, no thinned regions, unweighted metrics.

---

## 7. Design liberties and doubts

### 7.1 Liberties: choices made here that were not explicitly decided

| # | Where | Choice |
|---|---|---|
| 1 | §2 | Test numbering T1–T6 |
| 2 | §2.3 | Background ranges (density 64–1,024 points per unit, noise 0.05–0.5), strip width 2 and random position, 50 queries, 20 toys (10 for the GP), the "background" last column, the 0.01 floor |
| 3 | §2.6–2.7 | Reusing the active-learning splits and seeds; context = warm start + pool; small context = 256 rows; one small context per seed; Dec-ICL rows only if checkpoints arrive |
| 4 | §2.8 | $S_0$ of 256 rows, budgets 16/64/256, 8 rounds, shortlist 32 + 32, the $e(q)$-greedy control, 200 queries, 5 seeds |
| 5 | §3.4 | The six context-size bins; weighted skill against the weighted-mean constant |
| 6 | §3.6 | $c=0.1$ (fixed by the user after the bank's $e(q_\theta)$ quantiles; the bank median was rejected) |
| 7 | §3.7 | $\rho$ defined as a share of the region's non-query rows; independent draws for hole and tail; a thinned region returns at least one row |
| 8 | §3.8–3.9 | Subsampling kept in the i.i.d. prior; 32 tasks and 64 forwards per step; all queries kept in the subsampled pass |
| 9 | §3.10 | LR chosen per cell by its own weighting; the four carved winners as candidates for the final model |

### 7.2 Doubts about the new design

1. **The validation bank has no context below about 205 rows** (256 rows at 80%). Training now goes down to 64 and the toy set uses 64. The effect of the subsampled pass would be invisible in validation. Drawing the small tasks from 80 rows up (context 64) would cover it.
2. **Model selection on the test data.** The final model and $p$ are chosen with T2–T5, and the same tests are then reported. That is mildly optimistic. Either say so, or select on a fixed subset (for example T2 plus one dataset) and report on the rest.
3. **Small contexts may be forgotten in stages 2–3.** The old runs already lost a little at small context there. Now stage 1 is shorter, stages 2–3 are longer, and subsampling exists only in stage 1. Watch `n_le256` and `n_256-512` at steps 34,000–40,000. If they degrade, mix a share of stage-1-shape tasks into stages 2–3.
4. **Higher learning rates.** Head LR 1e-3 already showed spikes around step 15,000, so 3e-3 may be unstable. Fork LR 3e-4 is ten times the old maximum, with no penalty pulling the fork towards its pretrained weights; watch `fork_dist_from_init`.
5. **No ground-truth test beyond 1-D.** With the cut-graph and transfer tests gone, the only "is the number right" evidence in Phase 0 is 1-D with Gaussian noise. What the observed cut gives up (latent confounding) is no longer measured anywhere. A reliability plot from the validation bank costs nothing and could sit in the appendix.
6. **PRR-A may show a small gap.** $\hat A+\hat E$ and $e(q)$ differ only where the backbone is miscalibrated. On ordinary splits of these five datasets the difference could be within noise.
7. **PRR-B and retrieval depend on how sensitive $\hat E$ is to the context.** Both are new constructions with no published reference numbers. In PRR-B, 256 rows against 6,024 is my choice of contrast; the result may depend on it.
8. **The `n_grid` change** was made after seeing iLTMv2's toy figures. The reason is independent of the outcome, but it is a deviation from the "fixed before the first run" rule and should be stated once in the new Phase 0 document.

---

## References

- Bergna, R., Depeweg, S., & Hernández-Lobato, J. M. (2026). Decoupled PFNs: Identifiable epistemic-aleatoric decomposition via structured synthetic priors. arXiv:2605.06413
- Depeweg, S., Hernández-Lobato, J. M., Doshi-Velez, F., & Udluft, S. (2018). Decomposition of uncertainty in Bayesian deep learning for efficient and risk-sensitive learning. *ICML*.
- Hofman, P., Sale, Y., & Hüllermeier, E. (2024). Quantifying aleatoric and epistemic uncertainty with proper scoring rules. arXiv:2404.12215
- Houlsby, N., Huszár, F., Ghahramani, Z., & Lengyel, M. (2011). Bayesian active learning for classification and preference learning. arXiv:1112.5745
- Lahlou, S., Jain, M., Nekoei, H., Butoi, V. I., Bertin, P., Rector-Brooks, J., Korablyov, M., & Bengio, Y. (2023). DEUP: Direct epistemic uncertainty prediction. *TMLR*.
- Qu, J., Holzmüller, D., Varoquaux, G., & Le Morvan, M. (2026). TabICLv2: A better, faster, scalable, and open tabular foundation model. arXiv:2602.11139
