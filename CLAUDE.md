# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview
Experimental work for the iLTMv2 model: building a tabular foundation model (based on TabICLv2) with calibrated, decomposed epistemic/aleatoric uncertainty as a first-class deliverable, plus a coherence-based validity gate. In later stages, this uncertainty decomposition and coherence value will be used to improve model performance (context retrieval or prior design).

The authoritative source for design rationale, the three-phase structure, and all mathematical definitions is `iLTMv2_research_proposal.md` at the repo root — read it whenever a design decision, a symbol's meaning, or a proposal section number comes up. Key sections: §2 the score decomposition and its unbiased replicate estimators (shared by all phases), §3 Phase 0 (Toy-set tests, graph domain tests, and active learning tests + baseline converters), §4 Phase 1 (model training: supervision targets, why exact replication needs the observed-cut prior + its architecture + its training process), §5 Phase 2 Epistemic-guided context. Some parts of the project are still TBD, and they are marked as so.

Whenever the .md files in a sub-folder (e.g. `phase0_protocol.md`, `phase1_design.md`...) contradict the `iLTMv2_research_proposal.md`, do not act on any decision and ask the user for which action to take. The taken action will updated the document that contradicts it.

## Environment
- Conda environment: `uq-iltm-bio` (activate before running anything). Env spec is NOT `../environment.full.yml` at the outer repo root - we still haven't created a .yml file for this env.
- `tabicl` and `ud4pfn-B017` are NOT installed from PyPI/conda — they are editable installs (`pip install -e .`) of clones under `external/tabicl/` and `external/ud4pfn-B017/`. Edit those clones' source directly; changes take effect immediately, no reinstall needed. If an import fails, re-run `pip install -e external/tabicl` / `pip install -e external/ud4pfn-B017` from this directory. UD4PFN is installed but unused in the current version. Phase 1's own package `iltm2` is installed the same way (`pip install -e experiment1_uq_heads --no-deps`).
- The heteroscedastic GP (PyMC with the numpyro backend) runs in `uq-iltm-bio` too; its dependencies are already installed there.
- `external/tabicl/` — the real, full-scale TabICLv2 repo. Prior code lives in `external/tabicl/src/tabicl/prior/` (`_mlp_scm.py`, `_tree_scm.py`, `_graph_scm.py`, `_hp_sampling.py`, `_dataset.py`, `graph_lib/`). Both released TabICLv2 checkpoints were pretrained exclusively on the graph-based SCM prior (`graph_scm`), so `tabicl.prior.graph_lib` (used directly via `RandomDataset`/`RandomGraphFunction`) is the generator experiment1 and later work build on — see the abduction rule below for why exact replicates need the observed-cut variant of it. `tabicl.prior.MLPSCM` / `tabicl.prior.TreeSCM` are TabICLv1's prior; no v2 checkpoint was trained on them, so they are **not used** in this project.
- **`libstdc++` ABI mismatch gotcha:** importing `tabicl` (it pulls in `sklearn` → `scipy.spatial`) fails with `ImportError: ... CXXABI_1.3.15' not found` under plain `conda activate uq-iltm-bio` — the cluster's system `libstdc++.so.6` only provides up to `CXXABI_1.3.13`, but this env's compiled scipy/sklearn wheels need `1.3.15`, which only exists in the env's own bundled `libstdc++`. This reproduces both on the login node and inside SLURM jobs. Fix: prepend the env's lib dir before importing anything: `export LD_LIBRARY_PATH="/private/groups/ioannidislab/ajaras/conda_envs/uq-iltm-bio/lib:$LD_LIBRARY_PATH"` right after `conda activate uq-iltm-bio`.

## The abduction rule (why the generator choice matters)
Exact conditional replication requires sampling the **abduction posterior** `p_θ(U | Φ_θ(U) = x)` over the randomness `U` shared between `X` and `y`, then pushing it forward with fresh target-side noise. For arbitrary nonlinear `graph_scm` graphs this may be computationally difficult or may not admit a convenient closed form. Having the observed features **d-separate `y` from the remaining randomness** makes the posterior equal to the prior, so the abduction step disappears — a simple and sufficient route to exactness. (Sufficient, not necessary: linear-Gaussian shared paths, invertible width-preserving `Φ_θ`, and discrete `X` are also tractable.)

**What this means in practice.** The unmodified graph prior does not d-separate `y`: `x` and `y` share ancestors by construction, and node functions are dense or semi-dense maps that blend every input coordinate into every output coordinate. So exact replicates require the **observed-cut prior** (§4.3): generate `X` with the unmodified generator, then produce `y` from a small second graph that takes `X` plus its own private noise roots. Replicates = hold `X`, redraw the private roots, re-run the target branch. When adding a generator or a sampling path, check this before building on it.

**Graph-prior generator gotchas:**
- **As shipped, a second call to the graph function is a different τ.** `RandomNodeFunction._transform` re-draws the root point law (`RandomPoints.sample` rebuilds its `RandomFunction` and re-picks the base-point family every call), the per-node feature-importance weights, the `node_importance` scalar, and the noise σ if enabled. Cache all of these in `_fit` so only the root base points and the softmax draws are redrawn — this is a **prerequisite** for any replicate scheme, and it also silently invalidates any pool assembled from more than one call.
- **`PriorConfig.ensure_iid` defaults to `False`.** Set it to `True`. Otherwise `Standardize`, `RandomRescale`, `KumaraswamyWarping` and `CategoricalNeighborDiscretizer` are fit on the same rows they transform, so rows are not conditionally i.i.d. given τ.
- **Never run the replicate construction on an unmodified graph task.** With node noise off there is nothing downstream of `x` to redraw, so freezing root draws gives `ỹ ≡ y` (target silently zero); freezing only the feature-ancestor dims under-reports it instead. Valid only on the observed-cut prior — enforce with an assert that `X` is bit-identical across replicates.
- `add_gaussian_noise` is `False` in every v2 recipe; all aleatoric spread in the shipped prior is omitted-variable uncertainty.
- `RandomDataset.sample` iterates a Python `set`, so runs need `PYTHONHASHSEED=0` or the same seed gives different tasks.

## Running things on the cluster (SLURM)
This is a shared SLURM cluster (`phoenix-*` nodes) — do not run anything compute-heavy (model imports, tree/xgboost fits, training) directly in an interactive shell; the login/dev node has no guaranteed compute and shells there can hang indefinitely on things like `import sklearn`. Submit a batch job instead, even for quick sanity checks.

- Partitions (`sinfo`): `short` (1h, default), `medium` (12h), `long` (14d), `gpu` (7d, needs `--gres=gpu:...`), `high_priority` (7d). For a quick CPU-only check, use `short` — it queues fastest since it asks for the least.
- Minimal ad-hoc test job pattern (adapt `--time`/`--mem`/`--cpus-per-task` to the job):
  ```bash
  #!/bin/bash
  #SBATCH --job-name=my_check
  #SBATCH --partition=short
  #SBATCH --time=00:05:00
  #SBATCH --mem=4G
  #SBATCH --cpus-per-task=2
  #SBATCH --output=/private/home/ajaras/iLTM-bio/slurm_logs/my_check_%j.out
  #SBATCH --error=/private/home/ajaras/iLTM-bio/slurm_logs/my_check_%j.err

  eval "$(conda shell.bash hook)"
  conda activate uq-iltm-bio
  export LD_LIBRARY_PATH="/private/groups/ioannidislab/ajaras/conda_envs/uq-iltm-bio/lib:$LD_LIBRARY_PATH"
  cd /private/home/ajaras/iLTM-bio/UQ_project
  python -c "..."
  ```
  Submit with `sbatch script.sh` (prints `Submitted batch job <ID>`); check with `squeue -j <ID>`; logs land in `~/iLTM-bio/slurm_logs/`.
- The outer repo's own SLURM scripts live in `../sbatch/` (`train_iltm.sh`, `test_gpu.sh`, etc.) and mostly activate a *different* env (`iltm-bio-dev`) for the outer TabularMaestro/TabBioGen projects — don't copy that env name into UQ_project jobs; UQ_project uses `uq-iltm-bio`.
- Each experiment's `scripts/*.sh` follows this same pattern, except that logs go to `<experiment>/outputs/logs/<name>_%j.{out,err}` (experiment1: `/private/groups/ioannidislab/UQ_decoupling_exp/slurm_logs/`) instead of `~/iLTM-bio/slurm_logs/`. The `.sh` wrappers `cd` to this directory and forward their arguments to the matching `.py`, e.g. `sbatch experiment0_baseline/scripts/01_toy.sh --method hgp`. Run a single validation gate or unit test the same way, by submitting its `.sh` wrapper (e.g. `sbatch experiment0_baseline/scripts/00b_validate_estimators.sh`), then read the OK/FAIL lines in its `outputs/logs/` file.
- Beyond the minimal pattern, every experiment `.sh` also exports `PYTHONHASHSEED=0` (see the graph-prior gotcha below). The heteroscedastic-GP job (`01_toy.sh`) additionally sets `JAX_PLATFORMS=cpu`, `XLA_FLAGS="--xla_force_host_platform_device_count=2"` (one numpyro chain per host device, matching `chains` in `configs/methods.yaml`), and a per-job `PYTENSOR_FLAGS="compiledir=/tmp/pytensor_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"` so concurrent array tasks don't collide on PyTensor's compile cache. Copy these when writing a new PyMC/numpyro job. Array jobs log to `<name>_%A_%a.{out,err}`.

## Repo conventions
Each experiment is a top-level folder (`experimentN_name/`) containing:
- `CLAUDE.md` — that experiment's own thorough description, status, method, and gotchas. **Read it before working in that folder**, and keep its status sections (`## Current task`, `## Waiting on`) current.
- `scripts/` — numbered entry points as `NN_name.py` + `NN_name.sh` (SLURM wrapper) pairs (experiment1 also has `train_A.sh` / `train_B.sh` launchers, as its protocol specifies). Scripts run from this directory. How they import the experiment's code depends on its `src/`: experiment0's `src/` is a flat module folder — scripts put it on `sys.path` and import modules flat (`import evaluate as ev`); experiment1's `src/iltm2/` is an editable-installed package (`pip install -e experiment1_uq_heads`), imported as `iltm2.*` from its own scripts and from experiment0 (e.g. `iltm2.prior` for the T2 bank). Put code that another experiment must import in such a package. `00*` scripts are validation gates that must pass before the measurement scripts (`01_*`, …) are run. Unit tests are plain scripts in the same series (e.g. `00b_validate_estimators.py`, `00c_validate_adapters.py`): bare checks that print OK/FAIL per case and exit non-zero on any failure. pytest is not installed in `uq-iltm-bio`.
- `src/` — the experiment's own Python code (generator patches, estimators, normalization, plotting).
- `configs/` — YAML constants, loaded through `src/config.py` (which also defines the output paths). experiment0 currently has a frozen `protocol.yaml`, `toy_function.yaml`, `toy_rung{0,1,2}.yaml`, and `methods.yaml` (per-model settings, incl. the TabICLv2 checkpoint path); `cutgraph.yaml`, `transfer.yaml` and `al.yaml` are planned for T2–T4 and do not exist yet.
- `outputs/` — `raw/` (parquet), `figures/`, `tables/`, `logs/` (SLURM output), `archive/v<k>/` (superseded runs). Exception: experiment1 writes its heavy outputs (backbone copy, validation bank, checkpoints, wandb, SLURM logs, audits) to `/private/groups/ioannidislab/UQ_decoupling_exp/` (`phase1_protocol.md` §2).

Shared code across experiments goes in the top-level `common/` (currently empty).

`SCRAPPED/` holds a superseded design of experiment0 (its old protocol, other model adapters, outputs... Both from v1 and v2). It does not match the current protocol; do not import from it or copy code out of it without checking each piece against `phase0_protocol.md`.

The outer repo (`../`) is a separate, larger iLTM-bio project (TabBioGen, TabularMaestro, `../sbatch/`, `../ckpts/`); UQ_project is self-contained within it.

## Experiments
- `experiment0_baseline/` — Phase 0. Establish how the heteroscedastic GP (T1 only) and Dec-ICL (DecoupledPFN) split aleatoric vs epistemic uncertainty, graded against the realized task's exact truth: T1 toy set (one curated 1-D function, toy rungs 0–2, closed-form truth, figures only), T2 cut-graph test (observed-cut prior, replicate-based, with context holes), T3 transfer test (unmodified `graph_scm`, exact-match replicates on low-cardinality categorical features). TabICLv2 is a total-only reference (backbone check + T4), not a baseline. The epistemic truth is the excess CRPS of each model's *own* forecast, so every adapter must return its forecast CDF, not just Â/Ê. Work proceeds in the order of the protocol's §11 (T1 toy set and backbone check written; Dec-ICL, T2, T3, T4 next); its §1 gives the target file layout. Outputs are standardised to our proper scoring rule decomposition conventions, so models can be compared with ours. Then, do an Active Learning test to see how epistemic-based context retrieval works against PPD spread retrieval. The spec lives in `experiment0_baseline/phase0_protocol.md` — read it before writing any code there; its own `CLAUDE.md` is a thin pointer into that protocol plus the invariants/status that must stay in view on every edit.

- `experiment1_uq_heads/` — Phase 1 (in progress; a thin `CLAUDE.md` with status, plus two specs). `phase1_design.md` covers *what* is trained (heads, targets, loss, arms, curriculum, sweeps); `phase1_protocol.md` covers *how* it is built and run (code layout, storage, gated order of work, unit tests, smoke runs, wandb logging, launch plan). Where they disagree, design wins on what, protocol on how — and the research proposal outranks both (see the contradiction rule above). Train the new model's heads to estimate the aleatoric and epistemic uncertainty. Establishing a correct loss function and thorough training process, various architectures will be trained (sweeping through a set of hyperparameter combinations) to get the best performing checkpoint. Each architecture tries to decode the uncertainty's entropy from a different point of the backbone, checking what embedding is the most informative about the posterior distribution.

## Common estimators / conventions
- **Project scope is regression / CRPS.** The Brier instantiation (`e = 1 − Σp_k²`, estimator `1{y⁽¹⁾≠y⁽²⁾}`) is identical in form and is stated in the proposal, but is out of scope for the timeline. The log score has no unbiased finite-R estimator; secondary reporting only.
- Aleatoric target (R exact replicates from the observed-cut prior): unbiased CRPS-entropy `1/(R(R−1)) · Σ_{r<s}|y⁽ʳ⁾−y⁽ˢ⁾|` (R=2 case: `½|y⁽¹⁾−y⁽²⁾|`). R=2 suffices — the target must be unbiased, not accurate. However, higher R values should be used, since the convergence will be faster and identical sample drawing is not expensive.
- Divergence target for head 2, from the same two replicates: `½[S(q_θ,y⁽¹⁾)+S(q_θ,y⁽²⁾)] − ½|y⁽¹⁾−y⁽²⁾|`, also exactly unbiased.
- On the *unmodified* graph prior, no exact replicates exist (see the abduction rule above). The only ground truth available there is **exact matching on discrete features**, and it is an audit instrument, not a training target. Future designs may use kNN / kernel-smoothed draws as targets; never clip targets at zero (the point mass at exactly zero is real and a clip biases it upward).
- Freeze the post-processing chain (reg2cls, scaling, rejection filters) into a `fit`/`apply` pair and apply the **observed dataset's** constants to every replicate — never recompute per replicate. The prior's rejection filters truncate the high-aleatoric tail; log the empirical distribution of `e(p_τ)`.