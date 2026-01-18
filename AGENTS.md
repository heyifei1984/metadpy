# AGENTS.md

## Mission

Implement **RHMetaD**: a hierarchical Bayesian *regression* extension of HMeta-d, in **Python / metadpy**, matching the logic of the MATLAB HMeta-d regression implementation as closely as is practical for PyMC (NUTS).

The end result should be “cloneable → installable → runnable” and should include:
- A new public API function (or method) that fits the regression model.
- A PyMC model builder for the regression model.
- Tests (at least: shape checks + parameter recovery on simulated data).
- Minimal docs / example usage.
- A clean optional switch for Apple Silicon acceleration modes (CPU vs MLX/Metal).

## Ground-truth references to mirror

The MATLAB HMeta-d regression implementation uses JAGS model scripts named like:

```text
HMeta-d/Matlab/Bayes_metad_group_regress_nodp.txt
HMeta-d/Matlab/Bayes_metad_group_regress_nodp_2cov.txt
...
HMeta-d/Matlab/Bayes_metad_group_regress_nodp_5cov.txt
HMeta-d/Matlab/fit_meta_d_mcmc_regression.m
HMeta-d/Matlab/trials2counts.m
```

If these files exist in your workspace, treat them as **ground truth**. If they are not present locally, the full specification is in `docs/RHMetaD.md`.

> IMPORTANT: In MATLAB, regression is implemented in the **no-d′ (nodp)** variant:
> - Type-1 **d′ (d1)** and **criterion (c1)** are computed as point estimates from the counts,
> - then held **fixed** in the hierarchical model,
> - and meta-d′ is defined as `meta_d = Mratio * d1`.

## Scope

### In-scope (v1)
- Group-level regression of **log(Mratio)** on continuous covariate(s), as in MATLAB regression scripts.
- Multiple covariates (any P, not limited to 5).
- Inputs:
  - either trial-level DataFrame + subject column + covariate columns
  - or precomputed `nR_S1`, `nR_S2` + covariate matrix
- Outputs:
  - posterior (InferenceData or MultiTrace) containing `mu_logMratio`, `beta`, `logMratio`, `Mratio`, `meta_d`,
    and also criteria hyperparameters `mu_c2`, `sigma_c2`, and derived `sigma_logMratio`.
- Optional Apple Silicon acceleration toggles:
  - `compile_mode="MLX"` option via `pm.sample(compile_kwargs={"mode": "MLX"})`
  - `compile_mode=None` (default CPU)
  - Allow passing full `compile_kwargs` through.

### Explicitly out-of-scope (v1)
- Full “estimate d′ inside the hierarchical model” regression variant (not provided by MATLAB regression scripts).
- Repeated-measures regression / within-subject regression.
- Extensive benchmarking / performance tuning.

## Implementation constraints

- **Do not break** existing public APIs.
- Keep changes minimal and well-contained:
  - New model file under `metadpy/models/` (e.g., `group_level_regression_pymc.py`)
  - New user-facing function under `metadpy/bayesian.py` (e.g., `rhmetad(...)`)
- Follow repo tooling (black, isort, mypy if present; run existing test suite).
- Use vectorized PyTensor operations where possible.
- Handle common input errors with clear messages:
  - mismatched number of subjects between data and covariates
  - non-finite covariates
  - bad `nRatings`
  - invalid count shapes

## Quality bar / definition of done

- ✅ `pytest` passes locally
- ✅ basic simulated-data recovery test passes (β sign and rough magnitude)
- ✅ docs string example runs
- ✅ model runs end-to-end on small dataset (2–5 subjects, 4 ratings)

## Suggested work plan (agent)

1. **Read** `docs/RHMetaD.md` and confirm the model equations & data conventions.
2. Implement or finalize `preprocess_group(...)` (currently commented in `metadpy.bayesian` in older versions).
3. Implement PyMC regression model builder:
   - logMratio regression with robust Student-t + epsilon expansion (as in MATLAB)
   - criteria priors as in MATLAB (ordering constraints handled in a NUTS-compatible way)
4. Add `rhmetad(...)` entry point:
   - accepts either DataFrame or `nR_S1/nR_S2`
   - accepts covariates as DataFrame columns or matrix
   - exposes `compile_mode` / `compile_kwargs`
5. Tests: at minimum
   - shape validations
   - deterministic simulation + recovery (β > 0 should be recovered > 0)
6. Docs: short usage snippet and parameter notes.

