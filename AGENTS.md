# AGENTS.md

## Mission (what you are building)
Implement **HMeta-d (MATLAB/JAGS) parity** for the *hierarchical regression* model on metacognitive efficiency (meta-d′/d′ a.k.a. **Mratio**) inside this metadpy fork.

**Primary deliverable (v1):**
- A new public function (proposed: `metadpy.bayesian.rhmetad(...)`) that matches the MATLAB toolbox regression model implemented by:
  - `Matlab/fit_meta_d_mcmc_regression.m`
  - `Matlab/Bayes_metad_group_regress_nodp*.txt` (1–5 covariates)

This is the “**nodp**” regression model: **d′ and c are treated as fixed data** (computed from rating counts), and the hierarchical layer is on **logMratio** with a robust (t) subject-deviation term.

---

## Required reading before coding (do this first)
1) `PROJECT_PLAN.md`
2) `RHMetaD.md` (this repo’s spec/contract)
3) The existing `metadpy.bayesian.hmetad` implementation and its tests (for patterns, output formats, backend handling).
4) The copied MATLAB/JAGS reference files in this repo (see below).

---

## Reference files (must be present in-repo)
To avoid guessing, copy the following files from metacoglab/HMeta-d into this repo under:
`references/hmetad_matlab/Matlab/`

Minimum set:
- `fit_meta_d_mcmc_regression.m`
- `fit_meta_d_mcmc_group.m` (for data ordering explanation)
- `trials2counts.m` (for rating bin ordering)
- `Bayes_metad_group_regress_nodp.txt`
- `Bayes_metad_group_regress_nodp_2cov.txt`
- `Bayes_metad_group_regress_nodp_3cov.txt`
- `Bayes_metad_group_regress_nodp_4cov.txt`
- `Bayes_metad_group_regress_nodp_5cov.txt`
- `Bayes_metad_group_nodp.txt` (baseline non-regression “nodp” comparator)

If the repo already contains equivalent reference notes, do not duplicate; instead link to them.

---

## Scientific correctness rules (hard constraints)
### A) Regression target and equation (must match MATLAB/JAGS)
Regression is on **logMratio**:
- `logMratio[s] = mu_logMratio + Σ_j (mu_beta[j] * cov[j,s]) + epsilon_logMratio * delta[s]`
- `Mratio[s] = exp(logMratio[s])`
- `meta_d[s] = Mratio[s] * d1[s]`

Robust deviation term (as in the regression JAGS model):
- `delta[s] ~ StudentT(df=5, loc=0, scale=sigma_delta)` (JAGS uses precision `lambda_delta = sigma_delta^-2`)
- `epsilon_logMratio ~ Beta(1,1)`
- `sigma_logMratio = abs(epsilon_logMratio) * sigma_delta`

Priors (JAGS):
- `mu_logMratio ~ Normal(0, sd=1)`
- `mu_beta[j] ~ Normal(0, sd=1)`
- `sigma_delta ~ HalfNormal(sd=1)` (JAGS: Normal(0,1) truncated >0)

### B) Type 2 SDT likelihood and criteria priors (must match MATLAB/JAGS)
- The multinomial likelihood structure, probability construction via `phi(.)`, and the criteria priors/truncations must match `Bayes_metad_group_regress_nodp*.txt` (see RHMetaD.md for the full set of equations).

### C) d′ and c preprocessing (v1 must match MATLAB regression path)
- In v1, replicate the regression “nodp” model:
  - compute subject-level `d1[s]` and `c1[s]` from rating counts **outside** the Bayesian model, using the MATLAB algorithm (see RHMetaD.md).
  - pass `d1` and `c1` into the Bayesian model as **observed** arrays.

Any alternative (e.g., estimating d′ inside the model) must be treated as a separate v2 milestone.

---

## Backend and Apple Silicon guidance (keep it clean)
- Follow metadpy’s existing backend pattern (as in `hmetad`): support at least `backend="pymc"` and `backend="numpyro"` (JAX).
- **Do not promise GPU/Metal acceleration by default.**
  - If `backend="numpyro"` and the user has Apple’s `jax-metal` installed, JAX *may* use Metal GPU acceleration, but this is experimental upstream.
- Implementation rule:
  - Add backend selection in RHMeta-d without adding new heavyweight deps; rely on optional dependencies already used by `hmetad`.

---

## Non-goals (v1)
- Do not refactor existing `metadpy.bayesian.hmetad` unless required for code reuse.
- Do not add new PPL frameworks beyond what metadpy already supports.
- Do not implement trial-level regressors in v1.

---

## Tests (must add)
- Shape/invariant tests (probability blocks have correct shapes; no silent broadcasting errors).
- Simulation-based **β recovery** (regression coefficients).
- A minimal posterior predictive sanity check (PPC) or comparable check.

---

## PR checklist
- [ ] RHMetaD.md updated to reflect actual implementation details (no drifting spec)
- [ ] New tests added + passing (`pytest`)
- [ ] New public API documented + example snippet included
- [ ] Output includes posterior for β and Mratio/logMratio
- [ ] No unrelated refactors
