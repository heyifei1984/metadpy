# PROJECT_PLAN.md — RHMeta-d regression (MATLAB/JAGS parity)

## Goal (v1)
Implement the **hierarchical regression** model from the MATLAB/JAGS HMeta-d toolbox in this metadpy fork:
- MATLAB entry point: `fit_meta_d_mcmc_regression.m`
- JAGS model: `Bayes_metad_group_regress_nodp*.txt` (1–5 covariates)

Key properties of the v1 model:
- Regression is on **log(meta-d′/d′)** (“logMratio”).
- Uses a robust **Student-t (df=5)** subject deviation term with parameter expansion (epsilon * delta).
- Uses “**nodp**”: type-1 d′ and c are computed outside and treated as **fixed data**.

---

## Scope (v1)
### In scope
- New regression-capable function, consistent with existing API style (proposed: `metadpy.bayesian.rhmetad`).
- Support **P ≥ 1** covariates (not limited to 5, even though MATLAB ships 1–5-cov JAGS templates).
- Maintain metadpy backend pattern:
  - `backend="pymc"` (default)
  - `backend="numpyro"` (JAX sampler path)
- Documentation:
  - RHMetaD.md as the precise model contract
  - short usage example

### Out of scope (v1)
- Trial-level regressors
- Response-conditional regression model (unless explicitly specified)
- Estimating d′ inside the model (that corresponds to a different JAGS file family)

---

## Milestones (PR-oriented)

### M0 — Reference import + spec hardening
Deliverables:
- `references/hmetad_matlab/Matlab/` contains the minimum reference set listed in AGENTS.md.
- `RHMetaD.md` updated to:
  - match the JAGS equations/priors exactly
  - define I/O contract + expected variable names
  - document any necessary PyMC adaptations (e.g., probability floor + renormalization)

Acceptance:
- No **UNVERIFIED** items remain for the v1 regression model.

### M1 — Core model implementation
Deliverables:
- Internal model builder (location consistent with repo conventions, e.g. `metadpy/models/rhmetad.py`).
- Public API wrapper `metadpy.bayesian.rhmetad(...)` following the `hmetad` style:
  - supports `sample_model=True/False`
  - supports `backend="pymc" | "numpyro"`
  - returns `(model, idata)` or a DataFrame if `output="dataframe"` (match repo norms)

Acceptance:
- Runs end-to-end on a small simulated dataset.

### M2 — Validation tests
Deliverables:
- `test_rhmetad_shapes.py` (shapes + invariants)
- `test_rhmetad_recovery.py` (β recovery with simulation consistent with the model)
- `test_rhmetad_ppc.py` (minimal PPC)

Acceptance:
- `pytest` passes
- recovery is stable (avoid flaky thresholds)

### M3 — Documentation + example
Deliverables:
- Update docs pages / docstrings similarly to `hmetad`:
  - “how to supply covariates”
  - “how to interpret β” (exp(beta) multiplicative effect on Mratio)
- Note on Apple Silicon:
  - CPU path is the default
  - JAX/Metal path is “best-effort / experimental” and depends on user environment

Acceptance:
- A user can run RHMeta-d from docs alone.

---

## Definition of done (v1)
- RHMeta-d regression implemented with MATLAB/JAGS-parity priors and likelihood.
- β recovery demonstrated via tests.
- Backend selection integrated cleanly and consistently with existing `hmetad` patterns.
