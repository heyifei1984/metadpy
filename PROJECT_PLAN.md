# PROJECT_PLAN.md

## Project title (working)
RHMetaDPy: Hierarchical regression meta-d' / Mratio modeling in Python

(Alternate names: `metadpy-rhmetad`, `metadpy-regression`, `rhmetad`)

## Goal
Implement a hierarchical Bayesian regression extension for metacognitive efficiency (Mratio = meta-d'/d') that supports continuous subject-level covariates (e.g., trait measures) within the generative model.

This should be implemented as an extension to the repo’s existing hierarchical Bayesian meta-d'/HMeta-d-style model, not as a post-hoc regression on point estimates.

## Scope (v1)
### In scope
- A new regression-capable model function, e.g. `rhmetad(...)` (name can be finalized later).
- Continuous covariates at the subject level:
  - Support 1+ regressors
  - Include an intercept
  - Provide recommended preprocessing (centering / z-scoring) with clear documentation
- Returns:
  - Posterior for regression coefficients (beta)
  - Posterior for group-level parameters
  - Posterior for subject-level metacognitive efficiency (Mratio or logMratio)
  - A tidy summary table and a trace object (or the repo’s standard output)
- Tests:
  - Unit tests for shapes and invariants
  - Simulation-based parameter recovery test(s)

### Out of scope (v1)
- Arbitrary per-trial regressors (trial-level predictors)
- Complex multilevel structures beyond “subject-level covariates”
- Model comparison framework (WAIC/LOO) unless already present
- GPU acceleration work

## Milestones (PR-oriented)

### Milestone 0 — Repo scaffolding (PR-0)
Deliverables:
- `AGENTS.md` (this repo’s agent instructions)
- `PROJECT_PLAN.md`
- `docs/` folder created (if not already present)
- Issue templates (optional)

Acceptance:
- Codex can read these files and reliably follow them.

### Milestone 1 — Reference + spec (PR-1)
Deliverables:
- `docs/model_spec_rhmetad.md`
  - Defines the model in equations
  - Defines data inputs (trial DF + subject ids + confidence bins)
  - Defines exactly what is regressed (recommended: log(Mratio))
  - Defines priors (match existing repo defaults where possible)
  - Notes anything UNVERIFIED with a tracking issue link
- `docs/reference_notes_hmetad.md`
  - Short notes on how the MATLAB toolbox organizes parameters and naming
  - Links to key reference files / docs

Acceptance:
- Spec is precise enough that an independent dev could implement it.

### Milestone 2 — Minimal implementation (PR-2)
Deliverables:
- New module/function implementing the regression model
- Minimal example script or doc snippet showing usage on simulated data
- Outputs include regression coefficients + Mratio/logMratio

Acceptance:
- Runs end-to-end on a small simulated dataset
- No breaking changes to existing APIs

### Milestone 3 — Validation tests (PR-3)
Deliverables:
- Simulation utilities (if missing) to generate data with known beta
- Parameter recovery tests:
  - Fit model on simulated dataset
  - Verify posterior for beta concentrates near ground truth (within tolerance / HDI rule)
- Posterior predictive check (basic)

Acceptance:
- Tests pass in CI
- Recovery test is stable (does not flake)

### Milestone 4 — Documentation (PR-4)
Deliverables:
- A tutorial page / notebook showing:
  - Data format
  - Adding covariates
  - Interpreting beta posterior
- API docs updated
- Notes on limitations (edge cases: sparse errors, extreme confidence)

Acceptance:
- A new user can run the tutorial without reading source code.

### Milestone 5 — Release hygiene (PR-5)
Deliverables:
- Version bump (if repo uses versioning)
- Changelog entry
- Citation info (paper/toolbox references)
- Optional: example figure generation

Acceptance:
- Releasable state with docs + tests.

## Acceptance criteria (definition of done)
- A public regression API exists and is documented.
- The regression is inside the hierarchical model (not post-hoc).
- Simulation-based tests demonstrate beta recovery.
- Clear warnings/notes exist for data regimes known to be fragile for meta-d' models.

## Operating procedure with Codex (suggested)
### If using Codex Cloud
- Use a cloud task to implement one milestone per PR.
- Require the agent to open a PR and include:
  - Summary of changes
  - How it was tested
  - Any remaining UNVERIFIED items

### If using Codex CLI
- Run in interactive mode from the repo root and work milestone-by-milestone.
- Keep a clean git history; checkpoint before/after major changes.

## Risk register
- R1: Subtle likelihood / bin-order mistakes → mitigated by recovery tests + PPC.
- R2: Covariate misalignment to subjects → mitigated by explicit joins and tests.
- R3: Overfitting / identifiability issues in sparse data → mitigated by priors + docs.
