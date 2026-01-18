# PROJECT_PLAN.md

## Project name
RHMetaD: Regression Hierarchical Meta-d for metadpy

## Goal
Add hierarchical Bayesian **regression** modeling of metacognitive efficiency (log M-ratio) to `metadpy`, mirroring MATLAB HMeta-d regression (nodp) behavior.

## Primary success criteria
- A new function (proposed name: `rhmetad`) fits a group regression model and returns posterior samples.
- Matches MATLAB regression logic:
  - fixed type-1 d′ and c1 point estimates per subject,
  - hierarchical regression on log(Mratio),
  - same priors (mu_c2, sigma_c2, Student-t delta, Beta epsilon),
  - same likelihood construction for CR/FA/M/H counts.
- Works on Apple Silicon CPU; optional MLX acceleration is exposed via a clean argument.

## Milestones

### M0 — Lock the spec (docs-only)
**Deliverables**
- `docs/RHMetaD.md` fully specifies:
  - data conventions and ordering
  - pre-processing for d1/c1
  - exact priors
  - likelihood and probability equations
  - outputs and naming

**Exit criteria**
- No “UNVERIFIED” sections remain for the MATLAB regression model itself.

---

### M1 — Data preprocessing utilities (group-level)
**Tasks**
- Implement `preprocess_group(...)` (or equivalent) that:
  - groups trials by subject
  - calls `trials2counts(...)`
  - calls `extractParameters(...)` for each subject
  - returns arrays: `d1[S]`, `c1[S]`, `counts[S, 4K]`, plus `CR, FA, M, H`
- Implement covariate ingestion:
  - from DataFrame columns OR from matrix
  - optional z-scoring and NaN handling
  - strict alignment of cov rows to subject order

**Exit criteria**
- Unit test: given a small synthetic dataset with 2 subjects, output shapes match expectation.

---

### M2 — PyMC model builder (group regression)
**Tasks**
- Implement a model builder function:
  - e.g., `build_rhmetad_model(data_dict, X, ...) -> pm.Model`
- Must include:
  - `mu_logMratio`, `beta[p]`, `sigma_delta`, `epsilon_logMratio`
  - subject-level `delta[s]` and `logMratio[s]`
  - `Mratio[s] = exp(logMratio[s])`
  - `meta_d[s] = Mratio[s] * d1[s]`
  - criteria priors with ordering constraints
  - multinomial likelihood split into CR/FA/M/H (as in MATLAB regression scripts)
- Add optional `Tol` clamping for probabilities.

**Exit criteria**
- Smoke test: model builds and samples for 3 subjects × 4 ratings without errors.

---

### M3 — Public API + ergonomics
**Tasks**
- Add `rhmetad(...)` to `metadpy.bayesian` (and `__init__.py` if needed).
- Signature should support:
  - DataFrame + column names + subject + covariates
  - or precomputed `nR_S1/nR_S2` + `X`
- Provide sampling options:
  - `num_samples`, `num_chains`, `tune`, `target_accept`, `random_seed`
- Add Apple Silicon acceleration hook:
  - `compile_mode: str | None = None`
  - `compile_kwargs: dict | None = None`
  - If `compile_kwargs` not provided and `compile_mode` is, pass `compile_kwargs={"mode": compile_mode}` to `pm.sample`.

**Exit criteria**
- Example snippet in docstring runs end-to-end.

---

### M4 — Verification & tests
**Tasks**
- Add tests:
  - shape validation errors (bad cov dims, wrong nRatings, etc.)
  - parameter recovery test:
    - simulate data with known positive β
    - fit model
    - assert posterior mean(β) > 0 (and roughly near true)
- If possible, add a minimal cross-check vs MATLAB outputs (optional):
  - same synthetic counts
  - compare posterior mean logMratio correlation (not exact match).

**Exit criteria**
- `pytest` passes on CI / local.

---

### M5 — Documentation
**Tasks**
- Add `docs/` page or README section:
  - what RHMetaD does
  - input formats
  - interpretation of β and logMratio
  - Apple Silicon: CPU vs MLX caveats

**Exit criteria**
- docs build (if repo uses sphinx) OR README renders.

## Risks / tricky parts
- Enforcing ordered type-2 criteria in a NUTS-friendly way while staying faithful to MATLAB’s “sample then sort” approach.
- MLX backend is still evolving; many ops may be missing; must be optional.

## Non-goals
- full d′ estimation inside regression model
- within-subject regression
- “perfect numerical parity” with MATLAB JAGS chains

