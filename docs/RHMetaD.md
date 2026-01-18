# RHMetaD (RHMeta-d) specification — metadpy regression extension (v1)

**Purpose:** Add subject-level continuous regression on metacognitive efficiency inside hierarchical meta-d′.

---

## 1) Target quantity and regression

### 1.1 Definitions
For subject s:
- Mratio_s = meta-d′_s / d′_s
- M_s = log(Mratio_s) = log(meta-d′_s / d′_s)

### 1.2 Regression (embedded)
Let X be design matrix with shape (S, P).
- M_s = M0 + X_s · β + ε_s

Interpretation:
- Mratio_s = exp(M_s)
- A +1 change in predictor j multiplies Mratio by exp(β_j), holding others fixed.

---

## 2) Robust noise and parameter expansion (RHMeta-d style)

Use a robust subject deviation term (outlier-tolerant):
- δ_s ~ StudentT(df=5, loc=0, scale=σ_δ)
- ζ ~ Beta(1, 1)
- ε_s = ζ · δ_s

Priors:
- M0 ~ Normal(0, 1)
- β  ~ Normal(0, 1)   (vector length P)
- σ_δ ~ HalfNormal(1)

Notes:
- ζ is included for parameter expansion / sampling stability (as described in RHMeta-d context).
- If metadpy already has a different parameter-expansion pattern, document and justify any deviation.

---

## 3) Likelihood (meta-d′ part)

The confidence-count likelihood is “as implemented in metadpy’s current HMeta-d / hierarchical meta-d′ code”.

Data input per subject s:
- nR_S1[s, :] and nR_S2[s, :] : vectors of length 2*K (K confidence levels)
- nR_S1 and nR_S2 follow the metadpy ordering convention

Likelihood:
- nR_S1[s] ~ Multinomial(N_S1[s], p_S1[s])
- nR_S2[s] ~ Multinomial(N_S2[s], p_S2[s])

Where p_S1[s], p_S2[s] are computed from meta-d′ and type-2 criteria according to metadpy’s existing implementation.

**Critical coupling rule:**
- meta-d′_s must be linked to d′_s and M_s via:
  - Mratio_s = exp(M_s)
  - meta-d′_s = d′_s * Mratio_s
  (unless metadpy uses a different internal parameterization; if so, document it precisely.)

---

## 4) API contract (v1)

### 4.1 Proposed function
Public entry point:
- `metadpy.bayesian.rhmetad(nR_S1, nR_S2, X, *, ...) -> RHMetaDResult`

Required behavior:
- Accept one or multiple predictors (P ≥ 1).
- Include intercept M0 internally (do not require user to add column of ones unless explicitly documented).
- Ensure covariate rows align to subject index ordering in nR_S1/nR_S2. Must error loudly if mismatch is detected.

### 4.2 Return object must include
- posterior trace (PyMC InferenceData or equivalent)
- posterior summaries for:
  - M0, β, σ_δ, ζ
  - subject-level M_s and derived Mratio_s
- basic PPC or sampling diagnostics fields (at least one)

---

## 5) Validation requirements (must pass)

### 5.1 Parameter recovery (β)
Simulation-based test:
- Simulate S subjects with known β_true (and M0_true).
- Generate confidence data using the same meta-d′ generative logic as metadpy.
- Fit RHMeta-d.
- Assert posterior for β concentrates near β_true:
  - e.g., posterior mean within tolerance AND/or HDI contains β_true.

### 5.2 Posterior predictive sanity check
At least one PPC-like check:
- Generate replicated counts from posterior predictive and compare simple summaries
  (e.g., overall type-2 response proportions) to observed.

---

## 6) UNVERIFIED items (must be confirmed by reading MATLAB/JAGS or existing metadpy code)
List anything that must be confirmed precisely:
- [UNVERIFIED] exact ordering convention for nR_S1/nR_S2 in metadpy vs MATLAB HMeta-d
- [UNVERIFIED] exact priors/parameterization for type-2 criteria and any constraints
- [UNVERIFIED] whether metadpy estimates d′_s or treats it fixed/derived (document exact mechanism)

This section must be resolved (or explicitly justified) before calling the implementation “done”.
