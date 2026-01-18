# RHMetaD.md — MATLAB/JAGS parity spec for RHMeta-d regression (metadpy fork)

This document is a **precise implementation contract** for adding the HMeta-d hierarchical regression model to metadpy.

The v1 target is parity with the MATLAB/JAGS regression path:
- `Matlab/fit_meta_d_mcmc_regression.m`
- `Matlab/Bayes_metad_group_regress_nodp*.txt`

---

## Sources (authoritative references)

Primary (official toolbox):
- HMeta-d GitHub repo (official): metacoglab/HMeta-d

Primary (paper):
- Fleming, S. M. (2017). *HMeta-d: hierarchical Bayesian estimation of metacognitive efficiency from confidence ratings.* Neuroscience of Consciousness.

Apple Silicon / JAX Metal:
- Apple Developer: “Accelerated JAX on Mac — Metal plug-in”
- PyMC docs: `pymc.sampling.jax.sample_numpyro_nuts`
- PyMC issue tracker: Apple Silicon Metal sampling is experimental (may fail)

(Keep URLs in this file if you want; do not rely on memory.)

---

## 0) Glossary and shapes

Let:
- `S` = number of subjects
- `K` = number of confidence ratings per response (e.g., K=3 means 3 confidence levels)
- Each subject has `2*K` response categories per stimulus (S1 or S2 stimulus)

Data:
- `nR_S1[s, :]` shape `(S, 2*K)` (or list length S of vectors length 2*K)
- `nR_S2[s, :]` shape `(S, 2*K)`
- `cov` / `X` subject covariates:
  - MATLAB expects `cov` shape `(P, S)` (P covariates × S subjects)
  - Python API may accept `(S, P)` but MUST align to subjects deterministically.

---

## 1) Exact response-count ordering (must match HMeta-d)

Per subject, `nR_S1` and `nR_S2` encode counts conditional on stimulus class (S1 vs S2).

Ordering (from MATLAB docs and `trials2counts.m`):
- First K entries: responded “S1”, confidence from **high → low**
- Last K entries: responded “S2”, confidence from **low → high**

Example for K=3:
- index 1: responded S1, rating=3 (high)
- index 2: responded S1, rating=2
- index 3: responded S1, rating=1 (low)
- index 4: responded S2, rating=1 (low)
- index 5: responded S2, rating=2
- index 6: responded S2, rating=3 (high)

Construction from trial-level (MATLAB `trials2counts.m`):
- For S1-responses: loop rating r = K..1
- For S2-responses: loop rating r = 1..K

---

## 2) “nodp” preprocessing for type-1 parameters (d′ and criterion)

In v1 regression parity, the model uses **fixed** `d1[s]` and `c1[s]` computed from the rating counts, matching MATLAB (`fit_meta_d_mcmc_regression.m` and `fit_meta_d_mcmc_group.m`):

Given subject counts `nR_S1` and `nR_S2` (length 2*K):
1) Add a small adjustment ONLY for computing the point estimate:
   - `adj_f = 1 / (2*K)`
   - `nR_S1_adj = nR_S1 + adj_f`
   - `nR_S2_adj = nR_S2 + adj_f`

2) Build cumulative HR and FAR over criteria c = 2..(2*K):
   - `ratingHR[c-1]  = sum(nR_S2_adj[c:]) / sum(nR_S2_adj)`
   - `ratingFAR[c-1] = sum(nR_S1_adj[c:]) / sum(nR_S1_adj)`

3) Use the criterion at `t1_index = K` (i.e., the boundary between S1 and S2 responses):
   - `d1[s] = Φ^{-1}(ratingHR[K]) - Φ^{-1}(ratingFAR[K])`
   - `c1[s] = -0.5 * ( Φ^{-1}(ratingHR[K]) + Φ^{-1}(ratingFAR[K]) )`

Where Φ^{-1} is the standard normal inverse CDF.

Important:
- The padded counts are ONLY for the point estimate.
- The Bayesian likelihood uses the original counts.

---

## 3) Count matrix used by the JAGS model

JAGS regression model uses a `counts` matrix per subject created as:
- `counts[s, :] = [nR_S1[s, :] , nR_S2[s, :]]`

So `counts[s, :]` has length `4*K`.

Define totals per subject:
- `CR[s] = sum(counts[s, 1:K])`                      (S1 stim, responded S1)
- `FA[s] = sum(counts[s, K+1:2K])`                   (S1 stim, responded S2)
- `M[s]  = sum(counts[s, 2K+1:3K])`                  (S2 stim, responded S1)
- `H[s]  = sum(counts[s, 3K+1:4K])`                  (S2 stim, responded S2)

The likelihood is four multinomials:
- `counts_CR ~ Multinomial(CR, p_CR)`
- `counts_FA ~ Multinomial(FA, p_FA)`
- `counts_M  ~ Multinomial(M,  p_M)`
- `counts_H  ~ Multinomial(H,  p_H)`

---

## 4) Core equal-variance SDT meta-d′ construction

### 4.1 Mratio and meta-d′ coupling
Per subject:
- `logMratio[s] = log(meta_d′[s] / d1[s])`
- `Mratio[s] = exp(logMratio[s])`
- `mu[s] = Mratio[s] * d1[s]`
- `S2mu[s] =  mu[s] / 2`
- `S1mu[s] = -mu[s] / 2`

Equal variance SDT (sd=1 on both S1 and S2 distributions).

### 4.2 Normalization constants
Using Φ as standard normal CDF:
- `C_area_rS1[s] = Φ(c1[s] - S1mu[s])`
- `I_area_rS1[s] = Φ(c1[s] - S2mu[s])`
- `C_area_rS2[s] = 1 - Φ(c1[s] - S2mu[s])`
- `I_area_rS2[s] = 1 - Φ(c1[s] - S1mu[s])`

---

## 5) Type-2 criteria and their priors (must match JAGS)

For each subject s, there are `K-1` type-2 criteria on each response side:
- `cS1[s, 1:(K-1)]`  (criteria for response = S1)
- `cS2[s, 1:(K-1)]`  (criteria for response = S2)

### 5.1 Priors (JAGS)
Hyperpriors:
- `mu_c2 ~ Normal(0, sd=10)`        (JAGS precision 0.01)
- `sigma_c2 ~ HalfNormal(sd=10)`    (JAGS Normal(0,10) truncated >0)
- `lambda_c2 = sigma_c2^{-2}`

Subject criteria (raw, then ordered):
- for j=1..(K-1):
  - `cS1_raw[s,j] ~ Normal(-mu_c2, sd=sigma_c2) truncated (-inf, c1[s])`
  - `cS2_raw[s,j] ~ Normal( mu_c2, sd=sigma_c2) truncated (c1[s], +inf)`
- then enforce ordering:
  - `cS1[s,:] = sort(cS1_raw[s,:])`
  - `cS2[s,:] = sort(cS2_raw[s,:])`

**PyMC implementation note:** sorting is not differentiable; implement ordering via an ordered transform (equivalent up to a constant factor in the prior density). Preserve the effective truncated support relative to `c1[s]`.

---

## 6) Probability construction for the multinomials (JAGS equations)

Let `Tol = 1e-5` (as in MATLAB datastruct).

Define helper:
- `Φ(x; μ) = Φ(x - μ)` where Φ is standard normal CDF and SD=1.

### 6.1 Block 1: counts[s, 1:K] (CR block; S1 stim, responded S1)
- `p_CR[1] = Φ(cS1[1] - S1mu) / C_area_rS1`
- for k=1..(K-2):
  - `p_CR[k+1] = (Φ(cS1[k+1] - S1mu) - Φ(cS1[k] - S1mu)) / C_area_rS1`
- `p_CR[K] = (Φ(c1 - S1mu) - Φ(cS1[K-1] - S1mu)) / C_area_rS1`

### 6.2 Block 2: counts[s, K+1:2K] (FA block; S1 stim, responded S2)
- `p_FA[1] = ((1-Φ(c1 - S1mu)) - (1-Φ(cS2[1] - S1mu))) / I_area_rS2`
- for k=1..(K-2):
  - `p_FA[k+1] = ((1-Φ(cS2[k] - S1mu)) - (1-Φ(cS2[k+1] - S1mu))) / I_area_rS2`
- `p_FA[K] = (1-Φ(cS2[K-1] - S1mu)) / I_area_rS2`

### 6.3 Block 3: counts[s, 2K+1:3K] (M block; S2 stim, responded S1)
Same as CR but using S2mu and I_area_rS1:
- `p_M[1] = Φ(cS1[1] - S2mu) / I_area_rS1`
- for k=1..(K-2):
  - `p_M[k+1] = (Φ(cS1[k+1] - S2mu) - Φ(cS1[k] - S2mu)) / I_area_rS1`
- `p_M[K] = (Φ(c1 - S2mu) - Φ(cS1[K-1] - S2mu)) / I_area_rS1`

### 6.4 Block 4: counts[s, 3K+1:4K] (H block; S2 stim, responded S2)
Same as FA but using S2mu and C_area_rS2:
- `p_H[1] = ((1-Φ(c1 - S2mu)) - (1-Φ(cS2[1] - S2mu))) / C_area_rS2`
- for k=1..(K-2):
  - `p_H[k+1] = ((1-Φ(cS2[k] - S2mu)) - (1-Φ(cS2[k+1] - S2mu))) / C_area_rS2`
- `p_H[K] = (1-Φ(cS2[K-1] - S2mu)) / C_area_rS2`

### 6.5 Underflow handling
JAGS applies elementwise flooring:
- `pT = ifelse(p < Tol, Tol, p)`

**PyMC adaptation requirement:** If needed for `Multinomial`, floor then renormalize within each block so probabilities sum to 1. Document this explicitly if done.

---

## 7) Regression model on logMratio (exact JAGS logic)

Per subject s:
- `delta[s] ~ StudentT(df=5, loc=0, scale=sigma_delta)`
- `logMratio[s] = mu_logMratio + Σ_j mu_beta[j] * cov[j,s] + epsilon_logMratio * delta[s]`
- `Mratio[s] = exp(logMratio[s])`

Hyperpriors (JAGS):
- `mu_logMratio ~ Normal(0, sd=1)`
- for each covariate j:
  - `mu_beta[j] ~ Normal(0, sd=1)`
- `sigma_delta ~ HalfNormal(sd=1)`
- `epsilon_logMratio ~ Beta(1,1)`
- `sigma_logMratio = abs(epsilon_logMratio) * sigma_delta`

---

## 8) API contract (v1)

### 8.1 Proposed public entry point
`metadpy.bayesian.rhmetad(...)`

It should mirror `hmetad` conventions where possible:
- accept either:
  - direct `nR_S1`, `nR_S2` arrays, OR
  - trial-level DataFrame inputs (stimuli/accuracy/confidence + subject id)
- accept covariates in a way that **cannot silently reorder subjects** (e.g., aligned by subject id)

Must support:
- `backend="pymc"` (default) and `backend="numpyro"` (JAX)

Suggested signature sketch (adapt to repo norms):
- `rhmetad(data=None, nR_S1=None, nR_S2=None, covariates=None, subject=None, nRatings=None, ..., backend="pymc", sample_model=True, output="model", **kwargs)`

### 8.2 Required outputs
At minimum, return the same style outputs as `hmetad`:
- if `output="model"`: `(model, idata)`
- if `output="dataframe"`: per-subject summary including:
  - `d` (d1), `c` (c1), `meta_d`, `m_ratio`
  - regression coefficient posterior summaries (β)

---

## 9) Validation requirements (must pass)

### 9.1 β recovery
Simulate data *consistent with this model* (logMratio regression + SDT likelihood), fit rhmetad, and confirm β posterior concentrates near truth.

### 9.2 PPC / sanity
At least one posterior predictive check:
- generate replicated counts from posterior predictive
- compare simple summaries (e.g., marginal proportions per block) to observed

---

## 10) Apple Silicon / Metal note (documentation requirement)
- `backend="pymc"`: CPU sampling; on ARM64, PyTensor may use Apple Accelerate for BLAS (performance).
- `backend="numpyro"`: uses JAX-based NUTS; if user installs Apple’s `jax-metal`, JAX may execute on Metal GPU, but upstream support is experimental and may fail.

Document clearly; do not promise speedups.
