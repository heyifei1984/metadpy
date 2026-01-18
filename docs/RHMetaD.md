# RHMetaD.md

## What this document is
This is the **model specification** for implementing **RHMetaD** (Regression HMeta-d) in Python, intended to mirror the MATLAB **HMeta-d regression (nodp)** implementation.

The goal is to implement a group-level hierarchical Bayesian regression of **log metacognitive efficiency**:

- **Mratio** = meta-d′ / d′  
- Regression is on **log(Mratio)** using continuous covariates (trait measures, etc.).

## Reference implementation (MATLAB)
These are the canonical reference files in the MATLAB toolbox:

```text
HMeta-d/Matlab/fit_meta_d_mcmc_regression.m
HMeta-d/Matlab/Bayes_metad_group_regress_nodp.txt
HMeta-d/Matlab/Bayes_metad_group_regress_nodp_2cov.txt
...
HMeta-d/Matlab/Bayes_metad_group_regress_nodp_5cov.txt
HMeta-d/Matlab/trials2counts.m
```

If you need direct links, the upstream toolbox is hosted on GitHub (repo name: `HMeta-d`).  
(Links are provided in a code block at the end of this document.)

---

## 1) Data and ordering conventions

### 1.1 Counts vectors per subject
For a subject with **K = nRatings** discrete confidence levels:

- `nR_S1` is a length `2K` vector: counts when **stimulus S1** was presented.
- `nR_S2` is a length `2K` vector: counts when **stimulus S2** was presented.

Ordering (matches MATLAB `trials2counts.m`):
- First **K** entries: responded “S1”, with confidence from **high → low** (K, K-1, …, 1).
- Last **K** entries: responded “S2”, with confidence from **low → high** (1, 2, …, K).

So for K=3, `nR_S1 = [S1@3, S1@2, S1@1, S2@1, S2@2, S2@3]`.

### 1.2 Combined counts table
Define the concatenated vector:

- `counts = [nR_S1, nR_S2]` which has length `4K`.

From `counts`, define the four type-2 subsets (each length K):
- `CR_counts = counts[0:K]`
- `FA_counts = counts[K:2K]`
- `M_counts  = counts[2K:3K]`
- `H_counts  = counts[3K:4K]`

And their totals:
- `CR = sum(CR_counts)`
- `FA = sum(FA_counts)`
- `M  = sum(M_counts)`
- `H  = sum(H_counts)`

---

## 2) Preprocessing: fixed type-1 d′ and criterion (nodp)

MATLAB regression uses the **nodp** approach:
- compute type-1 `d1` and `c1` from the (padded) rating data,
- then treat them as **fixed** data in the hierarchical model.

### 2.1 Padding for stable type-1 estimates
Let `K = nRatings`.

Define:
- `adj_f = 1 / (2K)`
- `nR_S1_adj = nR_S1 + adj_f`
- `nR_S2_adj = nR_S2 + adj_f`

### 2.2 Rating-wise HR and FAR
For each criterion index `c` from `2` to `2K` (MATLAB loop corresponds to python slice indices):

- `ratingHR[c-1]  = sum(nR_S2_adj[c:]) / sum(nR_S2_adj)`
- `ratingFAR[c-1] = sum(nR_S1_adj[c:]) / sum(nR_S1_adj)`

Type-1 decision boundary index is:
- `t1_index = K`  (the boundary between “respond S1” vs “respond S2”)

So:
- `HR = ratingHR[t1_index]`
- `FAR = ratingFAR[t1_index]`

### 2.3 Type-1 d′ and criterion
Let `z(p) = Φ^{-1}(p)` where Φ is the standard normal CDF.

- `d1 = z(HR) - z(FAR)`
- `c1 = -0.5 * ( z(HR) + z(FAR) )`

These `d1` and `c1` are treated as **known constants** for each subject s.

---

## 3) Model: regression on log(Mratio)

Let there be S subjects and P covariates.

### 3.1 Covariates
- MATLAB expects `cov` shaped `(P, S)` (covariates × subjects).
- In Python, prefer `X` shaped `(S, P)` but accept `(P, S)` and transpose.

### 3.2 Priors / hyperpriors (match MATLAB regression scripts)
Priors are written here in standard notation; MATLAB/JAGS uses precisions.

**Criteria hyperpriors**
- `mu_c2 ~ Normal(0, 10)`  (JAGS: `dnorm(0, 0.01)`)
- `sigma_c2 ~ HalfNormal(10)` (JAGS: `dnorm(0,0.01) I(0,)`)

**Regression / Mratio hyperpriors**
- `mu_logMratio ~ Normal(0, 1)`
- `beta[p] ~ Normal(0, 1)` independently for p=1..P

**Robust subject deviation**
- `sigma_delta ~ HalfNormal(1)`
- `zeta = epsilon_logMratio ~ Beta(1, 1)` (Uniform(0,1))

Per subject:
- `delta[s] ~ StudentT(nu=5, mu=0, sigma=sigma_delta)`
- `logMratio[s] = mu_logMratio + X[s]·beta + zeta * delta[s]`
- `Mratio[s] = exp(logMratio[s])`

A useful derived quantity (matches MATLAB output):
- `sigma_logMratio = zeta * sigma_delta`  
  (JAGS writes `abs(epsilon_logMratio)*sigma_delta`, but `zeta>=0`.)

### 3.3 Meta-d′ from Mratio
In nodp regression, type-1 `d1[s]` is fixed, and:

- `meta_d[s] = Mratio[s] * d1[s]`

Define SDT means for the type-2 evidence axis:
- `S1mu[s] = -meta_d[s] / 2`
- `S2mu[s] = +meta_d[s] / 2`

---

## 4) Type-2 criteria priors (as in MATLAB regression scripts)

For each subject s and each rating boundary j = 1..(K-1):

- `cS1_raw[s,j] ~ Normal(-mu_c2, sigma_c2)` truncated to `(-∞, c1[s])`
- `cS2_raw[s,j] ~ Normal(+mu_c2, sigma_c2)` truncated to `(c1[s], +∞)`

Then enforce ordering (MATLAB/JAGS does this by sorting):
- `cS1[s,:] = sort(cS1_raw[s,:])`  (ascending)
- `cS2[s,:] = sort(cS2_raw[s,:])`  (ascending)

> Implementation note (PyMC): sorting is non-differentiable.  
> A NUTS-friendly equivalent is to parameterize **positive increments** away from `c1[s]` to ensure ordering.  
> As long as the resulting criteria satisfy the same constraints (monotone, on correct side of c1), the likelihood matches.

---

## 5) Likelihood: multinomials for CR / FA / M / H

Define a small tolerance:
- `Tol = 1e-5`

Define normalization constants:
- `C_area_rS1 = Φ(c1 - S1mu)`
- `I_area_rS2 = 1 - Φ(c1 - S1mu)`
- `I_area_rS1 = Φ(c1 - S2mu)`
- `C_area_rS2 = 1 - Φ(c1 - S2mu)`

### 5.1 CR probabilities (stimulus S1, responded S1; K bins)
Let `cS1[j]` be the ordered criteria (`j=1..K-1`).

- `p_CR[1] = Φ(cS1[1] - S1mu) / C_area_rS1`
- for k=1..K-2:  
  `p_CR[k+1] = [Φ(cS1[k+1] - S1mu) - Φ(cS1[k] - S1mu)] / C_area_rS1`
- `p_CR[K] = [Φ(c1 - S1mu) - Φ(cS1[K-1] - S1mu)] / C_area_rS1`

### 5.2 FA probabilities (stimulus S1, responded S2; K bins)
Let `cS2[j]` be ordered (`j=1..K-1`).

- `p_FA[1] = [Φ(cS2[1] - S1mu) - Φ(c1 - S1mu)] / I_area_rS2`
- for k=1..K-2:  
  `p_FA[k+1] = [Φ(cS2[k+1] - S1mu) - Φ(cS2[k] - S1mu)] / I_area_rS2`
- `p_FA[K] = [1 - Φ(cS2[K-1] - S1mu)] / I_area_rS2`

### 5.3 M probabilities (stimulus S2, responded S1; K bins)
- `p_M[1] = Φ(cS1[1] - S2mu) / I_area_rS1`
- for k=1..K-2:  
  `p_M[k+1] = [Φ(cS1[k+1] - S2mu) - Φ(cS1[k] - S2mu)] / I_area_rS1`
- `p_M[K] = [Φ(c1 - S2mu) - Φ(cS1[K-1] - S2mu)] / I_area_rS1`

### 5.4 H probabilities (stimulus S2, responded S2; K bins)
- `p_H[1] = [Φ(cS2[1] - S2mu) - Φ(c1 - S2mu)] / C_area_rS2`
- for k=1..K-2:  
  `p_H[k+1] = [Φ(cS2[k+1] - S2mu) - Φ(cS2[k] - S2mu)] / C_area_rS2`
- `p_H[K] = [1 - Φ(cS2[K-1] - S2mu)] / C_area_rS2`

### 5.5 Tol clamping
For numerical stability:
- `p_* = max(p_*, Tol)` elementwise  
(and renormalize if required by the chosen Multinomial implementation).

### 5.6 Likelihood statements (per subject)
- `CR_counts ~ Multinomial(n=CR, p=p_CR)`
- `FA_counts ~ Multinomial(n=FA, p=p_FA)`
- `M_counts  ~ Multinomial(n=M,  p=p_M)`
- `H_counts  ~ Multinomial(n=H,  p=p_H)`

This matches the MATLAB regression JAGS scripts exactly.

---

## 6) What the regression coefficients mean (plain language)

- `Mratio` is **metacognitive efficiency**: how informative a person’s confidence is about being correct, *relative* to their basic task sensitivity (d′).
- The model works on `log(Mratio)` so that:
  - regression is linear and unconstrained,
  - but Mratio itself stays positive (`exp(...)`).

If you have one covariate `x` and coefficient `β`:
- `log(Mratio) = intercept + β·x + noise`
- Therefore, **a 1-unit increase in x multiplies Mratio by exp(β)**.

So:
- `β > 0` → higher trait score predicts **better metacognitive efficiency**
- `β < 0` → higher trait score predicts **worse metacognitive efficiency**

The Student-t noise + epsilon scaling is a “robust” choice:
- it tolerates occasional outlier subjects without forcing β to absorb them.

---

## 7) Apple Silicon acceleration hook (PyMC / PyTensor)

PyMC supports a `compile_kwargs` argument in `pm.sample(...)` that is passed to compiled functions used by the step methods.  
This can be used to request alternative PyTensor compilation modes, including **MLX** (Apple Silicon acceleration) when available.

Implementation requirement:
- expose `compile_mode` (string) OR `compile_kwargs` (dict) in the RHMetaD API and pass through to `pm.sample`.

---

## Appendix: Reference links (GitHub)
(Placed in a code block to comply with “no raw URLs in prose”.)

```text
HMeta-d repository:
https://github.com/metacoglab/HMeta-d

Key MATLAB regression files:
https://github.com/metacoglab/HMeta-d/blob/master/Matlab/fit_meta_d_mcmc_regression.m
https://github.com/metacoglab/HMeta-d/blob/master/Matlab/Bayes_metad_group_regress_nodp.txt
https://github.com/metacoglab/HMeta-d/blob/master/Matlab/Bayes_metad_group_regress_nodp_2cov.txt
https://github.com/metacoglab/HMeta-d/blob/master/Matlab/trials2counts.m
```

