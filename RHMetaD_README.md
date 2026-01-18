# RHMetaD Regression (metadpy fork) - Lab Tutorial

This guide is for running the RHMetaD regression model from the lab fork:
https://github.com/heyifei1984/metadpy.git

The tutorial uses simulated data and runs in under a minute on CPU.

## Install from the lab fork

Pin to a tag or commit SHA for reproducibility:

```bash
pip install "git+https://github.com/heyifei1984/metadpy.git@<TAG_OR_SHA>"
```

Editable install for development:

```bash
git clone https://github.com/heyifei1984/metadpy.git
cd metadpy
pip install -e .
```

## Minimal end-to-end tutorial (simulated data)

```python
import numpy as np
import pandas as pd

from metadpy.bayesian import rhmetad
from metadpy.utils import trialSimulation

np.random.seed(123)

n_subjects = 3
n_trials = 120
n_ratings = 4
x = np.linspace(-1.0, 1.0, n_subjects)

beta = 0.6
mu_logmratio = -0.1
d1 = 1.2

frames = []
for idx, x_val in enumerate(x):
    meta_d = np.exp(mu_logmratio + beta * x_val) * d1
    df = trialSimulation(
        d=d1,
        metad=meta_d,
        c=0.0,
        nRatings=n_ratings,
        nTrials=n_trials,
    )
    df["Subject"] = idx
    df["Covariate"] = x_val
    frames.append(df)

data = pd.concat(frames, ignore_index=True)

model, idata = rhmetad(
    data=data,
    subject="Subject",
    stimuli="Stimuli",
    accuracy="Accuracy",
    confidence="Confidence",
    nRatings=n_ratings,
    covariates="Covariate",
    draws=50,
    tune=50,
    chains=1,
    random_seed=123,
    target_accept=0.9,
)

beta_mean = (
    idata.posterior["beta"].mean(dim=("chain", "draw")).to_numpy().squeeze()
)
mratio_mean = idata.posterior["Mratio"].mean(dim=("chain", "draw")).to_numpy()

print(f"beta mean: {beta_mean:.3f}")
print(f"Mratio mean (first subject): {mratio_mean[0]:.3f}")
```

## Alternative input mode: precomputed counts + X

The covariate matrix can be passed as either (subjects, covariates) or
Matlab-style (covariates, subjects). If `X` is square, it is treated as
(subjects x covariates) and is not transposed.

```python
import numpy as np
from metadpy.bayesian import rhmetad

nR_S1 = np.array([[52, 32, 35, 37, 26, 12, 4, 2]])
nR_S2 = np.array([[2, 5, 15, 22, 33, 38, 40, 45]])

X_sp = np.array([[0.2]])  # (subjects, covariates) preferred
model1, idata1 = rhmetad(nR_S1=nR_S1, nR_S2=nR_S2, nRatings=4, X=X_sp)

X_ps = X_sp.T  # (covariates, subjects) accepted and transposed
model2, idata2 = rhmetad(nR_S1=nR_S1, nR_S2=nR_S2, nRatings=4, X=X_ps)
```

## Optional acceleration (experimental)

You can pass PyTensor compile options via `compile_mode` or `compile_kwargs`.
For example, `compile_mode="MLX"` may be supported depending on your PyTensor
version and local setup. Default remains CPU with no platform auto-detection.

## Scripted smoke run

```bash
python examples/smoke_rhmetad.py
```

If PyTensor complains about compiledir permissions, set a temporary location:

```bash
PYTENSOR_FLAGS="compiledir=/tmp/pytensor" python examples/smoke_rhmetad.py
```
