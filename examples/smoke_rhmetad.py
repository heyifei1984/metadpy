import numpy as np
import pandas as pd

from metadpy.bayesian import rhmetad
from metadpy.utils import trialSimulation


def main():
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
        progressbar=False,
        cores=1,
    )

    beta_mean = (
        idata.posterior["beta"].mean(dim=("chain", "draw")).to_numpy().squeeze()
    )
    mr_mean = idata.posterior["Mratio"].mean(dim=("chain", "draw")).to_numpy()

    print(f"beta mean: {beta_mean:.3f}")
    print(f"Mratio mean (first subject): {mr_mean[0]:.3f}")


if __name__ == "__main__":
    main()
