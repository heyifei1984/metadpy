import numpy as np
import pandas as pd

from metadpy.bayesian import rhmetad


def _counts_to_trials(nr_s1, nr_s2, nratings, subject, covariates):
    rows = []

    def add_trials(stimulus, response, rating, count):
        accuracy = 1 if response == stimulus else 0
        for _ in range(int(count)):
            rows.append(
                {
                    "subject": subject,
                    "Stimuli": stimulus,
                    "Accuracy": accuracy,
                    "Confidence": rating,
                    "cov1": float(covariates[0]),
                    "cov2": float(covariates[1]),
                }
            )

    for idx in range(nratings):
        rating = nratings - idx
        add_trials(0, 0, rating, nr_s1[idx])
    for idx in range(nratings):
        rating = idx + 1
        add_trials(0, 1, rating, nr_s1[nratings + idx])
    for idx in range(nratings):
        rating = nratings - idx
        add_trials(1, 0, rating, nr_s2[idx])
    for idx in range(nratings):
        rating = idx + 1
        add_trials(1, 1, rating, nr_s2[nratings + idx])

    return rows


def main():
    nratings = 4
    nr_s1 = np.array(
        [
            [20, 15, 10, 5, 3, 2, 1, 1],
            [18, 14, 9, 4, 4, 3, 2, 1],
            [22, 16, 11, 6, 2, 2, 1, 1],
        ],
        dtype=int,
    )
    nr_s2 = np.array(
        [
            [2, 3, 5, 10, 12, 14, 16, 18],
            [3, 4, 6, 9, 11, 13, 15, 17],
            [1, 2, 4, 9, 12, 15, 17, 19],
        ],
        dtype=int,
    )
    x = np.array([[-1.0, 0.5], [0.0, -0.5], [1.0, 1.0]], dtype=float)

    model_counts, idata_counts = rhmetad(
        nR_S1=nr_s1,
        nR_S2=nr_s2,
        nRatings=nratings,
        X=x,
        draws=30,
        tune=30,
        chains=1,
        target_accept=0.9,
        random_seed=123,
    )
    print("precomputed posterior draws:", idata_counts.posterior.sizes["draw"])

    subjects = ["s1", "s2", "s3"]
    rows = []
    for idx, subj in enumerate(subjects):
        rows.extend(_counts_to_trials(nr_s1[idx], nr_s2[idx], nratings, subj, x[idx]))

    df = pd.DataFrame(rows)
    rhmetad(
        data=df,
        subject="subject",
        stimuli="Stimuli",
        accuracy="Accuracy",
        confidence="Confidence",
        nRatings=nratings,
        covariates=["cov1", "cov2"],
        draws=10,
        tune=10,
        chains=1,
        random_seed=123,
        sample_model=False,
    )
    print("dataframe input model built (set sample_model=True to sample).")

if __name__ == "__main__":
    main()
