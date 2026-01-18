# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import pytensor.tensor as pt
import pymc as pm


def phi(x):
    """Cumulative normal distribution."""
    return 0.5 + 0.5 * pt.erf(x / pt.sqrt(2))


def _clamp_probs(p, tol):
    p = pt.maximum(p, tol)
    return p / p.sum(axis=1, keepdims=True)


def rhmetad_groupLevel(
    data,
    X,
    sample_model=True,
    num_samples: int = 1000,
    num_chains: int = 4,
    **kwargs,
):
    """Hierarchical Bayesian regression of log Mratio (group level)."""
    n_ratings = data["nRatings"]
    n_subj = data["nSubj"]

    if X.ndim != 2:
        raise ValueError("Covariate matrix must be 2D (subjects x covariates).")

    n_cov = X.shape[1]
    X = pt.as_tensor_variable(X)
    d1 = pt.as_tensor_variable(data["d1"])
    c1 = pt.as_tensor_variable(data["c1"])
    tol = float(data["Tol"])

    with pm.Model() as model:
        mu_c2 = pm.Normal("mu_c2", mu=0.0, sigma=10.0)
        sigma_c2 = pm.HalfNormal("sigma_c2", sigma=10.0)

        mu_logMratio = pm.Normal("mu_logMratio", mu=0.0, sigma=1.0)
        beta = pm.Normal("beta", mu=0.0, sigma=1.0, shape=n_cov)

        sigma_delta = pm.HalfNormal("sigma_delta", sigma=1.0)
        epsilon_logMratio = pm.Beta("epsilon_logMratio", alpha=1.0, beta=1.0)
        delta = pm.StudentT(
            "delta",
            nu=5.0,
            mu=0.0,
            sigma=sigma_delta,
            shape=n_subj,
        )

        logMratio = pm.Deterministic(
            "logMratio",
            mu_logMratio + pt.dot(X, beta) + epsilon_logMratio * delta,
        )
        Mratio = pm.Deterministic("Mratio", pt.exp(logMratio))
        sigma_logMratio = pm.Deterministic(
            "sigma_logMratio", epsilon_logMratio * sigma_delta
        )
        meta_d = pm.Deterministic("meta_d", Mratio * d1)

        S1mu = -meta_d / 2.0
        S2mu = meta_d / 2.0

        c1_col = c1[:, None]
        cS1_raw = pm.TruncatedNormal(
            "cS1_raw",
            mu=-mu_c2,
            sigma=sigma_c2,
            upper=c1_col - tol,
            shape=(n_subj, n_ratings - 1),
        )
        cS2_raw = pm.TruncatedNormal(
            "cS2_raw",
            mu=mu_c2,
            sigma=sigma_c2,
            lower=c1_col + tol,
            shape=(n_subj, n_ratings - 1),
        )

        cS1 = pm.Deterministic("cS1", pt.sort(cS1_raw, axis=1))
        cS2 = pm.Deterministic("cS2", pt.sort(cS2_raw, axis=1))

        C_area_rS1 = phi(c1 - S1mu)
        I_area_rS2 = 1.0 - phi(c1 - S1mu)
        I_area_rS1 = phi(c1 - S2mu)
        C_area_rS2 = 1.0 - phi(c1 - S2mu)

        phi_c1_S1 = phi(c1 - S1mu)
        phi_c1_S2 = phi(c1 - S2mu)
        phi_cS1_S1 = phi(cS1 - S1mu[:, None])
        phi_cS1_S2 = phi(cS1 - S2mu[:, None])
        phi_cS2_S1 = phi(cS2 - S1mu[:, None])
        phi_cS2_S2 = phi(cS2 - S2mu[:, None])

        p_CR = pt.concatenate(
            (
                phi_cS1_S1[:, :1] / C_area_rS1[:, None],
                (phi_cS1_S1[:, 1:] - phi_cS1_S1[:, :-1]) / C_area_rS1[:, None],
                ((phi_c1_S1 - phi_cS1_S1[:, -1]) / C_area_rS1)[:, None],
            ),
            axis=1,
        )
        p_FA = pt.concatenate(
            (
                ((phi_cS2_S1[:, :1] - phi_c1_S1[:, None]) / I_area_rS2[:, None]),
                (phi_cS2_S1[:, 1:] - phi_cS2_S1[:, :-1]) / I_area_rS2[:, None],
                ((1.0 - phi_cS2_S1[:, -1]) / I_area_rS2)[:, None],
            ),
            axis=1,
        )
        p_M = pt.concatenate(
            (
                phi_cS1_S2[:, :1] / I_area_rS1[:, None],
                (phi_cS1_S2[:, 1:] - phi_cS1_S2[:, :-1]) / I_area_rS1[:, None],
                ((phi_c1_S2 - phi_cS1_S2[:, -1]) / I_area_rS1)[:, None],
            ),
            axis=1,
        )
        p_H = pt.concatenate(
            (
                ((phi_cS2_S2[:, :1] - phi_c1_S2[:, None]) / C_area_rS2[:, None]),
                (phi_cS2_S2[:, 1:] - phi_cS2_S2[:, :-1]) / C_area_rS2[:, None],
                ((1.0 - phi_cS2_S2[:, -1]) / C_area_rS2)[:, None],
            ),
            axis=1,
        )

        p_CR = _clamp_probs(p_CR, tol)
        p_FA = _clamp_probs(p_FA, tol)
        p_M = _clamp_probs(p_M, tol)
        p_H = _clamp_probs(p_H, tol)

        pm.Multinomial(
            "CR_counts",
            n=data["CR"],
            p=p_CR,
            shape=(n_subj, n_ratings),
            observed=data["CR_counts"],
        )
        pm.Multinomial(
            "FA_counts",
            n=data["FA"],
            p=p_FA,
            shape=(n_subj, n_ratings),
            observed=data["FA_counts"],
        )
        pm.Multinomial(
            "M_counts",
            n=data["M"],
            p=p_M,
            shape=(n_subj, n_ratings),
            observed=data["M_counts"],
        )
        pm.Multinomial(
            "H_counts",
            n=data["H"],
            p=p_H,
            shape=(n_subj, n_ratings),
            observed=data["H_counts"],
        )

        if sample_model is True:
            trace = pm.sample(
                return_inferencedata=True,
                chains=num_chains,
                draws=num_samples,
                **kwargs,
            )
            return model, trace

        return model
