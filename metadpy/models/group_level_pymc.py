# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Optional

import pytensor.tensor as pt
import pymc as pm


def phi(x):
    """Cumulative normal distribution."""
    return 0.5 + 0.5 * pt.erf(x / pt.sqrt(2))


def _clamp_probs(p, tol):
    p = pt.maximum(p, tol)
    return p / p.sum(axis=-1, keepdims=True)


def hmetad_groupLevel(
    data,
    sample_model=True,
    num_samples: int = 1000,
    num_chains: int = 4,
    draws: Optional[int] = None,
    tune: Optional[int] = None,
    chains: Optional[int] = None,
    target_accept: Optional[float] = None,
    random_seed: Optional[int] = None,
    **kwargs,
):
    """Hierarchical Bayesian meta-d' model (group level)."""
    n_ratings = data["nRatings"]
    n_subj = data["nSubj"]
    tol = float(data["Tol"])

    with pm.Model() as model:
        mu_d1 = pm.Normal("mu_d1", mu=0.0, sigma=1.0)
        sigma_d1 = pm.HalfNormal("sigma_d1", sigma=1.0)
        d1 = pm.Normal("d1", mu=mu_d1, sigma=sigma_d1, shape=n_subj)

        mu_c1 = pm.Normal("mu_c1", mu=0.0, sigma=1.0)
        sigma_c1 = pm.HalfNormal("sigma_c1", sigma=1.0)
        c1 = pm.Normal("c1", mu=mu_c1, sigma=sigma_c1, shape=n_subj)

        mu_meta_d = pm.Normal("mu_meta_d", mu=0.0, sigma=1.0)
        sigma_meta_d = pm.HalfNormal("sigma_meta_d", sigma=1.0)
        meta_d = pm.Normal("meta_d", mu=mu_meta_d, sigma=sigma_meta_d, shape=n_subj)

        # TYPE 1 SDT BINOMIAL MODEL
        h = phi(d1 / 2 - c1)
        f = phi(-d1 / 2 - c1)
        pm.Binomial("H", n=data["S"], p=h, observed=data["H"])
        pm.Binomial("FA", n=data["N"], p=f, observed=data["FA"])

        # Specify ordered prior on criteria bounded above and below by Type 1 c1
        cS1_hn = pm.HalfNormal("cS1_hn", sigma=1.0, shape=(n_subj, n_ratings - 1))
        cS1 = pm.Deterministic(
            "cS1", pt.sort(-cS1_hn, axis=1) + (c1[:, None] - tol)
        )

        cS2_hn = pm.HalfNormal("cS2_hn", sigma=1.0, shape=(n_subj, n_ratings - 1))
        cS2 = pm.Deterministic(
            "cS2", pt.sort(cS2_hn, axis=1) + (c1[:, None] - tol)
        )

        # Means of SDT distributions
        S1mu = -meta_d / 2.0
        S2mu = meta_d / 2.0

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
            if draws is None:
                draws = num_samples
            if chains is None:
                chains = num_chains

            sample_kwargs = dict(kwargs)
            if tune is not None:
                sample_kwargs["tune"] = tune
            if target_accept is not None:
                sample_kwargs["target_accept"] = target_accept
            if random_seed is not None:
                sample_kwargs["random_seed"] = random_seed

            trace = pm.sample(
                return_inferencedata=True,
                chains=chains,
                draws=draws,
                **sample_kwargs,
            )
            return model, trace

        return model


def hmetad_rm1way(
    data,
    sample_model=True,
    num_samples: int = 1000,
    num_chains: int = 4,
    draws: Optional[int] = None,
    tune: Optional[int] = None,
    chains: Optional[int] = None,
    target_accept: Optional[float] = None,
    random_seed: Optional[int] = None,
    **kwargs,
):
    """Hierarchical Bayesian meta-d' model with within-subject conditions."""
    n_ratings = data["nRatings"]
    n_subj = data["nSubj"]
    n_cond = data["nCond"]
    tol = float(data["Tol"])

    with pm.Model() as model:
        mu_d1 = pm.Normal("mu_d1", mu=0.0, sigma=1.0)
        sigma_d1 = pm.HalfNormal("sigma_d1", sigma=1.0)
        d1 = pm.Normal(
            "d1", mu=mu_d1, sigma=sigma_d1, shape=(n_subj, n_cond)
        )

        mu_c1 = pm.Normal("mu_c1", mu=0.0, sigma=1.0)
        sigma_c1 = pm.HalfNormal("sigma_c1", sigma=1.0)
        c1 = pm.Normal(
            "c1", mu=mu_c1, sigma=sigma_c1, shape=(n_subj, n_cond)
        )

        mu_meta_d = pm.Normal("mu_meta_d", mu=0.0, sigma=1.0)
        sigma_meta_d = pm.HalfNormal("sigma_meta_d", sigma=1.0)
        condition_offset_raw = pm.Normal("condition_offset_raw", mu=0.0, sigma=1.0, shape=n_cond)
        condition_offset = pm.Deterministic(
            "condition_offset", condition_offset_raw - pt.mean(condition_offset_raw)
        )
        meta_d = pm.Normal(
            "meta_d",
            mu=mu_meta_d + condition_offset,
            sigma=sigma_meta_d,
            shape=(n_subj, n_cond),
        )

        # TYPE 1 SDT BINOMIAL MODEL
        h = phi(d1 / 2 - c1)
        f = phi(-d1 / 2 - c1)
        pm.Binomial("H", n=data["S"], p=h, observed=data["H"])
        pm.Binomial("FA", n=data["N"], p=f, observed=data["FA"])

        # Specify ordered prior on criteria bounded above and below by Type 1 c1
        cS1_hn = pm.HalfNormal(
            "cS1_hn", sigma=1.0, shape=(n_subj, n_cond, n_ratings - 1)
        )
        cS1 = pm.Deterministic(
            "cS1", pt.sort(-cS1_hn, axis=2) + (c1[:, :, None] - tol)
        )

        cS2_hn = pm.HalfNormal(
            "cS2_hn", sigma=1.0, shape=(n_subj, n_cond, n_ratings - 1)
        )
        cS2 = pm.Deterministic(
            "cS2", pt.sort(cS2_hn, axis=2) + (c1[:, :, None] - tol)
        )

        # Means of SDT distributions
        S1mu = -meta_d / 2.0
        S2mu = meta_d / 2.0

        C_area_rS1 = phi(c1 - S1mu)
        I_area_rS2 = 1.0 - phi(c1 - S1mu)
        I_area_rS1 = phi(c1 - S2mu)
        C_area_rS2 = 1.0 - phi(c1 - S2mu)

        phi_c1_S1 = phi(c1 - S1mu)
        phi_c1_S2 = phi(c1 - S2mu)
        phi_cS1_S1 = phi(cS1 - S1mu[:, :, None])
        phi_cS1_S2 = phi(cS1 - S2mu[:, :, None])
        phi_cS2_S1 = phi(cS2 - S1mu[:, :, None])
        phi_cS2_S2 = phi(cS2 - S2mu[:, :, None])

        p_CR = pt.concatenate(
            (
                phi_cS1_S1[:, :, :1] / C_area_rS1[:, :, None],
                (phi_cS1_S1[:, :, 1:] - phi_cS1_S1[:, :, :-1])
                / C_area_rS1[:, :, None],
                ((phi_c1_S1 - phi_cS1_S1[:, :, -1]) / C_area_rS1)[:, :, None],
            ),
            axis=2,
        )
        p_FA = pt.concatenate(
            (
                ((phi_cS2_S1[:, :, :1] - phi_c1_S1[:, :, None]) / I_area_rS2[:, :, None]),
                (phi_cS2_S1[:, :, 1:] - phi_cS2_S1[:, :, :-1])
                / I_area_rS2[:, :, None],
                ((1.0 - phi_cS2_S1[:, :, -1]) / I_area_rS2)[:, :, None],
            ),
            axis=2,
        )
        p_M = pt.concatenate(
            (
                phi_cS1_S2[:, :, :1] / I_area_rS1[:, :, None],
                (phi_cS1_S2[:, :, 1:] - phi_cS1_S2[:, :, :-1])
                / I_area_rS1[:, :, None],
                ((phi_c1_S2 - phi_cS1_S2[:, :, -1]) / I_area_rS1)[:, :, None],
            ),
            axis=2,
        )
        p_H = pt.concatenate(
            (
                ((phi_cS2_S2[:, :, :1] - phi_c1_S2[:, :, None]) / C_area_rS2[:, :, None]),
                (phi_cS2_S2[:, :, 1:] - phi_cS2_S2[:, :, :-1])
                / C_area_rS2[:, :, None],
                ((1.0 - phi_cS2_S2[:, :, -1]) / C_area_rS2)[:, :, None],
            ),
            axis=2,
        )

        p_CR = _clamp_probs(p_CR, tol)
        p_FA = _clamp_probs(p_FA, tol)
        p_M = _clamp_probs(p_M, tol)
        p_H = _clamp_probs(p_H, tol)

        pm.Multinomial(
            "CR_counts",
            n=data["CR"],
            p=p_CR,
            shape=(n_subj, n_cond, n_ratings),
            observed=data["CR_counts"],
        )
        pm.Multinomial(
            "FA_counts",
            n=data["FA"],
            p=p_FA,
            shape=(n_subj, n_cond, n_ratings),
            observed=data["FA_counts"],
        )
        pm.Multinomial(
            "M_counts",
            n=data["M"],
            p=p_M,
            shape=(n_subj, n_cond, n_ratings),
            observed=data["M_counts"],
        )
        pm.Multinomial(
            "H_counts",
            n=data["H"],
            p=p_H,
            shape=(n_subj, n_cond, n_ratings),
            observed=data["H_counts"],
        )

        if sample_model is True:
            if draws is None:
                draws = num_samples
            if chains is None:
                chains = num_chains

            sample_kwargs = dict(kwargs)
            if tune is not None:
                sample_kwargs["tune"] = tune
            if target_accept is not None:
                sample_kwargs["target_accept"] = target_accept
            if random_seed is not None:
                sample_kwargs["random_seed"] = random_seed

            trace = pm.sample(
                return_inferencedata=True,
                chains=chains,
                draws=draws,
                **sample_kwargs,
            )
            return model, trace

        return model
