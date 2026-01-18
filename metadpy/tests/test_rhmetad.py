# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import numpy as np
import pymc as pm
import pytest
from scipy.stats import norm

from metadpy.bayesian import rhmetad


def _clamp_probs(p, tol=1e-5):
    p = np.clip(p, tol, None)
    return p / p.sum()


def _simulate_rhmetad_counts(
    rng,
    n_ratings,
    x,
    beta,
    mu_logmratio,
    d1,
    c1,
    c_s1,
    c_s2,
    n_cr,
    n_fa,
    n_m,
    n_h,
):
    n_subj = x.shape[0]
    nR_S1 = np.zeros((n_subj, 2 * n_ratings), dtype=int)
    nR_S2 = np.zeros((n_subj, 2 * n_ratings), dtype=int)

    for idx in range(n_subj):
        log_mratio = mu_logmratio + x[idx, 0] * beta
        meta_d = np.exp(log_mratio) * d1
        s1_mu = -meta_d / 2.0
        s2_mu = meta_d / 2.0

        phi_c1_s1 = norm.cdf(c1 - s1_mu)
        phi_c1_s2 = norm.cdf(c1 - s2_mu)
        phi_cs1_s1 = norm.cdf(c_s1 - s1_mu)
        phi_cs1_s2 = norm.cdf(c_s1 - s2_mu)
        phi_cs2_s1 = norm.cdf(c_s2 - s1_mu)
        phi_cs2_s2 = norm.cdf(c_s2 - s2_mu)

        c_area_rs1 = phi_c1_s1
        i_area_rs2 = 1.0 - phi_c1_s1
        i_area_rs1 = phi_c1_s2
        c_area_rs2 = 1.0 - phi_c1_s2

        p_cr = np.concatenate(
            (
                phi_cs1_s1[:1] / c_area_rs1,
                (phi_cs1_s1[1:] - phi_cs1_s1[:-1]) / c_area_rs1,
                np.array([(phi_c1_s1 - phi_cs1_s1[-1]) / c_area_rs1]),
            )
        )
        p_fa = np.concatenate(
            (
                np.array([(phi_cs2_s1[0] - phi_c1_s1) / i_area_rs2]),
                (phi_cs2_s1[1:] - phi_cs2_s1[:-1]) / i_area_rs2,
                np.array([(1.0 - phi_cs2_s1[-1]) / i_area_rs2]),
            )
        )
        p_m = np.concatenate(
            (
                phi_cs1_s2[:1] / i_area_rs1,
                (phi_cs1_s2[1:] - phi_cs1_s2[:-1]) / i_area_rs1,
                np.array([(phi_c1_s2 - phi_cs1_s2[-1]) / i_area_rs1]),
            )
        )
        p_h = np.concatenate(
            (
                np.array([(phi_cs2_s2[0] - phi_c1_s2) / c_area_rs2]),
                (phi_cs2_s2[1:] - phi_cs2_s2[:-1]) / c_area_rs2,
                np.array([(1.0 - phi_cs2_s2[-1]) / c_area_rs2]),
            )
        )

        p_cr = _clamp_probs(p_cr)
        p_fa = _clamp_probs(p_fa)
        p_m = _clamp_probs(p_m)
        p_h = _clamp_probs(p_h)

        cr_counts = rng.multinomial(n_cr, p_cr)
        fa_counts = rng.multinomial(n_fa, p_fa)
        m_counts = rng.multinomial(n_m, p_m)
        h_counts = rng.multinomial(n_h, p_h)

        nR_S1[idx, :n_ratings] = cr_counts
        nR_S1[idx, n_ratings:] = fa_counts
        nR_S2[idx, :n_ratings] = m_counts
        nR_S2[idx, n_ratings:] = h_counts

    return nR_S1, nR_S2


@pytest.fixture(scope="module")
def rhmetad_fit():
    rng = np.random.default_rng(123)
    n_subj = 8
    n_ratings = 4
    x = np.linspace(-1.5, 1.5, n_subj)[:, None]
    beta = 0.9
    mu_logmratio = 0.0
    d1 = 1.5
    c1 = 0.0
    c_s1 = np.array([-1.5, -0.5, -0.1])
    c_s2 = np.array([0.1, 0.5, 1.5])

    nR_S1, nR_S2 = _simulate_rhmetad_counts(
        rng=rng,
        n_ratings=n_ratings,
        x=x,
        beta=beta,
        mu_logmratio=mu_logmratio,
        d1=d1,
        c1=c1,
        c_s1=c_s1,
        c_s2=c_s2,
        n_cr=60,
        n_fa=40,
        n_m=40,
        n_h=60,
    )

    model, idata = rhmetad(
        nR_S1=nR_S1,
        nR_S2=nR_S2,
        nRatings=n_ratings,
        X=x,
        draws=100,
        tune=100,
        chains=1,
        random_seed=123,
        target_accept=0.9,
        progressbar=False,
        cores=1,
        compute_convergence_checks=False,
    )

    return model, idata, nR_S1, nR_S2, n_ratings


def test_rhmetad_invalid_nratings():
    nR_S1 = np.array([[10, 9, 8, 7, 6, 5, 4, 3]])
    nR_S2 = np.array([[3, 4, 5, 6, 7, 8, 9, 10]])
    x = np.array([[0.1]])
    with pytest.raises(ValueError):
        rhmetad(
            nR_S1=nR_S1,
            nR_S2=nR_S2,
            nRatings=1,
            X=x,
            sample_model=False,
        )


def test_rhmetad_mismatched_subjects():
    nR_S1 = np.array(
        [
            [10, 9, 8, 7, 6, 5, 4, 3],
            [11, 10, 9, 8, 7, 6, 5, 4],
        ]
    )
    nR_S2 = np.array(
        [
            [3, 4, 5, 6, 7, 8, 9, 10],
            [4, 5, 6, 7, 8, 9, 10, 11],
        ]
    )
    x = np.array([[0.1], [0.2], [0.3]])
    with pytest.raises(ValueError):
        rhmetad(
            nR_S1=nR_S1,
            nR_S2=nR_S2,
            nRatings=4,
            X=x,
            sample_model=False,
        )


def test_rhmetad_parameter_recovery(rhmetad_fit):
    _, idata, *_ = rhmetad_fit
    beta_mean = (
        idata.posterior["beta"].mean(dim=("chain", "draw")).to_numpy().squeeze()
    )
    assert beta_mean > 0.1


def test_rhmetad_ppc_totals(rhmetad_fit):
    model, idata, nR_S1, nR_S2, n_ratings = rhmetad_fit
    with model:
        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=["CR_counts", "FA_counts", "M_counts", "H_counts"],
            random_seed=123,
            progressbar=False,
            return_inferencedata=False,
        )

    cr_obs = nR_S1[:, :n_ratings]
    fa_obs = nR_S1[:, n_ratings:]
    m_obs = nR_S2[:, :n_ratings]
    h_obs = nR_S2[:, n_ratings:]

    cr_totals = cr_obs.sum(axis=1)
    fa_totals = fa_obs.sum(axis=1)
    m_totals = m_obs.sum(axis=1)
    h_totals = h_obs.sum(axis=1)

    assert np.all(ppc["CR_counts"].sum(axis=-1) == cr_totals)
    assert np.all(ppc["FA_counts"].sum(axis=-1) == fa_totals)
    assert np.all(ppc["M_counts"].sum(axis=-1) == m_totals)
    assert np.all(ppc["H_counts"].sum(axis=-1) == h_totals)
