# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import os
import sys
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union, overload

import arviz as az
import numpy as np
import pandas as pd
import pandas_flavor as pf
from arviz import InferenceData

from metadpy.sdt import criterion, dprime
from metadpy.utils import discreteRatings, trials2counts

if TYPE_CHECKING is True:
    from pymc.backends.base import MultiTrace
    from pymc.model import Model


@overload
def hmetad(
    data: None,
    nR_S1: Union[List, np.ndarray],
    nR_S2: Union[List, np.ndarray],
    nRatings: Optional[int],
    subject: None,
    within: None,
    nbins: int,
    padding: bool = False,
    padAmount: Optional[float] = None,
    output: str = "model",
) -> "Tuple[Union[Model, Callable], Optional[Union[InferenceData, MultiTrace]]]":
    ...


@overload
def hmetad(
    data: pd.DataFrame,
    stimuli: str,
    accuracy: str,
    confidence: str,
    nRatings: Optional[int],
    subject: None,
    within: None,
    nbins: int,
    padding: bool = False,
    padAmount: Optional[float] = None,
    output: str = "model",
) -> "Tuple[Union[Model, Callable], Optional[Union[InferenceData, MultiTrace]]]":
    ...


@overload
def hmetad(
    data: pd.DataFrame,
    stimuli: str,
    accuracy: str,
    confidence: str,
    nRatings: Optional[int],
    subject: str,
    within: None,
    nbins: int,
    padding: bool = False,
    padAmount: Optional[float] = None,
    output: str = "model",
) -> "Tuple[Union[Model, Callable], Optional[Union[InferenceData, MultiTrace]]]":
    ...


@overload
def hmetad(
    data: pd.DataFrame,
    stimuli: str,
    accuracy: str,
    confidence: str,
    nRatings: Optional[int],
    subject: str,
    within: str,
    nbins: int,
    padding: bool = False,
    padAmount: Optional[float] = None,
    sample_model: bool = True,
    output: str = "model",
) -> "Tuple[Union[Model, Callable], Optional[Union[InferenceData, MultiTrace]]]":
    ...


@pf.register_dataframe_method
def hmetad(
    data=None,
    nR_S1=None,
    nR_S2=None,
    stimuli="Stimuli",
    accuracy="Accuracy",
    confidence="Confidence",
    nRatings=None,
    within=None,
    between=None,
    subject=None,
    nbins=4,
    padding=False,
    padAmount=None,
    sample_model=True,
    output: str = "model",
    num_samples: int = 1000,
    num_chains: int = 4,
    **kwargs
):
    """Bayesian meta-d' model with hyperparametes at the group level.

    Parameters
    ----------
    data :
        Dataframe. Note that this function can also directly be used as a Pandas
        method, in which case this argument is no longer needed.
    nR_S1 :
        Confience ratings (stimuli 1, correct and incorrect).
    nR_S2 :
        Confience ratings (stimuli 2, correct and incorrect).
    stimuli :
        Name of the column containing the stimuli.
    accuracy :
        Name of the columns containing the accuracy.
    confidence :
        Name of the column containing the confidence ratings.
    nRatings :
        Number of discrete ratings. If a continuous rating scale was used, and
        the number of unique ratings does not match `nRatings`, will convert to
        discrete ratings using :py:func:`metadpy.utils.discreteRatings`.
    within :
        Name of column containing the within factor (condition comparison).
    between :
        Name of column containing the between subject factor (group
        comparison).
    subject :
        Name of column containing the subject identifier (only required if a
        within-subject or a between-subject factor is provided).
    nbins :
        If a continuous rating scale was using, `nbins` define the number of
        discrete ratings when converting using
        :py:func:`metadpy.utils.discreteRatings`. The default value is `4`.
    padding :
        If `True`, each response count in the output has the value of padAmount
        added to it. Padding cells is desirable if trial counts of 0 interfere
        with model fitting. If False, trial counts are not manipulated and 0s
        may be present in the response count output. Default value for padding
        is `False`.
    padAmount :
        The value to add to each response count if padCells is set to 1.
        Default value is 1/(2*nRatings)
    sample_model :
        If `False`, only the model is returned without sampling.
    output :
        The kind of outpute expected. If `"model"`, will return the model function and
        the traces. If `"dataframe"`, will return a dataframe containing `d` (dprime),
        `c` (criterion), `meta_d` (the meta-d prime) and `m_ratio` (`meta_d/d`).
    num_samples :
        The number of samples per chains to draw (defaults to `1000`).
    num_chains :
        The number of chains (defaults to `4`).
    **kwargs : keyword arguments
        All keyword arguments are passed to `func::pymc.sampling.sample`.

    Returns
    -------
    If `output="model"`:

    model :
        The model PyMC as a :py:class:`pymc.Model`.
    traces :
        A `MultiTrace` or `ArviZ InferenceData` object that contains the samples. Only
        returned if `sample_model` is set to `True`, otherwise set to None.

    or

    results :
        If `output="dataframe"`, :py:class:`pandas.DataFrame` containing the values for
        the following variables:

        * d-prime (d)
        * criterion (c)
        * meta-d' (meta_d)
        * m-ratio (m_ratio)

    Examples
    --------
    1. Subject-level

    Notes
    -----
    This function will compute hierarchical Bayesian estimation of metacognitive
    efficiency as described in [1]_. The model can be fitter at the subject level, at
    the group level and can account for repeated measures by providing the corresponding
    `subject`, `between` and `within` factors.

    If the confidence levels have more unique values than `nRatings`, the confience
    column will be discretized using py:func:`metadpy.utils.discreteRatings`.

    Raises
    ------
    ValueError
        When the number of ratings is not provided.
        If data is None and nR_S1 or nR_S2 not provided.
        If the backend is not `"numpyro"` or `"pymc"`.

    References
    ----------
    .. [1] Fleming, S.M. (2017) HMeta-d: hierarchical Bayesian estimation of
      metacognitive efficiency from confidence ratings, Neuroscience of
      Consciousness, 3(1) nix007, https://doi.org/10.1093/nc/nix007.

    """
    modelScript = os.path.dirname(__file__) + "/models/"
    sys.path.append(modelScript)

    if (nR_S1 is not None) & (nR_S2 is not None):
        nR_S1, nR_S2 = np.asarray(nR_S1), np.asarray(nR_S2)
        if nRatings is not None:
            assert len(nR_S1) / 2 == nRatings
        else:
            nRatings = len(nR_S1) / 2

    if nRatings is None:
        raise ValueError("You should provide the number of ratings")

    if data is None:
        if (nR_S1 is None) or (nR_S2 is None):
            raise ValueError(
                "If data is None, you should provide"
                " the nR_S1 and nR_S2 vectors instead."
            )
    else:
        if data[confidence].nunique() > nRatings:

            # If a continuous rating scale was used (if N unique ratings > nRatings)
            # transform confidence to discrete ratings
            print(
                (
                    "The confidence columns contains more unique values than nRatings. "
                    "The ratings are going to be discretized using "
                    "metadpy.utils.discreteRatings()"
                )
            )
            new_ratings, _ = discreteRatings(data[confidence].to_numpy(), nbins=nbins)
            data.loc[:, confidence] = new_ratings

    ###############
    # Subject level
    if (within is None) & (between is None) & (subject is None):

        if data is not None:
            nR_S1, nR_S2 = trials2counts(
                data=data,
                stimuli=stimuli,
                accuracy=accuracy,
                confidence=confidence,
                nRatings=nRatings,
                padding=padding,
                padAmount=padAmount,
            )

        pymcData = extractParameters(np.asarray(nR_S1), np.asarray(nR_S2))

        from subject_level_pymc import hmetad_subjectLevel

        model_output = hmetad_subjectLevel(
            pymcData,
            sample_model=sample_model,
            num_chains=num_chains,
            num_samples=num_samples,
        )
        supports_dataframe_output = True

    #############
    # Group level
    elif (within is None) & (between is None) & (subject is not None):

        pymcData = preprocess_hmetad_group(
            data=data,
            nR_S1=nR_S1,
            nR_S2=nR_S2,
            subject=subject,
            stimuli=stimuli,
            accuracy=accuracy,
            confidence=confidence,
            nRatings=nRatings,
            nbins=nbins,
            padding=padding,
            padAmount=padAmount,
        )

        from group_level_pymc import hmetad_groupLevel

        model_output = hmetad_groupLevel(
            pymcData,
            sample_model=sample_model,
            num_chains=num_chains,
            num_samples=num_samples,
        )
        supports_dataframe_output = False

    ###################
    # Repeated-measures
    elif (within is not None) & (between is None) & (subject is not None):

        pymcData = preprocess_rm1way(
            data=data,
            subject=subject,
            within=within,
            stimuli=stimuli,
            accuracy=accuracy,
            confidence=confidence,
            nRatings=nRatings,
            nbins=nbins,
            padding=padding,
            padAmount=padAmount,
        )

        from group_level_pymc import hmetad_rm1way

        model_output = hmetad_rm1way(
            pymcData,
            sample_model=sample_model,
            num_chains=num_chains,
            num_samples=num_samples,
        )
        supports_dataframe_output = False

    ##########
    # Sampling
    if sample_model is True:
        model, traces = model_output

        if output == "model":
            return model, traces
        elif output == "dataframe":
            if supports_dataframe_output is False:
                raise ValueError("output='dataframe' is only supported for subject-level.")
            return pd.DataFrame(
                {
                    "d": [pymcData["d1"]],
                    "c": [pymcData["c1"]],
                    "meta_d": [
                        az.summary(traces, var_names=["meta_d"])["mean"]["meta_d"]
                    ],
                    "m_ratio": [
                        az.summary(traces, var_names=["meta_d"])["mean"]["meta_d"]
                        / pymcData["d1"]
                    ],
                }
            )
    else:
        return model_output, None


@pf.register_dataframe_method
def rhmetad(
    data=None,
    nR_S1=None,
    nR_S2=None,
    stimuli="Stimuli",
    accuracy="Accuracy",
    confidence="Confidence",
    nRatings=None,
    subject=None,
    covariates=None,
    X=None,
    nbins=4,
    padding=False,
    padAmount=None,
    zscore_covariates=False,
    drop_na=False,
    backend: str = "pymc",
    sample_model=True,
    num_samples: int = 1000,
    num_chains: int = 4,
    draws: Optional[int] = None,
    tune: Optional[int] = None,
    chains: Optional[int] = None,
    target_accept: Optional[float] = None,
    random_seed: Optional[int] = None,
    compile_mode: Optional[str] = None,
    compile_kwargs: Optional[Dict] = None,
    **kwargs,
):
    """Bayesian RHMeta-d regression model (group level, nodp).

    Parameters
    ----------
    data :
        Dataframe. Note that this function can also directly be used as a Pandas
        method, in which case this argument is no longer needed.
    nR_S1 :
        Confience ratings (stimuli 1, correct and incorrect).
    nR_S2 :
        Confience ratings (stimuli 2, correct and incorrect).
    stimuli :
        Name of the column containing the stimuli.
    accuracy :
        Name of the columns containing the accuracy.
    confidence :
        Name of the column containing the confidence ratings.
    nRatings :
        Number of discrete ratings. If a continuous rating scale was used, and
        the number of unique ratings does not match `nRatings`, will convert to
        discrete ratings using :py:func:`metadpy.utils.discreteRatings`.
    subject :
        Name of column containing the subject identifier.
    covariates :
        Column name or list of columns containing subject-level covariates.
    X :
        Covariate matrix shaped (subjects x covariates).
    nbins :
        Number of discrete ratings when converting continuous confidence to
        discrete ratings.
    padding :
        If `True`, each response count in the output has the value of padAmount
        added to it.
    padAmount :
        The value to add to each response count if padding is set to 1.
    zscore_covariates :
        If `True`, z-score covariates across subjects.
    drop_na :
        If `True`, drop subjects with non-finite covariates.
    backend :
        Backend to use, either `"pymc"` or `"numpyro"`.
    sample_model :
        If `False`, only the model is returned without sampling.
    num_samples :
        The number of samples per chains to draw (defaults to `1000`).
    num_chains :
        The number of chains (defaults to `4`).
    draws :
        Number of draws to sample. If `None`, uses `num_samples`.
    tune :
        Number of tuning steps. If `None`, PyMC defaults are used.
    chains :
        Number of chains to sample. If `None`, uses `num_chains`.
    target_accept :
        Target acceptance probability passed to `pm.sample`.
    random_seed :
        Random seed passed to `pm.sample`.
    compile_mode :
        Optional PyTensor compile mode.
    compile_kwargs :
        Full `compile_kwargs` dict passed to `pm.sample`.
    **kwargs : keyword arguments
        All keyword arguments are passed to `func::pymc.sampling.sample`.

    Returns
    -------
    model :
        The model PyMC as a :py:class:`pymc.Model`.
    traces :
        A `MultiTrace` or `ArviZ InferenceData` object that contains the samples. Only
        returned if `sample_model` is set to `True`, otherwise set to None.

    """
    if backend not in ["pymc", "numpyro"]:
        raise ValueError("Invalid backend provided - should be 'pymc' or 'numpyro'.")
    if backend != "pymc":
        raise ValueError("The numpyro backend is not implemented yet.")

    if compile_kwargs is not None and compile_mode is not None:
        raise ValueError("Provide compile_mode or compile_kwargs, not both.")
    if compile_kwargs is None and compile_mode is not None:
        compile_kwargs = {"mode": compile_mode}

    if "compile_kwargs" in kwargs:
        raise ValueError("compile_kwargs should be passed as a named argument.")

    modelScript = os.path.dirname(__file__) + "/models/"
    sys.path.append(modelScript)

    pymc_data, X = preprocess_group(
        data=data,
        nR_S1=nR_S1,
        nR_S2=nR_S2,
        stimuli=stimuli,
        accuracy=accuracy,
        confidence=confidence,
        nRatings=nRatings,
        subject=subject,
        covariates=covariates,
        X=X,
        nbins=nbins,
        padding=padding,
        padAmount=padAmount,
        zscore_covariates=zscore_covariates,
        drop_na=drop_na,
    )

    from group_level_regression_pymc import rhmetad_groupLevel

    if compile_kwargs is not None:
        kwargs = {**kwargs, "compile_kwargs": compile_kwargs}

    model_output = rhmetad_groupLevel(
        pymc_data,
        X,
        sample_model=sample_model,
        num_chains=num_chains,
        num_samples=num_samples,
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
        random_seed=random_seed,
        **kwargs,
    )

    if sample_model is True:
        return model_output

    return model_output, None


def extractParameters(
    nR_S1: Union[List[int], np.ndarray], nR_S2: Union[List[int], np.ndarray]
) -> Dict:
    """Extract rates and task parameters.

    Parameters
    ----------
    nR_S1, nR_S2 :
        Total number of responses in each response category, conditional on
        presentation of S1 and S2. e.g. if `nR_S1 = [100 50 20 10 5 1]`, then
        when stimulus S1 was presented, the subject had the following response
        counts:
        * responded S1, rating=3 : 100 times
        * responded S1, rating=2 : 50 times
        * responded S1, rating=1 : 20 times
        * responded S2, rating=1 : 10 times
        * responded S2, rating=2 : 5 times
        * responded S2, rating=3 : 1 time

    Return
    ------
    data :
        Dictionary of rates and task parameters.

    See Also
    --------
    hmetad

    """
    if isinstance(nR_S1, list):
        nR_S1 = np.array(nR_S1, dtype=float)
    if isinstance(nR_S2, list):
        nR_S2 = np.array(nR_S2, dtype=float)

    Tol = 1e-05
    nratings = int(len(nR_S1) / 2)

    # Adjust to ensure non-zero counts for type 1 d' point estimate
    adj_f = 1 / ((nratings) * 2)

    nR_S1_adj = nR_S1 + adj_f
    nR_S2_adj = nR_S2 + adj_f

    ratingHR: List[float] = []
    ratingFAR: List[float] = []
    for c in range(1, int(nratings * 2)):
        ratingHR.append(sum(nR_S2_adj[c:]) / sum(nR_S2_adj))
        ratingFAR.append(sum(nR_S1_adj[c:]) / sum(nR_S1_adj))

    d1 = dprime(
        data=None,
        stimuli=None,
        responses=None,
        hit_rate=ratingHR[nratings - 1],
        fa_rate=ratingFAR[nratings - 1],
    )
    c1 = criterion(
        data=None,
        hit_rate=ratingHR[nratings - 1],
        fa_rate=ratingFAR[nratings - 1],
        stimuli=None,
        responses=None,
    )
    counts = np.hstack([nR_S1, nR_S2])

    # Type 1 counts
    N = sum(counts[: (nratings * 2)])
    S = sum(counts[(nratings * 2) : (nratings * 4)])
    H = sum(counts[(nratings * 3) : (nratings * 4)])
    M = sum(counts[(nratings * 2) : (nratings * 3)])
    FA = sum(counts[(nratings) : (nratings * 2)])
    CR = sum(counts[:(nratings)])

    # Data preparation for model
    data = {
        "d1": d1,
        "c1": c1,
        "counts": counts,
        "nratings": nratings,
        "Tol": Tol,
        "FA": FA,
        "CR": CR,
        "M": M,
        "H": H,
        "N": N,
        "S": S,
    }

    return data


def preprocess_group(
    data: Optional[pd.DataFrame],
    nR_S1: Optional[Union[List, np.ndarray]] = None,
    nR_S2: Optional[Union[List, np.ndarray]] = None,
    subject: Optional[str] = None,
    stimuli: str = "Stimuli",
    accuracy: str = "Accuracy",
    confidence: str = "Confidence",
    nRatings: Optional[int] = None,
    covariates: Optional[Union[str, List[str]]] = None,
    X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    nbins: int = 4,
    padding: bool = False,
    padAmount: Optional[float] = None,
    zscore_covariates: bool = False,
    drop_na: bool = False,
) -> Tuple[Dict, np.ndarray]:
    """Preprocess group data for RHMeta-d regression (nodp).

    Notes
    -----
    If ``X`` is square (``n_subj`` x ``n_subj``), it is treated as
    ``(subjects x covariates)`` and is not transposed.
    """

    if covariates is not None and X is not None:
        raise ValueError("Provide covariates or X, not both.")

    if data is None:
        if (X is None) and (covariates is None):
            raise ValueError("You should provide covariates or X.")
        if (nR_S1 is None) or (nR_S2 is None):
            raise ValueError(
                "If data is None, you should provide the nR_S1 and nR_S2 arrays instead."
            )
        nR_S1, nR_S2, nRatings = _coerce_group_counts(nR_S1, nR_S2, nRatings)
        subjects = np.arange(nR_S1.shape[0])
    else:
        if subject is None:
            raise ValueError("You should provide the subject column name.")
        if nRatings is None:
            raise ValueError("You should provide the number of ratings.")
        nRatings = _validate_nratings(nRatings)

        if data[confidence].nunique() > nRatings:
            print(
                (
                    "The confidence columns contains more unique values than nRatings. "
                    "The ratings are going to be discretized using "
                    "metadpy.utils.discreteRatings()"
                )
            )
            new_ratings, _ = discreteRatings(data[confidence].to_numpy(), nbins=nbins)
            data = data.copy()
            data.loc[:, confidence] = new_ratings

        subjects = pd.unique(data[subject])
        nR_S1_list = []
        nR_S2_list = []
        for sub in subjects:
            nR_S1_sub, nR_S2_sub = trials2counts(
                data=data[data[subject] == sub],
                stimuli=stimuli,
                accuracy=accuracy,
                confidence=confidence,
                nRatings=nRatings,
                padding=padding,
                padAmount=padAmount,
            )
            nR_S1_list.append(nR_S1_sub)
            nR_S2_list.append(nR_S2_sub)
        nR_S1 = np.asarray(nR_S1_list)
        nR_S2 = np.asarray(nR_S2_list)

    n_subj = nR_S1.shape[0]
    d1 = np.zeros(n_subj, dtype=float)
    c1 = np.zeros(n_subj, dtype=float)
    counts = np.zeros((n_subj, 4 * nRatings), dtype=float)

    for idx in range(n_subj):
        this_data = extractParameters(nR_S1[idx], nR_S2[idx])
        d1[idx] = this_data["d1"]
        c1[idx] = this_data["c1"]
        counts[idx] = this_data["counts"]

    CR_counts = counts[:, :nRatings]
    FA_counts = counts[:, nRatings : 2 * nRatings]
    M_counts = counts[:, 2 * nRatings : 3 * nRatings]
    H_counts = counts[:, 3 * nRatings : 4 * nRatings]
    CR = CR_counts.sum(axis=1)
    FA = FA_counts.sum(axis=1)
    M = M_counts.sum(axis=1)
    H = H_counts.sum(axis=1)

    if covariates is not None:
        if data is None:
            raise ValueError("Covariates require a dataframe input.")
        covariate_cols = [covariates] if isinstance(covariates, str) else list(covariates)
        if len(covariate_cols) == 0:
            raise ValueError("Covariates must include at least one column.")
        missing = [col for col in covariate_cols if col not in data.columns]
        if missing:
            raise ValueError(f"Covariate columns not found: {missing}.")
        grouped = data.groupby(subject, sort=False)[covariate_cols]
        if (grouped.nunique(dropna=False) > 1).any().any():
            raise ValueError("Covariates must be constant within subject.")
        cov_df = grouped.first().reindex(subjects)
        X = cov_df.to_numpy()
    elif X is not None:
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        else:
            X = np.asarray(X)
    else:
        raise ValueError("You should provide covariates or X.")

    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError("Covariate matrix must be 2D (subjects x covariates).")
    if X.shape[0] == n_subj:
        pass
    elif X.shape[1] == n_subj:
        X = X.T
    else:
        raise ValueError(
            "Covariate matrix must have shape (n_subj, P) or (P, n_subj) "
            f"with n_subj={n_subj}. Got {X.shape}."
        )

    finite_mask = np.isfinite(X).all(axis=1)
    if not finite_mask.all():
        if drop_na is True:
            X = X[finite_mask]
            d1 = d1[finite_mask]
            c1 = c1[finite_mask]
            counts = counts[finite_mask]
            CR_counts = CR_counts[finite_mask]
            FA_counts = FA_counts[finite_mask]
            M_counts = M_counts[finite_mask]
            H_counts = H_counts[finite_mask]
            CR = CR[finite_mask]
            FA = FA[finite_mask]
            M = M[finite_mask]
            H = H[finite_mask]
            subjects = subjects[finite_mask]
            n_subj = X.shape[0]
            if n_subj == 0:
                raise ValueError("All subjects were dropped due to non-finite covariates.")
        else:
            raise ValueError("Covariates contain non-finite values.")

    if zscore_covariates is True:
        means = X.mean(axis=0)
        stds = X.std(axis=0, ddof=0)
        if np.any(stds == 0):
            raise ValueError("Cannot z-score covariates with zero variance.")
        X = (X - means) / stds

    pymc_data = {
        "d1": d1,
        "c1": c1,
        "counts": counts,
        "CR_counts": CR_counts,
        "FA_counts": FA_counts,
        "M_counts": M_counts,
        "H_counts": H_counts,
        "CR": CR,
        "FA": FA,
        "M": M,
        "H": H,
        "nRatings": nRatings,
        "nSubj": n_subj,
        "Tol": 1e-05,
        "subjects": subjects,
    }

    return pymc_data, X


def preprocess_hmetad_group(
    data: Optional[pd.DataFrame],
    nR_S1: Optional[Union[List, np.ndarray]] = None,
    nR_S2: Optional[Union[List, np.ndarray]] = None,
    subject: Optional[str] = None,
    stimuli: str = "Stimuli",
    accuracy: str = "Accuracy",
    confidence: str = "Confidence",
    nRatings: Optional[int] = None,
    nbins: int = 4,
    padding: bool = False,
    padAmount: Optional[float] = None,
) -> Dict:
    """Preprocess group data for hierarchical Bayesian meta-d'."""

    if data is None:
        nR_S1, nR_S2, nRatings = _coerce_group_counts(nR_S1, nR_S2, nRatings)
        subjects = np.arange(nR_S1.shape[0])
    else:
        if subject is None:
            raise ValueError("You should provide the subject column name.")
        if nRatings is None:
            raise ValueError("You should provide the number of ratings.")
        nRatings = _validate_nratings(nRatings)

        if data[confidence].nunique() > nRatings:
            print(
                (
                    "The confidence columns contains more unique values than nRatings. "
                    "The ratings are going to be discretized using "
                    "metadpy.utils.discreteRatings()"
                )
            )
            new_ratings, _ = discreteRatings(data[confidence].to_numpy(), nbins=nbins)
            data = data.copy()
            data.loc[:, confidence] = new_ratings

        subjects = pd.unique(data[subject])
        nR_S1_list = []
        nR_S2_list = []
        for sub in subjects:
            nR_S1_sub, nR_S2_sub = trials2counts(
                data=data[data[subject] == sub],
                stimuli=stimuli,
                accuracy=accuracy,
                confidence=confidence,
                nRatings=nRatings,
                padding=padding,
                padAmount=padAmount,
            )
            nR_S1_list.append(nR_S1_sub)
            nR_S2_list.append(nR_S2_sub)
        nR_S1 = np.asarray(nR_S1_list)
        nR_S2 = np.asarray(nR_S2_list)

    n_subj = nR_S1.shape[0]
    d1 = np.zeros(n_subj, dtype=float)
    c1 = np.zeros(n_subj, dtype=float)
    counts = np.zeros((n_subj, 4 * nRatings), dtype=float)
    H = np.zeros(n_subj, dtype=float)
    FA = np.zeros(n_subj, dtype=float)
    S = np.zeros(n_subj, dtype=float)
    N = np.zeros(n_subj, dtype=float)

    for idx in range(n_subj):
        this_data = extractParameters(nR_S1[idx], nR_S2[idx])
        d1[idx] = this_data["d1"]
        c1[idx] = this_data["c1"]
        counts[idx] = this_data["counts"]
        H[idx] = this_data["H"]
        FA[idx] = this_data["FA"]
        S[idx] = this_data["S"]
        N[idx] = this_data["N"]

    CR_counts = counts[:, :nRatings]
    FA_counts = counts[:, nRatings : 2 * nRatings]
    M_counts = counts[:, 2 * nRatings : 3 * nRatings]
    H_counts = counts[:, 3 * nRatings : 4 * nRatings]
    CR = CR_counts.sum(axis=1)
    M = M_counts.sum(axis=1)

    pymc_data = {
        "d1": d1,
        "c1": c1,
        "counts": counts,
        "CR_counts": CR_counts,
        "FA_counts": FA_counts,
        "M_counts": M_counts,
        "H_counts": H_counts,
        "CR": CR,
        "FA": FA,
        "M": M,
        "H": H,
        "S": S,
        "N": N,
        "nRatings": nRatings,
        "nSubj": n_subj,
        "Tol": 1e-05,
        "subjects": subjects,
    }

    return pymc_data


def preprocess_rm1way(
    data: pd.DataFrame,
    subject: str,
    within: str,
    stimuli: str = "Stimuli",
    accuracy: str = "Accuracy",
    confidence: str = "Confidence",
    nRatings: Optional[int] = None,
    nbins: int = 4,
    padding: bool = False,
    padAmount: Optional[float] = None,
) -> Dict:
    """Preprocess repeated measures data."""
    if nRatings is None:
        raise ValueError("You should provide the number of ratings.")
    nRatings = _validate_nratings(nRatings)

    if data[confidence].nunique() > nRatings:
        print(
            (
                "The confidence columns contains more unique values than nRatings. "
                "The ratings are going to be discretized using "
                "metadpy.utils.discreteRatings()"
            )
        )
        new_ratings, _ = discreteRatings(data[confidence].to_numpy(), nbins=nbins)
        data = data.copy()
        data.loc[:, confidence] = new_ratings

    subjects = pd.unique(data[subject])
    conditions = pd.unique(data[within])
    n_subj = len(subjects)
    n_cond = len(conditions)

    counts = np.zeros((n_subj, n_cond, 4 * nRatings), dtype=float)
    d1 = np.zeros((n_subj, n_cond), dtype=float)
    c1 = np.zeros((n_subj, n_cond), dtype=float)
    H = np.zeros((n_subj, n_cond), dtype=float)
    FA = np.zeros((n_subj, n_cond), dtype=float)
    S = np.zeros((n_subj, n_cond), dtype=float)
    N = np.zeros((n_subj, n_cond), dtype=float)

    for sub_idx, sub in enumerate(subjects):
        for cond_idx, cond in enumerate(conditions):
            subset = data[(data[subject] == sub) & (data[within] == cond)]
            nR_S1_sub, nR_S2_sub = trials2counts(
                data=subset,
                stimuli=stimuli,
                accuracy=accuracy,
                confidence=confidence,
                nRatings=nRatings,
                padding=padding,
                padAmount=padAmount,
            )
            this_data = extractParameters(nR_S1_sub, nR_S2_sub)
            d1[sub_idx, cond_idx] = this_data["d1"]
            c1[sub_idx, cond_idx] = this_data["c1"]
            counts[sub_idx, cond_idx, :] = this_data["counts"]
            H[sub_idx, cond_idx] = this_data["H"]
            FA[sub_idx, cond_idx] = this_data["FA"]
            S[sub_idx, cond_idx] = this_data["S"]
            N[sub_idx, cond_idx] = this_data["N"]

    CR_counts = counts[:, :, :nRatings]
    FA_counts = counts[:, :, nRatings : 2 * nRatings]
    M_counts = counts[:, :, 2 * nRatings : 3 * nRatings]
    H_counts = counts[:, :, 3 * nRatings : 4 * nRatings]
    CR = CR_counts.sum(axis=2)
    M = M_counts.sum(axis=2)

    pymc_data = {
        "d1": d1,
        "c1": c1,
        "counts": counts,
        "CR_counts": CR_counts,
        "FA_counts": FA_counts,
        "M_counts": M_counts,
        "H_counts": H_counts,
        "CR": CR,
        "FA": FA,
        "M": M,
        "H": H,
        "S": S,
        "N": N,
        "nRatings": nRatings,
        "nSubj": n_subj,
        "nCond": n_cond,
        "Tol": 1e-05,
        "subjects": subjects,
        "conditions": conditions,
    }

    return pymc_data


def _validate_nratings(nRatings: Optional[int]) -> int:
    if nRatings is None:
        raise ValueError("You should provide the number of ratings.")
    try:
        nRatings_int = int(nRatings)
    except (TypeError, ValueError):
        raise ValueError("nRatings should be a positive integer.")
    if nRatings_int < 2 or nRatings_int != nRatings:
        raise ValueError("nRatings should be a positive integer.")
    return nRatings_int


def _coerce_group_counts(nR_S1, nR_S2, nRatings):
    if (nR_S1 is None) or (nR_S2 is None):
        raise ValueError(
            "If data is None, you should provide the nR_S1 and nR_S2 arrays instead."
        )
    nR_S1 = np.asarray(nR_S1)
    nR_S2 = np.asarray(nR_S2)
    if nR_S1.shape != nR_S2.shape:
        raise ValueError("nR_S1 and nR_S2 must have the same shape.")
    if nR_S1.ndim == 1:
        nR_S1 = nR_S1[None, :]
        nR_S2 = nR_S2[None, :]
    if nR_S1.ndim != 2:
        raise ValueError("nR_S1 and nR_S2 must be 1D or 2D arrays.")
    if nRatings is None:
        if nR_S1.shape[1] % 2 != 0:
            raise ValueError("nR_S1 and nR_S2 must have length 2 * nRatings.")
        nRatings = nR_S1.shape[1] // 2
    else:
        nRatings = _validate_nratings(nRatings)
        if nR_S1.shape[1] != 2 * nRatings:
            raise ValueError("nR_S1 and nR_S2 must have length 2 * nRatings.")
    if np.any(nR_S1 < 0) or np.any(nR_S2 < 0):
        raise ValueError("nR_S1 and nR_S2 must be non-negative.")
    return nR_S1, nR_S2, nRatings


# def preprocess_rm1way(
#     data: pd.DataFrame,
#     subject: str,
#     stimuli: str,
#     within: str,
#     accuracy: str,
#     confidence: str,
#     nRatings: int,
# ) -> Dict:
#     """Preprocess repeated measures data.

#     Parameters
#     ----------
#     data : :py:class:`pandas.DataFrame`
#         Dataframe. Note that this function can also directly be used as a
#         Pandas method, in which case this argument is no longer needed.
#     subject : string
#         Name of column containing the subject identifier (only required if a
#         within-subject or a between-subject factor is provided).
#     stimuli : string
#         Name of the column containing the stimuli.
#     within : string
#         Name of column containing the within factor (condition comparison).
#     accuracy : string
#         Name of the columns containing the accuracy.
#     confidence : string
#         Name of the column containing the confidence ratings.
#     nRatings : int
#         Number of discrete ratings. If a continuous rating scale was used, and
#         the number of unique ratings does not match `nRatings`, will convert to
#         discrete ratings using :py:func:`metadpy.utils.discreteRatings`.

#     Return
#     ------
#     pymcData : Dict

#     """
#     pymcData = {
#         "nSubj": data[subject].nunique(),
#         "subID": [],
#         "nCond": data[within].nunique(),
#         "condition": [],
#         "hits": [],
#         "falsealarms": [],
#         "s": [],
#         "n": [],
#         "nRatings": nRatings,
#         "Tol": 1e-05,
#         "cr": [],
#         "m": [],
#     }
#     pymcData["counts"] = np.zeros(
#         (pymcData["nSubj"], pymcData["nCond"], pymcData["nRatings"] * 4)
#     )
#     pymcData["hits"] = np.zeros((pymcData["nSubj"], pymcData["nCond"]))
#     pymcData["falsealarms"] = np.zeros((pymcData["nSubj"], pymcData["nCond"]))
#     pymcData["s"] = np.zeros((pymcData["nSubj"], pymcData["nCond"]))
#     pymcData["n"] = np.zeros((pymcData["nSubj"], pymcData["nCond"]))
#     pymcData["m"] = np.zeros((pymcData["nSubj"], pymcData["nCond"]))
#     pymcData["cr"] = np.zeros((pymcData["nSubj"], pymcData["nCond"]))
#     pymcData["condition"] = np.zeros((pymcData["nSubj"], pymcData["nCond"]))
#     pymcData["subID"] = np.zeros((pymcData["nSubj"], pymcData["nCond"]))
#     pymcData["c1"] = np.zeros((pymcData["nSubj"], pymcData["nCond"]))
#     pymcData["d1"] = np.zeros((pymcData["nSubj"], pymcData["nCond"]))

#     for nSub, sub in enumerate(data[subject].unique()):
#         for nCond, cond in enumerate(data[within].unique()):
#             nR_S1, nR_S2 = trials2counts(
#                 data=data[(data[subject] == sub) & (data[within] == cond)],
#                 stimuli=stimuli,
#                 accuracy=accuracy,
#                 confidence=confidence,
#                 nRatings=nRatings,
#             )

#             this_data = extractParameters(nR_S1, nR_S2)
#             pymcData["subID"][nSub, nCond] = nSub
#             pymcData["condition"][nSub, nCond] = nCond
#             pymcData["s"][nSub, nCond] = this_data["S"]
#             pymcData["n"][nSub, nCond] = this_data["N"]
#             pymcData["m"][nSub, nCond] = this_data["M"]
#             pymcData["cr"][nSub, nCond] = this_data["CR"]
#             pymcData["hits"][nSub, nCond] = this_data["H"]
#             pymcData["falsealarms"][nSub, nCond] = this_data["FA"]
#             pymcData["c1"][nSub, nCond] = this_data["c1"]
#             pymcData["d1"][nSub, nCond] = this_data["d1"]
#             pymcData["counts"][nSub, nCond, :] = this_data["counts"]

#     pymcData["subID"] = np.array(pymcData["subID"], dtype="int")
#     pymcData["condition"] = np.array(pymcData["condition"], dtype="int")
#     pymcData["s"] = np.array(pymcData["s"], dtype="int")
#     pymcData["n"] = np.array(pymcData["n"], dtype="int")
#     pymcData["m"] = np.array(pymcData["m"], dtype="int")
#     pymcData["cr"] = np.array(pymcData["cr"], dtype="int")
#     pymcData["counts"] = np.array(pymcData["counts"], dtype="int")
#     pymcData["hits"] = np.array(pymcData["hits"], dtype="int")
#     pymcData["falsealarms"] = np.array(pymcData["falsealarms"], dtype="int")

#     return pymcData
