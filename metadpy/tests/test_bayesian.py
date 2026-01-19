# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
import pymc as pm
import pytest

from metadpy import load_dataset
from metadpy.bayesian import extractParameters, hmetad, hmetad_pooled, preprocess_group
from metadpy.utils import ratings2df


class Testsdt(TestCase):
    def test_extractParameters(self):
        """Test preprocess function"""
        nR_S1 = [52, 32, 35, 37, 26, 12, 4, 2]
        nR_S2 = [2, 5, 15, 22, 33, 38, 40, 45]
        data = extractParameters(nR_S1, nR_S2)
        assert round(data["d1"], 3) == 1.535
        assert round(data["c1"]) == 0
        assert np.all(
            data["counts"]
            == np.array([52, 32, 35, 37, 26, 12, 4, 2, 2, 5, 15, 22, 33, 38, 40, 45])
        )
        assert data["nratings"] == 4
        assert data["Tol"] == 1e-05
        assert data["FA"] == 44
        assert data["CR"] == 156
        assert data["M"] == 44
        assert data["H"] == 156
        assert data["N"] == 200
        assert data["S"] == 200

    def test_hmetad(self):
        """Test hmetad function"""
        group_df = load_dataset("rm")

        ####################
        # Test subject level
        ####################
        model, _ = hmetad(
            nR_S1=np.array([52, 32, 35, 37, 26, 12, 4, 2]),
            nR_S2=np.array([2, 5, 15, 22, 33, 38, 40, 45]),
            nRatings=4,
            sample_model=False,
        )
        assert isinstance(model, pm.Model)

        with pytest.raises(ValueError):
            model = hmetad(
                data=None,
                nR_S1=None,
                nR_S2=None,
                nRatings=4,
                sample_model=False,
            )

        this_df = group_df[(group_df.Subject == 0) & (group_df.Condition == 0)]
        with pytest.raises(ValueError):
            model, _ = hmetad(
                data=this_df,
                nRatings=None,
                stimuli="Stimuli",
                accuracy="Accuracy",
                confidence="Confidence",
                sample_model=False,
            )

        model, _ = hmetad(
            data=this_df,
            nRatings=4,
            stimuli="Stimuli",
            accuracy="Accuracy",
            confidence="Confidence",
            sample_model=False,
        )
        assert isinstance(model, pm.Model)

        # Force ratings discretization
        model, _ = hmetad(
            data=this_df,
            nRatings=3,
            stimuli="Stimuli",
            accuracy="Accuracy",
            confidence="Confidence",
            sample_model=False,
        )
        assert isinstance(model, pm.Model)

        # Using nR_S1 and nR_S2 vectors as inputs
        pymc_df = hmetad(
            nR_S1=np.array([52, 32, 35, 37, 26, 12, 4, 2]),
            nR_S2=np.array([2, 5, 15, 22, 33, 38, 40, 45]),
            nRatings=4,
            output="dataframe",
        )

        assert round(pymc_df["d"].values[0], 2) - 1.53 < 0.01
        assert round(pymc_df["c"].values[0], 2) - 0.0 < 0.01
        assert round(pymc_df["meta_d"].values[0], 2) - 1.58 < 0.01
        assert round(pymc_df["m_ratio"].values[0], 2) - 1.03 < 0.01

        # Using a dataframe as input
        this_df = ratings2df(
            nR_S1=np.array([52, 32, 35, 37, 26, 12, 4, 2]),
            nR_S2=np.array([2, 5, 15, 22, 33, 38, 40, 45]),
        )
        pymc_df = hmetad(
            data=this_df,
            nRatings=4,
            stimuli="Stimuli",
            accuracy="Accuracy",
            confidence="Confidence",
            output="dataframe",
        )

        assert round(pymc_df["d"].values[0], 2) - 1.53 < 0.01
        assert round(pymc_df["c"].values[0], 2) - 0.0 < 0.01
        assert round(pymc_df["meta_d"].values[0], 2) - 1.58 < 0.01
        assert round(pymc_df["m_ratio"].values[0], 2) - 1.03 < 0.01

    def test_hmetad_group_level(self):
        nR_S1 = np.array(
            [
                [52, 32, 35, 37, 26, 12, 4, 2],
                [50, 30, 33, 35, 22, 10, 6, 3],
            ]
        )
        nR_S2 = np.array(
            [
                [2, 5, 15, 22, 33, 38, 40, 45],
                [3, 6, 12, 20, 30, 35, 42, 48],
            ]
        )
        model, _ = hmetad(
            nR_S1=nR_S1,
            nR_S2=nR_S2,
            nRatings=4,
            subject="Subject",
            sample_model=False,
        )
        assert isinstance(model, pm.Model)

    def test_hmetad_pooled_counts(self):
        nR_S1 = np.array(
            [
                [52, 32, 35, 37, 26, 12, 4, 2],
                [50, 30, 33, 35, 22, 10, 6, 3],
            ]
        )
        nR_S2 = np.array(
            [
                [2, 5, 15, 22, 33, 38, 40, 45],
                [3, 6, 12, 20, 30, 35, 42, 48],
            ]
        )
        model, _ = hmetad_pooled(
            nR_S1=nR_S1,
            nR_S2=nR_S2,
            nRatings=4,
            sample_model=False,
        )
        assert isinstance(model, pm.Model)

    def test_hmetad_pooled_within(self):
        nR_S1 = np.array([52, 32, 35, 37, 26, 12, 4, 2])
        nR_S2 = np.array([2, 5, 15, 22, 33, 38, 40, 45])
        base_df = ratings2df(nR_S1=nR_S1, nR_S2=nR_S2)
        frames = []
        for sub in [0, 1]:
            for cond in [0, 1]:
                df = base_df.copy()
                df["Subject"] = sub
                df["Condition"] = cond
                frames.append(df)
        pooled_df = pd.concat(frames, ignore_index=True)

        models, _ = hmetad_pooled(
            data=pooled_df,
            nRatings=4,
            stimuli="Stimuli",
            accuracy="Accuracy",
            confidence="Confidence",
            within="Condition",
            sample_model=False,
        )
        assert isinstance(models[0], pm.Model)
        assert isinstance(models[1], pm.Model)

    def test_hmetad_rm1way(self):
        nR_S1 = np.array([52, 32, 35, 37, 26, 12, 4, 2])
        nR_S2 = np.array([2, 5, 15, 22, 33, 38, 40, 45])
        base_df = ratings2df(nR_S1=nR_S1, nR_S2=nR_S2)
        frames = []
        for sub in [0, 1]:
            for cond in [0, 1]:
                df = base_df.copy()
                df["Subject"] = sub
                df["Condition"] = cond
                frames.append(df)
        rm_df = pd.concat(frames, ignore_index=True)

        model, _ = hmetad(
            data=rm_df,
            nRatings=4,
            stimuli="Stimuli",
            accuracy="Accuracy",
            confidence="Confidence",
            subject="Subject",
            within="Condition",
            sample_model=False,
        )
        assert isinstance(model, pm.Model)

    def test_preprocess_group_covariate_layouts(self):
        nR_S1 = np.array(
            [
                [10, 9, 8, 7, 6, 5, 4, 3],
                [11, 10, 9, 8, 7, 6, 5, 4],
                [12, 11, 10, 9, 8, 7, 6, 5],
            ]
        )
        nR_S2 = np.array(
            [
                [3, 4, 5, 6, 7, 8, 9, 10],
                [4, 5, 6, 7, 8, 9, 10, 11],
                [5, 6, 7, 8, 9, 10, 11, 12],
            ]
        )
        X_sp = np.array([[0.1, 1.2], [0.2, 1.3], [0.3, 1.4]])
        _, X_out = preprocess_group(
            data=None, nR_S1=nR_S1, nR_S2=nR_S2, nRatings=4, X=X_sp
        )
        assert np.array_equal(X_out, X_sp)

        X_ps = X_sp.T
        _, X_out_ps = preprocess_group(
            data=None, nR_S1=nR_S1, nR_S2=nR_S2, nRatings=4, X=X_ps
        )
        assert np.array_equal(X_out_ps, X_sp)

        X_square = np.eye(nR_S1.shape[0])
        _, X_out_square = preprocess_group(
            data=None, nR_S1=nR_S1, nR_S2=nR_S2, nRatings=4, X=X_square
        )
        assert np.array_equal(X_out_square, X_square)
        doc = preprocess_group.__doc__ or ""
        assert "square" in doc.lower()
        assert "not transposed" in doc.lower()


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
