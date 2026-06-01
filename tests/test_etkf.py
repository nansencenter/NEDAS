import numpy as np
import unittest
from NEDAS.assim_tools.assimilators.ETKF.core import (
    ensemble_transform_weights, apply_ensemble_transform,
)


class TestEnsembleTransformWeights(unittest.TestCase):

    def _make_data(self, nens=10, nlobs=5, seed=7):
        rng = np.random.default_rng(seed)
        obs_prior = rng.normal(0, 1, (nens, nlobs))
        obs = rng.normal(0, 1, nlobs)
        obs_err = np.ones(nlobs) * 0.5
        lfactor = np.ones(nlobs)
        return obs, obs_err, obs_prior, lfactor

    def test_weight_matrix_shape(self):
        obs, obs_err, obs_prior, lfactor = self._make_data(nens=10, nlobs=5)
        W = ensemble_transform_weights(obs, obs_err, obs_prior, lfactor)
        self.assertEqual(W.shape, (10, 10))

    def test_column_sums_equal_one(self):
        obs, obs_err, obs_prior, lfactor = self._make_data(nens=10, nlobs=5)
        W = ensemble_transform_weights(obs, obs_err, obs_prior, lfactor)
        np.testing.assert_allclose(W.sum(axis=0), 1.0, atol=1e-5)

    def test_huge_obs_error_gives_near_identity(self):
        nens = 8
        rng = np.random.default_rng(3)
        obs_prior = rng.normal(0, 1, (nens, 1))
        obs = np.array([0.0])
        obs_err = np.array([1e6])
        lfactor = np.ones(1)
        W = ensemble_transform_weights(obs, obs_err, obs_prior, lfactor)
        np.testing.assert_allclose(W, np.eye(nens), atol=1e-3)

    def test_posterior_spread_not_greater_than_prior(self):
        rng = np.random.default_rng(5)
        nens = 20
        obs_prior = rng.normal(0, 2, (nens, 3))
        obs = rng.normal(0, 1, 3)
        obs_err = np.ones(3)
        lfactor = np.ones(3)
        W = ensemble_transform_weights(obs, obs_err, obs_prior, lfactor)
        prior_ens = rng.normal(0, 2, nens)
        post_ens = apply_ensemble_transform(prior_ens, W)
        self.assertLessEqual(np.std(post_ens), np.std(prior_ens) + 1e-10)


class TestApplyEnsembleTransform(unittest.TestCase):

    def test_identity_weights_leave_ensemble_unchanged(self):
        nens = 10
        prior = np.random.default_rng(0).normal(0, 1, nens)
        post = apply_ensemble_transform(prior, np.eye(nens))
        np.testing.assert_allclose(post, prior, rtol=1e-12)

    def test_output_length_matches_input(self):
        nens = 15
        prior = np.random.default_rng(1).normal(0, 1, nens)
        post = apply_ensemble_transform(prior, np.eye(nens))
        self.assertEqual(len(post), nens)

    def test_uniform_weights_collapse_to_mean(self):
        nens = 8
        prior = np.arange(1.0, nens + 1)
        W = np.full((nens, nens), 1.0 / nens)
        post = apply_ensemble_transform(prior, W)
        np.testing.assert_allclose(post, np.mean(prior), rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
