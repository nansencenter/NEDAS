import numpy as np
import unittest
from NEDAS.assim_tools.assimilators.EAKF.core import (
    obs_increment_eakf, update_ensemble,
)


class TestObsIncrementEAKF(unittest.TestCase):

    def _centered_prior(self, mean=0.0, std=1.0, nens=20, seed=42):
        rng = np.random.default_rng(seed)
        ens = rng.normal(0, std, nens)
        ens -= ens.mean()
        return (ens + mean).astype(np.float64)

    def test_zero_spread_gives_zero_increment(self):
        prior = np.ones(10) * 5.0
        incr = obs_increment_eakf(prior, obs=5.0, obs_err=1.0)
        np.testing.assert_array_equal(incr, 0.0)

    def test_posterior_mean_is_weighted_average(self):
        prior = self._centered_prior(mean=0.0, std=2.0, nens=100)
        obs, obs_err = 3.0, 1.0
        incr = obs_increment_eakf(prior, obs=obs, obs_err=obs_err)
        post = prior + incr
        prior_var = np.var(prior, ddof=1)
        var_ratio = obs_err**2 / (prior_var + obs_err**2)
        expected_mean = var_ratio * np.mean(prior) + (1 - var_ratio) * obs
        self.assertAlmostEqual(np.mean(post), expected_mean, places=10)

    def test_posterior_variance_reduced(self):
        prior = self._centered_prior(mean=0.0, std=3.0, nens=50)
        incr = obs_increment_eakf(prior, obs=1.0, obs_err=2.0)
        self.assertLess(np.var(prior + incr, ddof=1), np.var(prior, ddof=1))

    def test_posterior_variance_formula(self):
        prior = self._centered_prior(mean=0.0, std=2.0, nens=200)
        obs_err = 1.5
        incr = obs_increment_eakf(prior, obs=1.0, obs_err=obs_err)
        prior_var = np.var(prior, ddof=1)
        var_ratio = obs_err**2 / (prior_var + obs_err**2)
        self.assertAlmostEqual(np.var(prior + incr, ddof=1), var_ratio * prior_var, places=8)

    def test_increment_mean_equals_mean_shift(self):
        prior = self._centered_prior(mean=2.0, std=1.0, nens=20)
        incr = obs_increment_eakf(prior, obs=5.0, obs_err=0.5)
        self.assertAlmostEqual(np.mean(incr), np.mean(prior + incr) - np.mean(prior), places=12)


class TestUpdateEnsembleEAKF(unittest.TestCase):

    def test_no_update_when_obs_prior_var_zero(self):
        obs_prior = np.ones(10) * 3.0
        obs_incr = np.ones(10)
        ens_prior = np.random.default_rng(0).normal(0, 1, (10, 1, 3))
        ens_post = update_ensemble(ens_prior.copy(), obs_prior, obs_incr, np.ones(3))
        np.testing.assert_array_equal(ens_post, ens_prior)

    def test_zero_localization_gives_no_update(self):
        rng = np.random.default_rng(1)
        nens = 10
        obs_prior = rng.normal(0, 1, nens)
        obs_incr = rng.normal(0, 0.5, nens)
        ens_prior = rng.normal(0, 1, (nens, 1, 5))
        ens_post = update_ensemble(ens_prior.copy(), obs_prior, obs_incr, np.zeros(5))
        np.testing.assert_array_equal(ens_post, ens_prior)

    def test_nonzero_localization_changes_state(self):
        rng = np.random.default_rng(2)
        nens = 20
        obs_prior = rng.normal(0, 1, nens)
        obs_incr = rng.normal(0, 0.5, nens)
        ens_prior = rng.normal(0, 1, (nens, 1, 3))
        ens_post = update_ensemble(ens_prior.copy(), obs_prior, obs_incr, np.ones(3))
        self.assertFalse(np.allclose(ens_post, ens_prior))


if __name__ == '__main__':
    unittest.main()
