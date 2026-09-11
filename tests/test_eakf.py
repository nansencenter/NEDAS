import numpy as np
import unittest
from NEDAS.assim_tools.assimilators.EAKF.core import (
    obs_increment_eakf, update_ensemble, obs_increment_eakf_hybrid, update_ensemble_hybrid,
)
from NEDAS.assim_tools.assimilators.ETKF.core import ensemble_transform_weights as etkf_transform_weights


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


class TestEAKFHybrid(unittest.TestCase):
    """Serial EAKF with the hybrid covariance P = (1-beta)*P_d + beta*static_var_scaling*P_s, a separate
    batch of static members: without static members it is the plain EAKF, and for a single observation
    it equals the batch hybrid ETKF for both perturbation updates."""

    def setUp(self):
        rng = np.random.default_rng(5)
        self.nens_dynamic, self.nens_static, self.nstate = 6, 9, 4
        self.ens_dynamic = rng.normal(0, 1, (self.nens_dynamic, self.nstate))  # members as rows
        self.ens_static = rng.normal(0.5, 2, (self.nens_static, self.nstate))
        self.obs, self.obs_err = 0.7, 0.8
        self.beta, self.static_var_scaling = 0.4, 0.3

    def test_no_static_members_matches_plain_eakf(self):
        obs_prior = self.ens_dynamic[:, 0].copy()
        obs_incr_ref = obs_increment_eakf(obs_prior, self.obs, self.obs_err)
        ens_post_ref = update_ensemble(self.ens_dynamic.copy(), obs_prior, obs_incr_ref, np.ones(self.nstate))
        for hybrid_perturbation in (False, True):
            obs_incr = obs_increment_eakf_hybrid(obs_prior, np.zeros(0), self.obs, self.obs_err, 1.0, 0.0,
                                                 hybrid_perturbation)
            np.testing.assert_allclose(obs_incr, obs_incr_ref, atol=1e-14)
            ens_post = update_ensemble_hybrid(self.ens_dynamic.copy(), np.zeros((0, self.nstate)), obs_prior,
                                              np.zeros(0), obs_incr_ref, np.ones(self.nstate), 1.0, 0.0,
                                              hybrid_perturbation)
            np.testing.assert_allclose(ens_post, ens_post_ref, atol=1e-13)

    def test_single_obs_matches_batch_etkf_hybrid(self):
        weight_dynamic = 1 - self.beta
        weight_static = self.beta * self.static_var_scaling
        fac_dynamic = np.sqrt(weight_dynamic) / np.sqrt(self.nens_dynamic - 1)
        fac_static = np.sqrt(weight_static) / np.sqrt(self.nens_static - 1)
        obs_prior, obs_prior_static = self.ens_dynamic[:, 0].copy(), self.ens_static[:, 0].copy()
        ens_static = self.ens_static.copy()
        for hybrid_perturbation in (False, True):
            obs_incr = obs_increment_eakf_hybrid(obs_prior, obs_prior_static, self.obs, self.obs_err,
                                                 weight_dynamic, weight_static, hybrid_perturbation)
            post_eakf = update_ensemble_hybrid(self.ens_dynamic.copy(), ens_static, obs_prior, obs_prior_static,
                                               obs_incr, np.ones(self.nstate), weight_dynamic, weight_static,
                                               hybrid_perturbation)
            weights, weights_static = etkf_transform_weights(np.array([self.obs]), np.array([self.obs_err]),
                                                             obs_prior[:, None].copy(), obs_prior_static[:, None].copy(),
                                                             np.ones(1), np.eye(self.nens_dynamic), False,
                                                             fac_dynamic, fac_static, hybrid_perturbation)
            post_etkf = (self.ens_dynamic.T @ weights + self.ens_static.T @ weights_static).T
            np.testing.assert_allclose(post_eakf, post_etkf, atol=1e-10)
        np.testing.assert_array_equal(ens_static, self.ens_static)  # static members not updated


if __name__ == '__main__':
    unittest.main()
