import numpy as np
import unittest
from NEDAS.assim_tools.assimilators.TopazDEnKF.core import (
    ensemble_transform_weights,
    apply_ensemble_transform,
    local_analysis_main,
)
from NEDAS.assim_tools.localization.distance_based import step_func


def _make_data(nens=12, nlobs=5, seed=7):
    rng = np.random.default_rng(seed)
    obs_prior = rng.normal(0, 1, (nens, nlobs))
    obs = rng.normal(0, 1, nlobs)
    obs_err = np.ones(nlobs) * 0.5
    lfactor = np.ones(nlobs)
    return obs, obs_err, obs_prior, lfactor


class TestTopazEnsembleTransformWeights(unittest.TestCase):

    def test_shape(self):
        obs, obs_err, obs_prior, lfactor = _make_data(nens=12, nlobs=5)
        W = ensemble_transform_weights(obs, obs_err, obs_prior, lfactor, rfactor=1.0, kfactor=1e6)
        self.assertEqual(W.shape, (12, 12))

    def test_column_sums_one(self):
        obs, obs_err, obs_prior, lfactor = _make_data(nens=12, nlobs=5)
        W = ensemble_transform_weights(obs, obs_err, obs_prior, lfactor, rfactor=1.0, kfactor=1e6)
        np.testing.assert_allclose(W.sum(axis=0), 1.0, atol=1e-5)

    def test_huge_obs_error_gives_near_identity(self):
        nens = 8
        rng = np.random.default_rng(3)
        obs_prior = rng.normal(0, 1, (nens, 2))
        obs = rng.normal(0, 1, 2)
        obs_err = np.array([1e8, 1e8])
        lfactor = np.ones(2)
        W = ensemble_transform_weights(obs, obs_err, obs_prior, lfactor, rfactor=1.0, kfactor=1e6)
        np.testing.assert_allclose(W, np.eye(nens), atol=1e-3)

    def test_obs_dominated_case_reduces_spread(self):
        # With tiny obs error, the posterior ensemble should have smaller spread
        nens = 20
        rng = np.random.default_rng(5)
        obs_prior = rng.normal(0, 2, (nens, 3))
        obs = rng.normal(0, 1, 3)
        obs_err = np.ones(3) * 0.01
        lfactor = np.ones(3)
        W = ensemble_transform_weights(obs, obs_err, obs_prior, lfactor, rfactor=1.0, kfactor=1e6)
        prior_ens = rng.normal(0, 2, nens)
        post_ens = apply_ensemble_transform(prior_ens, W)
        self.assertLess(np.std(post_ens), np.std(prior_ens))

    def test_rfactor_reduces_spread_less(self):
        # rfactor > 1 inflates obs error variance for spread update → less spread reduction
        nens = 20
        rng = np.random.default_rng(9)
        obs_prior = rng.normal(0, 2, (nens, 3))
        obs = rng.normal(0, 0.5, 3)
        obs_err = np.ones(3) * 0.5
        lfactor = np.ones(3)
        prior_ens = rng.normal(0, 2, nens)

        W1 = ensemble_transform_weights(obs, obs_err, obs_prior, lfactor, rfactor=1.0, kfactor=1e6)
        W4 = ensemble_transform_weights(obs, obs_err, obs_prior, lfactor, rfactor=4.0, kfactor=1e6)

        std1 = np.std(apply_ensemble_transform(prior_ens, W1))
        std4 = np.std(apply_ensemble_transform(prior_ens, W4))
        self.assertGreater(std4, std1)

    def test_obs_space_regime_nlobs_less_than_nens(self):
        # nlobs < nens → obs-space Cholesky branch
        obs, obs_err, obs_prior, lfactor = _make_data(nens=20, nlobs=4, seed=11)
        W = ensemble_transform_weights(obs, obs_err, obs_prior, lfactor, rfactor=1.0, kfactor=1e6)
        self.assertEqual(W.shape, (20, 20))
        np.testing.assert_allclose(W.sum(axis=0), 1.0, atol=1e-5)


class TestTopazApplyEnsembleTransform(unittest.TestCase):

    def test_identity_leaves_ensemble_unchanged(self):
        nens = 10
        prior = np.random.default_rng(0).normal(0, 1, nens)
        post = apply_ensemble_transform(prior, np.eye(nens))
        np.testing.assert_allclose(post, prior, rtol=1e-12)

    def test_uniform_weights_collapse_to_mean(self):
        nens = 8
        prior = np.arange(1.0, nens + 1)
        W = np.full((nens, nens), 1.0 / nens)
        post = apply_ensemble_transform(prior, W)
        np.testing.assert_allclose(post, np.mean(prior), rtol=1e-12)

    def test_output_length(self):
        nens = 15
        prior = np.ones(nens)
        post = apply_ensemble_transform(prior, np.eye(nens))
        self.assertEqual(len(post), nens)


class TestLocalAnalysisMain(unittest.TestCase):

    def _run_local_analysis(self, nens=20, nfld=3, nlobs=5, seed=42, nlobs_max=None):
        if nlobs_max is None:
            nlobs_max = nlobs
        rng = np.random.default_rng(seed)
        state_prior = rng.normal(0, 2, (nens, nfld))
        obs_prior   = rng.normal(0, 2, (nens, nlobs))
        obs     = rng.normal(0, 1, nlobs)
        obs_err = np.ones(nlobs) * 0.5
        hlfactor  = np.ones(nlobs)
        state_z   = np.zeros(nfld)
        obs_z     = np.zeros(nlobs)
        vroi      = 1.0
        state_t   = np.zeros(nfld)
        obs_t     = np.zeros(nlobs)
        troi      = 1.0
        impact_on_state = np.ones((nlobs, nfld))

        prior_copy = state_prior.copy()
        local_analysis_main(
            state_prior, obs_prior, obs, obs_err, hlfactor,
            state_z, obs_z, vroi, step_func,
            state_t, obs_t, troi, step_func,
            impact_on_state, 1.0, 1e6, nlobs_max,
        )
        return prior_copy, state_prior  # (before, after)

    def test_in_place_modification(self):
        before, after = self._run_local_analysis()
        self.assertFalse(np.allclose(before, after))

    def test_posterior_spread_not_greater_than_prior(self):
        before, after = self._run_local_analysis()
        for n in range(before.shape[1]):
            self.assertLessEqual(np.std(after[:, n]), np.std(before[:, n]) + 1e-8)

    def test_zero_spread_state_unchanged(self):
        nens, nfld, nlobs = 20, 3, 5
        rng = np.random.default_rng(1)
        state_prior = np.zeros((nens, nfld))  # zero spread
        obs_prior   = rng.normal(0, 1, (nens, nlobs))
        obs     = rng.normal(0, 1, nlobs)
        obs_err = np.ones(nlobs) * 0.5
        hlfactor  = np.ones(nlobs)
        state_z = np.zeros(nfld)
        obs_z   = np.zeros(nlobs)
        state_t = np.zeros(nfld)
        obs_t   = np.zeros(nlobs)
        impact_on_state = np.ones((nlobs, nfld))

        before = state_prior.copy()
        local_analysis_main(
            state_prior, obs_prior, obs, obs_err, hlfactor,
            state_z, obs_z, 1.0, step_func,
            state_t, obs_t, 1.0, step_func,
            impact_on_state, 1.0, 1e6, nlobs,
        )
        np.testing.assert_array_equal(state_prior, before)

    def test_nlobs_max_limits_obs_used(self):
        # With nlobs_max=1 vs nlobs_max=5, results differ
        nens, nfld, nlobs = 20, 3, 5
        rng = np.random.default_rng(2)
        obs_prior = rng.normal(0, 2, (nens, nlobs))
        obs = rng.normal(0, 1, nlobs)
        obs_err = np.ones(nlobs) * 0.5
        hlfactor = np.ones(nlobs)
        state_z = np.zeros(nfld)
        obs_z   = np.zeros(nlobs)
        state_t = np.zeros(nfld)
        obs_t   = np.zeros(nlobs)
        impact_on_state = np.ones((nlobs, nfld))

        state1 = rng.normal(0, 2, (nens, nfld))
        state2 = state1.copy()

        local_analysis_main(state1, obs_prior, obs, obs_err, hlfactor,
                            state_z, obs_z, 1.0, step_func,
                            state_t, obs_t, 1.0, step_func,
                            impact_on_state, 1.0, 1e6, 1)

        local_analysis_main(state2, obs_prior, obs, obs_err, hlfactor,
                            state_z, obs_z, 1.0, step_func,
                            state_t, obs_t, 1.0, step_func,
                            impact_on_state, 1.0, 1e6, nlobs)

        self.assertFalse(np.allclose(state1, state2))


if __name__ == '__main__':
    unittest.main()
