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


def _plain_denkf_weights(obs, obs_err, obs_prior, lfactor, rfactor):
    """plain DEnKF: no static members, the dynamic ensemble covariance alone"""
    nens, nlobs = obs_prior.shape
    weights, _ = ensemble_transform_weights(obs, obs_err, obs_prior, np.zeros((0, nlobs)), lfactor, rfactor,
                                            1.0 / np.sqrt(nens - 1), 0.0, False)
    return weights


def _plain_local_analysis_main(state_prior, obs_prior, *args):
    """local_analysis_main without static members"""
    nens, nfld = state_prior.shape
    local_analysis_main(state_prior, obs_prior, np.zeros((0, nfld)), np.zeros((0, obs_prior.shape[1])),
                        *args, 1.0 / np.sqrt(nens - 1), 0.0, False)


class TestTopazEnsembleTransformWeights(unittest.TestCase):

    def test_shape(self):
        obs, obs_err, obs_prior, lfactor = _make_data(nens=12, nlobs=5)
        W = _plain_denkf_weights(obs, obs_err, obs_prior, lfactor, rfactor=1.0)
        self.assertEqual(W.shape, (12, 12))

    def test_column_sums_one(self):
        obs, obs_err, obs_prior, lfactor = _make_data(nens=12, nlobs=5)
        W = _plain_denkf_weights(obs, obs_err, obs_prior, lfactor, rfactor=1.0)
        np.testing.assert_allclose(W.sum(axis=0), 1.0, atol=1e-5)

    def test_huge_obs_error_gives_near_identity(self):
        nens = 8
        rng = np.random.default_rng(3)
        obs_prior = rng.normal(0, 1, (nens, 2))
        obs = rng.normal(0, 1, 2)
        obs_err = np.array([1e8, 1e8])
        lfactor = np.ones(2)
        W = _plain_denkf_weights(obs, obs_err, obs_prior, lfactor, rfactor=1.0)
        np.testing.assert_allclose(W, np.eye(nens), atol=1e-3)

    def test_obs_dominated_case_reduces_spread(self):
        # With tiny obs error, the posterior ensemble should have smaller spread
        nens = 20
        rng = np.random.default_rng(5)
        obs_prior = rng.normal(0, 2, (nens, 3))
        obs = rng.normal(0, 1, 3)
        obs_err = np.ones(3) * 0.01
        lfactor = np.ones(3)
        W = _plain_denkf_weights(obs, obs_err, obs_prior, lfactor, rfactor=1.0)
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

        W1 = _plain_denkf_weights(obs, obs_err, obs_prior, lfactor, rfactor=1.0)
        W4 = _plain_denkf_weights(obs, obs_err, obs_prior, lfactor, rfactor=4.0)

        std1 = np.std(apply_ensemble_transform(prior_ens, W1))
        std4 = np.std(apply_ensemble_transform(prior_ens, W4))
        self.assertGreater(std4, std1)

    def test_obs_space_regime_nlobs_less_than_nens(self):
        # nlobs < nens → obs-space Cholesky branch
        obs, obs_err, obs_prior, lfactor = _make_data(nens=20, nlobs=4, seed=11)
        W = _plain_denkf_weights(obs, obs_err, obs_prior, lfactor, rfactor=1.0)
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
        impact_on_variable = np.ones((nlobs, nfld))

        prior_copy = state_prior.copy()
        _plain_local_analysis_main(
            state_prior, obs_prior, obs, obs_err, hlfactor,
            state_z, obs_z, vroi, step_func,
            state_t, obs_t, troi, step_func,
            impact_on_variable, 1.0, nlobs_max,
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
        impact_on_variable = np.ones((nlobs, nfld))

        before = state_prior.copy()
        _plain_local_analysis_main(
            state_prior, obs_prior, obs, obs_err, hlfactor,
            state_z, obs_z, 1.0, step_func,
            state_t, obs_t, 1.0, step_func,
            impact_on_variable, 1.0, nlobs,
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
        impact_on_variable = np.ones((nlobs, nfld))

        state1 = rng.normal(0, 2, (nens, nfld))
        state2 = state1.copy()

        _plain_local_analysis_main(state1, obs_prior, obs, obs_err, hlfactor,
                            state_z, obs_z, 1.0, step_func,
                            state_t, obs_t, 1.0, step_func,
                            impact_on_variable, 1.0, 1)

        _plain_local_analysis_main(state2, obs_prior, obs, obs_err, hlfactor,
                            state_z, obs_z, 1.0, step_func,
                            state_t, obs_t, 1.0, step_func,
                            impact_on_variable, 1.0, nlobs)

        self.assertFalse(np.allclose(state1, state2))


class TestTopazHybridCovariance(unittest.TestCase):
    """Hybrid DEnKF with nens_dynamic dynamic members and a separate batch of nens_static static members,
    P = (1-beta)*P_d + beta*static_var_scaling*P_s; H picks the first nobs state components.
    nobs=3 exercises the obs-space solve, nobs=20 the ensemble-space solve."""

    def _case(self, nobs, nstate, seed=31):
        rng = np.random.default_rng(seed)
        self.nens_dynamic, self.nens_static, self.nobs, self.nstate = 6, 9, nobs, nstate
        self.ens_dynamic = rng.normal(0, 1, (self.nens_dynamic, nstate))  # members as rows
        self.ens_static = rng.normal(0.5, 2, (self.nens_static, nstate))
        self.obs = rng.normal(0, 1, nobs)
        self.obs_err = rng.uniform(0.5, 1.0, nobs)
        self.beta, self.static_var_scaling = 0.4, 0.3

    def _weights(self, hybrid_perturbation, rfactor=1.0, beta=None):
        beta = self.beta if beta is None else beta
        fac_dynamic = np.sqrt(1 - beta) / np.sqrt(self.nens_dynamic - 1)
        fac_static = np.sqrt(beta * self.static_var_scaling) / np.sqrt(self.nens_static - 1)
        return ensemble_transform_weights(self.obs, self.obs_err, self.ens_dynamic[:, :self.nobs].copy(),
                                          self.ens_static[:, :self.nobs].copy(), np.ones(self.nobs), rfactor,
                                          fac_dynamic, fac_static, hybrid_perturbation)

    def _post_dynamic(self, weights, weights_static):
        return (self.ens_dynamic.T @ weights + self.ens_static.T @ weights_static).T

    def _gain(self, rfactor=1.0):
        P = ((1 - self.beta) * np.cov(self.ens_dynamic.T)
             + self.beta * self.static_var_scaling * np.cov(self.ens_static.T))
        H = np.eye(self.nstate)[:self.nobs]
        R = np.diag(self.obs_err**2) * rfactor
        return P @ H.T @ np.linalg.inv(H @ P @ H.T + R), H

    def test_beta_zero_reduces_to_plain_denkf(self):
        for nobs, nstate in ((3, 5), (20, 20)):
            self._case(nobs, nstate)
            for rfactor in (1.0, 2.0):
                weights_ref = _plain_denkf_weights(self.obs, self.obs_err, self.ens_dynamic[:, :nobs].copy(),
                                                   np.ones(nobs), rfactor)
                for hybrid_perturbation in (False, True):
                    weights, weights_static = self._weights(hybrid_perturbation, rfactor, beta=0.0)
                    np.testing.assert_allclose(weights, weights_ref, atol=1e-10)
                    np.testing.assert_allclose(weights_static, 0, atol=1e-12)

    def test_mean_update_uses_hybrid_gain(self):
        for nobs, nstate in ((3, 5), (20, 20)):
            self._case(nobs, nstate)
            K, H = self._gain()
            mean_dynamic = self.ens_dynamic.mean(0)
            mean_ref = mean_dynamic + K @ (self.obs - H @ mean_dynamic)
            for hybrid_perturbation in (False, True):
                for rfactor in (1.0, 2.0):
                    weights, weights_static = self._weights(hybrid_perturbation, rfactor)
                    np.testing.assert_allclose(weights.sum(axis=0), 1.0, atol=1e-10)
                    np.testing.assert_allclose(weights_static.sum(axis=0), 0.0, atol=1e-10)
                    np.testing.assert_allclose(self._post_dynamic(weights, weights_static).mean(0), mean_ref, atol=1e-10)

    def test_hybrid_perturbations_use_half_hybrid_gain(self):
        # A_d <- A_d - 0.5 K H A_d with K the gain of the hybrid covariance (obs error inflated by rfactor)
        for nobs, nstate in ((3, 5), (20, 20)):
            self._case(nobs, nstate)
            for rfactor in (1.0, 2.0):
                K, H = self._gain(rfactor)
                anom_dynamic = self.ens_dynamic - self.ens_dynamic.mean(0)
                post = self._post_dynamic(*self._weights(True, rfactor))
                np.testing.assert_allclose(post - post.mean(0),
                                           anom_dynamic - 0.5 * (K @ H @ anom_dynamic.T).T, atol=1e-10)

    def test_dynamic_perturbations_ignore_static(self):
        for nobs, nstate in ((3, 5), (20, 20)):
            self._case(nobs, nstate)
            weights_ref = _plain_denkf_weights(self.obs, self.obs_err, self.ens_dynamic[:, :nobs].copy(),
                                               np.ones(nobs), 2.0)
            post_ref = (self.ens_dynamic.T @ weights_ref).T
            post = self._post_dynamic(*self._weights(False, 2.0))
            np.testing.assert_allclose(post - post.mean(0), post_ref - post_ref.mean(0), atol=1e-10)


if __name__ == '__main__':
    unittest.main()
