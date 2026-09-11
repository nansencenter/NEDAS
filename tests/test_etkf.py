import numpy as np
import unittest
from NEDAS.assim_tools.assimilators.ETKF.core import (
    ensemble_transform_weights, apply_ensemble_transform,
    mean_preserving_rotation, local_analysis_main,
)
from NEDAS.assim_tools.localization.distance_based import gaspari_cohn_func


def _eye(nens):
    return np.eye(nens)


def _plain_etkf_weights(obs, obs_err, obs_prior, local_factor, rotation, use_eigen=False):
    """plain ETKF: no static members, the dynamic ensemble covariance alone"""
    nens, nlobs = obs_prior.shape
    weights, _ = ensemble_transform_weights(obs, obs_err, obs_prior, np.zeros((0, nlobs)), local_factor,
                                            rotation, use_eigen, 1.0 / np.sqrt(nens - 1), 0.0, False)
    return weights


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
        W = _plain_etkf_weights(obs, obs_err, obs_prior, lfactor, _eye(10))
        self.assertEqual(W.shape, (10, 10))

    def test_column_sums_equal_one(self):
        obs, obs_err, obs_prior, lfactor = self._make_data(nens=10, nlobs=5)
        W = _plain_etkf_weights(obs, obs_err, obs_prior, lfactor, _eye(10))
        np.testing.assert_allclose(W.sum(axis=0), 1.0, atol=1e-5)

    def test_huge_obs_error_gives_near_identity(self):
        nens = 8
        rng = np.random.default_rng(3)
        obs_prior = rng.normal(0, 1, (nens, 1))
        obs = np.array([0.0])
        obs_err = np.array([1e6])
        lfactor = np.ones(1)
        W = _plain_etkf_weights(obs, obs_err, obs_prior, lfactor, _eye(nens))
        np.testing.assert_allclose(W, np.eye(nens), atol=1e-3)

    def test_posterior_spread_not_greater_than_prior(self):
        rng = np.random.default_rng(5)
        nens = 20
        obs_prior = rng.normal(0, 2, (nens, 3))
        obs = rng.normal(0, 1, 3)
        obs_err = np.ones(3)
        lfactor = np.ones(3)
        W = _plain_etkf_weights(obs, obs_err, obs_prior, lfactor, _eye(nens))
        prior_ens = rng.normal(0, 2, nens)
        post_ens = apply_ensemble_transform(prior_ens, W)
        self.assertLessEqual(np.std(post_ens), np.std(prior_ens) + 1e-10)


class TestTransformSolvers(unittest.TestCase):
    """The 'svd' (decompose S directly) and 'eigen' (decompose I + S S^T)
    solvers must yield the same transform up to numerical precision."""

    def test_svd_matches_eigen(self):
        rng = np.random.default_rng(11)
        nens, nlobs = 12, 6
        obs_prior = rng.normal(0, 1, (nens, nlobs))
        obs = rng.normal(0, 1, nlobs)
        obs_err = np.ones(nlobs) * 0.7
        lfactor = rng.uniform(0, 1, nlobs)
        W_svd = _plain_etkf_weights(obs, obs_err, obs_prior, lfactor, _eye(nens), False)
        W_eig = _plain_etkf_weights(obs, obs_err, obs_prior, lfactor, _eye(nens), True)
        np.testing.assert_allclose(W_svd, W_eig, atol=1e-10)


class TestRandomRotation(unittest.TestCase):

    def test_rotation_fixes_ones_and_is_orthogonal(self):
        np.random.seed(1)
        U = mean_preserving_rotation(10)
        np.testing.assert_allclose(U @ np.ones(10), np.ones(10), atol=1e-10)
        np.testing.assert_allclose(U @ U.T, np.eye(10), atol=1e-10)

    def test_rotation_preserves_column_sums(self):
        rng = np.random.default_rng(13)
        obs_prior = rng.normal(0, 1, (12, 6))
        obs = rng.normal(0, 1, 6)
        obs_err = np.ones(6) * 0.7
        lfactor = rng.uniform(0, 1, 6)
        np.random.seed(0)
        G = mean_preserving_rotation(12)
        W = _plain_etkf_weights(obs, obs_err, obs_prior, lfactor, G)
        np.testing.assert_allclose(W.sum(axis=0), 1.0, atol=1e-5)

    def test_rotation_preserves_analysis_covariance(self):
        """A random rotation reshuffles the anomalies but must leave the
        posterior ensemble covariance unchanged."""
        rng = np.random.default_rng(17)
        nens, nlobs = 12, 6
        obs_prior = rng.normal(0, 1, (nens, nlobs))
        obs = rng.normal(0, 1, nlobs)
        obs_err = np.ones(nlobs) * 0.7
        lfactor = rng.uniform(0, 1, nlobs)
        W0 = _plain_etkf_weights(obs, obs_err, obs_prior, lfactor, _eye(nens))
        np.random.seed(2)
        G = mean_preserving_rotation(nens)
        W1 = _plain_etkf_weights(obs, obs_err, obs_prior, lfactor, G)
        # rotation must actually change the transform
        self.assertGreater(np.max(np.abs(W0 - W1)), 1e-3)
        # but the posterior covariance of an arbitrary prior anomaly set is invariant
        A = rng.normal(0, 2, (nens, 4))
        P0 = (A.T @ W0) @ (A.T @ W0).T
        P1 = (A.T @ W1) @ (A.T @ W1).T
        np.testing.assert_allclose(P0, P1, atol=1e-8)

    def test_shared_rotation_preserves_cross_grid_covariance(self):
        """Regression test for filter divergence: two grid points with
        DIFFERENT local transforms must keep their cross-covariance when the
        SAME rotation is applied, but lose it under independent rotations.
        NEDAS therefore uses one shared rotation per analysis."""
        rng = np.random.default_rng(23)
        nens = 14
        # two grid points see different local obs -> different transforms
        opx = rng.normal(0, 1, (nens, 5)); ox = rng.normal(0, 1, 5)
        opy = rng.normal(0, 1, (nens, 4)); oy = rng.normal(0, 1, 4)
        ex = np.ones(5) * 0.6; lx = rng.uniform(0, 1, 5)
        ey = np.ones(4) * 0.6; ly = rng.uniform(0, 1, 4)
        # correlated prior anomalies at the two points
        z = rng.normal(0, 1, nens)
        ax = z + 0.3 * rng.normal(0, 1, nens)
        ay = z + 0.3 * rng.normal(0, 1, nens)
        ax -= ax.mean(); ay -= ay.mean()

        def cross_cov(Wx, Wy):
            px = ax @ Wx - (ax @ Wx).mean()
            py = ay @ Wy - (ay @ Wy).mean()
            return np.sum(px * py) / (nens - 1)

        # reference (no rotation)
        Wx0 = _plain_etkf_weights(ox, ex, opx, lx, _eye(nens))
        Wy0 = _plain_etkf_weights(oy, ey, opy, ly, _eye(nens))
        c_ref = cross_cov(Wx0, Wy0)

        # SAME rotation at both points -> cross-cov preserved
        np.random.seed(7)
        G = mean_preserving_rotation(nens)
        Wx_s = _plain_etkf_weights(ox, ex, opx, lx, G)
        Wy_s = _plain_etkf_weights(oy, ey, opy, ly, G)
        self.assertAlmostEqual(cross_cov(Wx_s, Wy_s), c_ref, places=8)

        # DIFFERENT rotations at the two points -> cross-cov corrupted
        np.random.seed(7); Gx = mean_preserving_rotation(nens)
        np.random.seed(99); Gy = mean_preserving_rotation(nens)
        Wx_d = _plain_etkf_weights(ox, ex, opx, lx, Gx)
        Wy_d = _plain_etkf_weights(oy, ey, opy, ly, Gy)
        self.assertGreater(abs(cross_cov(Wx_d, Wy_d) - c_ref), 1e-3)


class TestHybridCovariance(unittest.TestCase):
    """Hybrid ETKF-OI with nens_dynamic dynamic members and a separate batch of nens_static
    static members, P = (1-beta)*P_d + beta*alpha*P_s. H picks the first nobs state
    components; uniform obs error, so the Whitaker-Hamill reduced gain is unambiguous."""

    def setUp(self):
        rng = np.random.default_rng(31)
        self.nens_dynamic, self.nens_static, self.nobs, self.nstate = 6, 9, 3, 5
        self.ens_dynamic = rng.normal(0, 1, (self.nens_dynamic, self.nstate))  # members as rows
        self.ens_static = rng.normal(0.5, 2, (self.nens_static, self.nstate))
        self.obs = rng.normal(0, 1, self.nobs)
        self.obs_err = np.ones(self.nobs) * 0.8
        self.beta, self.alpha = 0.4, 0.3

    def _weights(self, hybrid_perturbation, beta=None, use_eigen=False):
        beta = self.beta if beta is None else beta
        fac_dynamic = np.sqrt(1 - beta) / np.sqrt(self.nens_dynamic - 1)
        fac_static = np.sqrt(beta * self.alpha) / np.sqrt(self.nens_static - 1)
        return ensemble_transform_weights(self.obs, self.obs_err,
                                          self.ens_dynamic[:, :self.nobs].copy(), self.ens_static[:, :self.nobs].copy(),
                                          np.ones(self.nobs), _eye(self.nens_dynamic), use_eigen,
                                          fac_dynamic, fac_static, hybrid_perturbation)

    def _post_dynamic(self, weights, weights_static):
        return (self.ens_dynamic.T @ weights + self.ens_static.T @ weights_static).T

    def _plain_etkf(self):
        weights = _plain_etkf_weights(self.obs, self.obs_err, self.ens_dynamic[:, :self.nobs].copy(),
                                      np.ones(self.nobs), _eye(self.nens_dynamic))
        return weights, (self.ens_dynamic.T @ weights).T

    def _gain_terms(self):
        P = (1 - self.beta) * np.cov(self.ens_dynamic.T) + self.beta * self.alpha * np.cov(self.ens_static.T)
        H = np.eye(self.nstate)[:self.nobs]
        R = np.diag(self.obs_err**2)
        return P, H, R

    def test_beta_zero_reduces_to_plain_etkf(self):
        weights_ref, _ = self._plain_etkf()
        for hybrid_perturbation in (False, True):
            weights, weights_static = self._weights(hybrid_perturbation, beta=0.0)
            np.testing.assert_allclose(weights, weights_ref, atol=1e-10)
            np.testing.assert_allclose(weights_static, 0, atol=1e-12)

    def test_mean_update_uses_hybrid_gain(self):
        P, H, R = self._gain_terms()
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        mean_dynamic = self.ens_dynamic.mean(0)
        mean_ref = mean_dynamic + K @ (self.obs - H @ mean_dynamic)
        for hybrid_perturbation in (False, True):
            for use_eigen in (False, True):
                weights, weights_static = self._weights(hybrid_perturbation, use_eigen=use_eigen)
                np.testing.assert_allclose(weights.sum(axis=0), 1.0, atol=1e-10)
                np.testing.assert_allclose(weights_static.sum(axis=0), 0.0, atol=1e-10)
                np.testing.assert_allclose(self._post_dynamic(weights, weights_static).mean(0), mean_ref, atol=1e-10)

    def test_dynamic_perturbations_ignore_static(self):
        _, post_ref = self._plain_etkf()
        post = self._post_dynamic(*self._weights(False))
        np.testing.assert_allclose(post - post.mean(0), post_ref - post_ref.mean(0), atol=1e-10)

    def test_hybrid_perturbations_use_reduced_gain(self):
        # A_d <- A_d - K~ H A_d, K~ = P H^T (sqrt(HPH^T+R))^-T (sqrt(HPH^T+R) + sqrt(R))^-1
        P, H, R = self._gain_terms()
        def sqrtm(matrix):
            eigval, eigvec = np.linalg.eigh(matrix)
            return (eigvec * np.sqrt(eigval)) @ eigvec.T
        innov_cov_sqrt = sqrtm(H @ P @ H.T + R)
        reduced_gain = P @ H.T @ np.linalg.inv(innov_cov_sqrt).T @ np.linalg.inv(innov_cov_sqrt + sqrtm(R))
        anom_dynamic = self.ens_dynamic - self.ens_dynamic.mean(0)
        post = self._post_dynamic(*self._weights(True))
        np.testing.assert_allclose(post - post.mean(0),
                                   anom_dynamic - (reduced_gain @ H @ anom_dynamic.T).T, atol=1e-10)


class TestLocalAnalysisStaticMembers(unittest.TestCase):

    def test_static_members_enter_the_dynamic_update(self):
        """local_analysis_main updates the dynamic members with weights and weights_static,
        and leaves the static members unchanged (state = the observed quantities, no localization)"""
        rng = np.random.default_rng(41)
        nens_dynamic, nens_static, nobs = 6, 9, 3
        ens_dynamic = rng.normal(0, 1, (nens_dynamic, nobs))
        ens_static = rng.normal(0.5, 2, (nens_static, nobs))
        obs, obs_err = rng.normal(0, 1, nobs), np.ones(nobs) * 0.8
        fac_dynamic = np.sqrt(0.6 / (nens_dynamic - 1))
        fac_static = np.sqrt(0.4 * 0.3 / (nens_static - 1))
        weights, weights_static = ensemble_transform_weights(obs, obs_err, ens_dynamic.copy(), ens_static.copy(),
                                                             np.ones(nobs), _eye(nens_dynamic), False,
                                                             fac_dynamic, fac_static, True)
        post_ref = (ens_dynamic.T @ weights + ens_static.T @ weights_static).T

        state_dynamic, state_static = ens_dynamic.copy(), ens_static.copy()
        zeros, ones = np.zeros(nobs), np.ones(nobs)
        local_analysis_main(state_dynamic, ens_dynamic.copy(), state_static, ens_static.copy(),
                            obs, obs_err, ones,
                            zeros, zeros, ones, gaspari_cohn_func,
                            zeros, zeros, ones, gaspari_cohn_func,
                            np.ones((nobs, nobs)), _eye(nens_dynamic), False,
                            fac_dynamic, fac_static, True)
        np.testing.assert_allclose(state_dynamic, post_ref, atol=1e-10)
        np.testing.assert_allclose(state_static, ens_static)


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
