import numpy as np
import unittest
from NEDAS.assim_tools.assimilators.ETKF.core import (
    ensemble_transform_weights, apply_ensemble_transform,
    mean_preserving_rotation,
)


def _eye(nens):
    return np.eye(nens)


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
        W = ensemble_transform_weights(obs, obs_err, obs_prior, lfactor, _eye(10))
        self.assertEqual(W.shape, (10, 10))

    def test_column_sums_equal_one(self):
        obs, obs_err, obs_prior, lfactor = self._make_data(nens=10, nlobs=5)
        W = ensemble_transform_weights(obs, obs_err, obs_prior, lfactor, _eye(10))
        np.testing.assert_allclose(W.sum(axis=0), 1.0, atol=1e-5)

    def test_huge_obs_error_gives_near_identity(self):
        nens = 8
        rng = np.random.default_rng(3)
        obs_prior = rng.normal(0, 1, (nens, 1))
        obs = np.array([0.0])
        obs_err = np.array([1e6])
        lfactor = np.ones(1)
        W = ensemble_transform_weights(obs, obs_err, obs_prior, lfactor, _eye(nens))
        np.testing.assert_allclose(W, np.eye(nens), atol=1e-3)

    def test_posterior_spread_not_greater_than_prior(self):
        rng = np.random.default_rng(5)
        nens = 20
        obs_prior = rng.normal(0, 2, (nens, 3))
        obs = rng.normal(0, 1, 3)
        obs_err = np.ones(3)
        lfactor = np.ones(3)
        W = ensemble_transform_weights(obs, obs_err, obs_prior, lfactor, _eye(nens))
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
        W_svd = ensemble_transform_weights(obs, obs_err, obs_prior, lfactor, _eye(nens), False)
        W_eig = ensemble_transform_weights(obs, obs_err, obs_prior, lfactor, _eye(nens), True)
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
        W = ensemble_transform_weights(obs, obs_err, obs_prior, lfactor, G)
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
        W0 = ensemble_transform_weights(obs, obs_err, obs_prior, lfactor, _eye(nens))
        np.random.seed(2)
        G = mean_preserving_rotation(nens)
        W1 = ensemble_transform_weights(obs, obs_err, obs_prior, lfactor, G)
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
        Wx0 = ensemble_transform_weights(ox, ex, opx, lx, _eye(nens))
        Wy0 = ensemble_transform_weights(oy, ey, opy, ly, _eye(nens))
        c_ref = cross_cov(Wx0, Wy0)

        # SAME rotation at both points -> cross-cov preserved
        np.random.seed(7)
        G = mean_preserving_rotation(nens)
        Wx_s = ensemble_transform_weights(ox, ex, opx, lx, G)
        Wy_s = ensemble_transform_weights(oy, ey, opy, ly, G)
        self.assertAlmostEqual(cross_cov(Wx_s, Wy_s), c_ref, places=8)

        # DIFFERENT rotations at the two points -> cross-cov corrupted
        np.random.seed(7); Gx = mean_preserving_rotation(nens)
        np.random.seed(99); Gy = mean_preserving_rotation(nens)
        Wx_d = ensemble_transform_weights(ox, ex, opx, lx, Gx)
        Wy_d = ensemble_transform_weights(oy, ey, opy, ly, Gy)
        self.assertGreater(abs(cross_cov(Wx_d, Wy_d) - c_ref), 1e-3)


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
