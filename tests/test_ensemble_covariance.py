import numpy as np
import unittest
from types import SimpleNamespace
from NEDAS.assim_tools.covariance import get_covariance
from NEDAS.assim_tools.covariance.ensemble import ensemble_covariance


class TestGetCovariance(unittest.TestCase):
    def _get(self, nens=10, **covariance_def):
        return get_covariance(SimpleNamespace(nens=nens, covariance_def=covariance_def))

    def test_legacy_config_loads_as_pure_ensemble(self):
        cov = self._get(type='ensemble', config_file=None)
        self.assertEqual((cov.beta, cov.nens_static), (0.0, 0))

    def test_unknown_legacy_type_raises(self):
        with self.assertRaises(NotImplementedError):
            self._get(type='static')

    def test_invalid_settings_raise(self):
        for kwargs in ({'beta': 1.5, 'nens_static': 5},   # beta outside [0, 1]
                       {'beta': 0.5},                     # beta > 0 without static members
                       {'alpha': 0.0},
                       {'nens': 1, 'beta': 0.5, 'nens_static': 5}):  # hybrid needs 2 dynamic members
            with self.assertRaises(ValueError, msg=kwargs):
                self._get(**kwargs)


class TestEnsembleCovariance(unittest.TestCase):
    """ensemble_covariance expects zero-mean anomaly arrays as input
    (it sums x^2 directly, without subtracting the mean)."""

    def _zero_mean(self, arr):
        return arr - arr.mean(axis=0)

    def test_output_shapes_1d(self):
        rng = np.random.default_rng(0)
        nens, nstate, nobs = 10, 4, 3
        state_ens = self._zero_mean(rng.normal(0, 1, (nens, nstate)))
        obs_ens   = self._zero_mean(rng.normal(0, 1, (nens, nobs)))
        sv, ov, corr = ensemble_covariance(state_ens, obs_ens)
        self.assertEqual(sv.shape, (nstate,))
        self.assertEqual(ov.shape, (nobs,))
        self.assertEqual(corr.shape, (nstate, nobs))

    def test_variance_is_positive(self):
        rng = np.random.default_rng(1)
        nens = 20
        state_ens = self._zero_mean(rng.normal(0, 2, (nens, 5)))
        obs_ens   = self._zero_mean(rng.normal(0, 1, (nens, 3)))
        sv, ov, _ = ensemble_covariance(state_ens, obs_ens)
        self.assertTrue(np.all(sv > 0))
        self.assertTrue(np.all(ov > 0))

    def test_correlation_in_range(self):
        rng = np.random.default_rng(2)
        nens = 15
        state_ens = self._zero_mean(rng.normal(0, 1, (nens, 4)))
        obs_ens   = self._zero_mean(rng.normal(0, 1, (nens, 4)))
        _, _, corr = ensemble_covariance(state_ens, obs_ens)
        self.assertTrue(np.all(corr >= -1.0 - 1e-10))
        self.assertTrue(np.all(corr <=  1.0 + 1e-10))

    def test_perfect_correlation_for_identical_variables(self):
        rng = np.random.default_rng(3)
        nens = 20
        base = self._zero_mean(rng.normal(0, 1, (nens, 1)))
        _, _, corr = ensemble_covariance(base, base)
        np.testing.assert_allclose(corr[0, 0], 1.0, atol=1e-10)

    def test_variance_formula_matches_numpy(self):
        rng = np.random.default_rng(4)
        nens = 30
        state_ens = self._zero_mean(rng.normal(0, 3, (nens, 6)))
        obs_ens   = self._zero_mean(rng.normal(0, 1, (nens, 2)))
        sv, ov, _ = ensemble_covariance(state_ens, obs_ens)
        # numpy var with ddof=1 on zero-mean data ≡ sum(x^2)/(nens-1)
        np.testing.assert_allclose(sv, np.var(state_ens, axis=0, ddof=1), atol=1e-10)
        np.testing.assert_allclose(ov, np.var(obs_ens,   axis=0, ddof=1), atol=1e-10)

    def test_mismatched_ensemble_sizes_raises(self):
        state_ens = np.zeros((10, 3))
        obs_ens   = np.zeros((8,  3))
        with self.assertRaises(ValueError):
            ensemble_covariance(state_ens, obs_ens)

    def test_anticorrelated_gives_negative_correlation(self):
        rng = np.random.default_rng(5)
        nens = 20
        base = self._zero_mean(rng.normal(0, 1, (nens, 1)))
        _, _, corr = ensemble_covariance(base, -base)
        np.testing.assert_allclose(corr[0, 0], -1.0, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
