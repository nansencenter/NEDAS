"""
Tests for NEDAS.diag.metrics.ensemble.
"""
import unittest
import numpy as np
from NEDAS.diag.metrics.ensemble import (
    spread, crps, mean_crps, brier_score,
)


class TestSpread(unittest.TestCase):
    def test_zero_spread_identical_members(self):
        ens = np.ones((10, 3, 3))
        self.assertEqual(spread(ens), 0.0)

    def test_known_spread(self):
        rng = np.random.default_rng(0)
        ens = rng.normal(0, 2.0, (20000, 1))
        self.assertAlmostEqual(spread(ens), 2.0, delta=0.05)


class TestCRPS(unittest.TestCase):
    def test_perfect_deterministic_ensemble_matches_mae(self):
        # all members identical == deterministic forecast, CRPS reduces to MAE
        tr = np.array([0.0, 1.0])
        ens = np.stack([np.array([2.0, 2.0])] * 5)
        np.testing.assert_allclose(crps(ens, tr), np.abs(np.array([2.0, 2.0]) - tr))
        self.assertAlmostEqual(mean_crps(ens, tr), np.mean(np.abs(np.array([2.0, 2.0]) - tr)))

    def test_crps_matches_pairwise_definition(self):
        # cross-check against the brute-force pairwise CRPS formula
        rng = np.random.default_rng(1)
        ens = rng.normal(0, 1, (30, 5))
        tr = rng.normal(0, 1, (5,))
        term1 = np.mean(np.abs(ens - tr[np.newaxis, :]), axis=0)
        term2 = np.mean(np.abs(ens[:, np.newaxis, :] - ens[np.newaxis, :, :]), axis=(0, 1))
        expected = term1 - 0.5 * term2
        np.testing.assert_allclose(crps(ens, tr), expected, atol=1e-10)


class TestBrierScore(unittest.TestCase):
    def test_perfect_forecast_is_zero(self):
        tr = np.array([1.0, -1.0])
        ens = np.stack([np.array([1.0, -1.0])] * 10)
        self.assertEqual(brier_score(ens, tr, threshold=0.0), 0.0)

    def test_maximally_wrong_is_one(self):
        tr = np.array([1.0])
        ens = np.stack([np.array([-1.0])] * 10)
        self.assertEqual(brier_score(ens, tr, threshold=0.0), 1.0)


if __name__ == '__main__':
    unittest.main()
