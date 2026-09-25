"""
Tests for NEDAS.diag.metrics.non_gaussianity.
"""
import unittest
import numpy as np
from NEDAS.diag.metrics.non_gaussianity import (
    skewness, excess_kurtosis, negentropy, summarize_non_gaussianity,
)


class TestSkewness(unittest.TestCase):
    def test_gaussian_skew_near_zero(self):
        rng = np.random.default_rng(0)
        ens = rng.normal(0, 1, (5000, 4, 5))
        sk = skewness(ens)
        self.assertEqual(sk.shape, (4, 5))
        self.assertTrue(np.all(np.abs(sk) < 0.2))

    def test_lognormal_positive_skew(self):
        rng = np.random.default_rng(1)
        ens = rng.lognormal(0, 1, (5000, 1))
        self.assertGreater(skewness(ens)[0], 1.0)

    def test_left_skewed_negative(self):
        rng = np.random.default_rng(2)
        ens = -rng.lognormal(0, 0.7, (5000, 1))
        self.assertLess(skewness(ens)[0], -1.0)

    def test_zero_variance_is_zero_not_nan(self):
        ens = np.ones((20, 3))
        np.testing.assert_array_equal(skewness(ens), 0.0)


class TestExcessKurtosis(unittest.TestCase):
    def test_gaussian_kurtosis_near_zero(self):
        rng = np.random.default_rng(3)
        ens = rng.normal(0, 1, (5000, 4))
        self.assertTrue(np.all(np.abs(excess_kurtosis(ens)) < 0.5))

    def test_lognormal_leptokurtic(self):
        rng = np.random.default_rng(4)
        ens = rng.lognormal(0, 1, (5000, 1))
        self.assertGreater(excess_kurtosis(ens)[0], 5.0)

    def test_bimodal_platykurtic(self):
        # symmetric two-hump mixture: near-zero skew, excess kurtosis ~ -2
        rng = np.random.default_rng(5)
        half = 2500
        ens = np.concatenate([rng.normal(-3, 0.3, (half, 1)),
                              rng.normal(3, 0.3, (half, 1))])
        self.assertTrue(np.abs(skewness(ens)[0]) < 0.3)
        self.assertLess(excess_kurtosis(ens)[0], -1.5)

    def test_zero_variance_is_zero(self):
        ens = np.ones((20, 3))
        np.testing.assert_array_equal(excess_kurtosis(ens), 0.0)


class TestNegentropy(unittest.TestCase):
    def test_gaussian_negentropy_near_zero(self):
        rng = np.random.default_rng(6)
        ens = rng.normal(0, 1, (5000, 4, 4))
        ng = negentropy(ens)
        self.assertEqual(ng.shape, (4, 4))
        self.assertTrue(np.all(ng >= 0))
        self.assertTrue(np.all(ng < 0.2))

    def test_lognormal_negentropy_positive(self):
        rng = np.random.default_rng(7)
        ens = rng.lognormal(0, 1, (5000, 1))
        self.assertGreater(negentropy(ens)[0], 0.2)

    def test_skewed_larger_than_gaussian(self):
        rng = np.random.default_rng(8)
        gauss = rng.normal(0, 1, (5000, 1))
        skew = rng.lognormal(0, 1, (5000, 1))
        self.assertGreater(negentropy(skew)[0], negentropy(gauss)[0] + 0.1)

    def test_zero_variance_is_zero(self):
        ens = np.ones((20, 3))
        np.testing.assert_array_equal(negentropy(ens), 0.0)

    def test_too_few_members_raises(self):
        with self.assertRaises(ValueError):
            negentropy(np.ones((2, 3)))


class TestMaskAndSummary(unittest.TestCase):
    def test_masked_column_stays_nan(self):
        rng = np.random.default_rng(9)
        ens = rng.normal(0, 1, (100, 2, 3))
        ens[:, 1, :] = np.nan  # masked row (whole columns masked)
        for fn in (skewness, excess_kurtosis, negentropy):
            out = fn(ens)
            self.assertTrue(np.all(np.isnan(out[1, :])))
            self.assertTrue(np.all(np.isfinite(out[0, :])))

    def test_summarize_scalars_and_shapes(self):
        rng = np.random.default_rng(10)
        ens = np.concatenate([rng.lognormal(0, 1, (5000, 2)),
                              rng.normal(0, 1, (5000, 2))], axis=1)
        summ = summarize_non_gaussianity(ens)
        self.assertEqual(set(summ), {'skewness', 'excess_kurtosis', 'negentropy'})
        for v in summ.values():
            self.assertTrue(np.isfinite(v))
        # lognormal column dominates the means -> all clearly positive
        self.assertGreater(summ['skewness'], 0.1)
        self.assertGreater(summ['excess_kurtosis'], 0.5)
        self.assertGreater(summ['negentropy'], 0.1)


if __name__ == '__main__':
    unittest.main()
