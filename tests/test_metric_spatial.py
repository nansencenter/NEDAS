"""
Tests for NEDAS.diag.metrics.spatial.
"""
import unittest
import numpy as np
from NEDAS.diag.metrics.spatial import rmse, pattern_corr


class TestRMSE(unittest.TestCase):
    def test_perfect_match_is_zero(self):
        fld = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(rmse(fld, fld.copy()), 0.0)

    def test_known_offset(self):
        fld = np.full((2, 2), 3.0)
        tr = np.zeros((2, 2))
        self.assertAlmostEqual(rmse(fld, tr), 3.0)


class TestPatternCorr(unittest.TestCase):
    def test_perfect_positive_correlation(self):
        tr = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(pattern_corr(tr.copy(), tr), 1.0)

    def test_perfect_negative_correlation(self):
        tr = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(pattern_corr(-tr, tr), -1.0)

    def test_constant_field_is_nan(self):
        tr = np.ones(4)
        self.assertTrue(np.isnan(pattern_corr(tr.copy(), tr)))


if __name__ == '__main__':
    unittest.main()
