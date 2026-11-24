"""
Tests for NEDAS.diag.metrics.sea_ice.
"""
import unittest
import numpy as np
from NEDAS.diag.metrics.sea_ice import iiee


class TestIIEE(unittest.TestCase):
    def test_perfect_agreement_is_zero(self):
        fld = np.array([0.0, 0.05, 0.2, 0.9, 1.0])
        self.assertEqual(iiee(fld, fld.copy()), 0.0)

    def test_counts_disagreeing_cells(self):
        fld    = np.array([0.9, 0.9, 0.05, 0.05])
        fld_tr = np.array([0.9, 0.05, 0.9, 0.05])
        # cells 1 and 2 disagree (one says ice, other says no-ice)
        self.assertEqual(iiee(fld, fld_tr), 2.0)

    def test_threshold_boundary(self):
        fld    = np.array([0.16, 0.14])
        fld_tr = np.array([0.14, 0.14])
        # only the first cell crosses the default 0.15 threshold differently
        self.assertEqual(iiee(fld, fld_tr, threshold=0.15), 1.0)
        # with a threshold both sides sit below, no disagreement
        self.assertEqual(iiee(fld, fld_tr, threshold=0.5), 0.0)

    def test_cell_area_scales_result(self):
        fld    = np.array([0.9, 0.05])
        fld_tr = np.array([0.05, 0.9])
        self.assertEqual(iiee(fld, fld_tr), 2.0)
        self.assertEqual(iiee(fld, fld_tr, cell_area=100.0), 200.0)

    def test_nan_cells_excluded(self):
        fld    = np.array([np.nan, 0.9, 0.05])
        fld_tr = np.array([0.9,    0.9, 0.9])
        # cell 0 is masked in fld -> never counted even though fld_tr says ice
        # cell 2 disagrees (0.05 vs 0.9)
        self.assertEqual(iiee(fld, fld_tr), 1.0)

    def test_2d_field(self):
        fld    = np.array([[0.9, 0.05], [0.9, 0.9]])
        fld_tr = np.array([[0.9, 0.9],  [0.05, 0.9]])
        self.assertEqual(iiee(fld, fld_tr), 2.0)


if __name__ == '__main__':
    unittest.main()
