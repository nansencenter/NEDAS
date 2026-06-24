import numpy as np
import unittest
from NEDAS.grid.grid_1d import Grid1D


class TestGrid1DRegularGrid(unittest.TestCase):

    def setUp(self):
        self.g = Grid1D.regular_grid(0.0, 10.0, 1.0)

    def test_nx(self):
        self.assertEqual(self.g.nx, 10)

    def test_dx(self):
        self.assertAlmostEqual(self.g.dx, 1.0)

    def test_xmin_xmax(self):
        self.assertAlmostEqual(self.g.xmin, 0.0)
        self.assertAlmostEqual(self.g.xmax, 9.0)

    def test_regular_flag(self):
        self.assertTrue(self.g.regular)

    def test_x_values(self):
        np.testing.assert_allclose(self.g.x, np.arange(0.0, 10.0, 1.0))

    def test_y_is_zeros(self):
        np.testing.assert_array_equal(self.g.y, 0.0)


class TestGrid1DFindIndex(unittest.TestCase):

    def setUp(self):
        self.g = Grid1D.regular_grid(0.0, 5.0, 1.0)

    def test_exact_grid_points_found_as_inside(self):
        # query at an interior grid point
        inside, vertices, in_coords, nearest = self.g.find_index(np.array([2.5]))
        self.assertTrue(inside[0])

    def test_outside_range_not_inside(self):
        inside, _, _, _ = self.g.find_index(np.array([-1.0, 10.0]))
        self.assertFalse(any(inside))

    def test_midpoint_vertices(self):
        # between x=1 and x=2, vertices should be indices 1 and 2
        inside, vertices, in_coords, nearest = self.g.find_index(np.array([1.5]))
        self.assertTrue(inside[0])
        self.assertAlmostEqual(in_coords[0], 0.5)


class TestGrid1DInterp(unittest.TestCase):

    def setUp(self):
        self.g = Grid1D.regular_grid(0.0, 5.0, 1.0)  # x = [0,1,2,3,4]

    def test_linear_interp_on_linear_function(self):
        fld = self.g.x * 2.0  # f(x) = 2x
        x_query = np.array([1.5, 2.5, 3.5])
        result = self.g.interp(fld, x=x_query, method='linear')
        np.testing.assert_allclose(result, x_query * 2.0, atol=1e-12)

    def test_nearest_interp(self):
        fld = np.array([0., 1., 2., 3., 4.])
        # x=1.3 should snap to x=1 (index 1)
        result = self.g.interp(fld, x=np.array([1.3]), method='nearest')
        np.testing.assert_allclose(result, [1.0], atol=1e-12)

    def test_out_of_range_returns_nan(self):
        fld = self.g.x.copy()
        result = self.g.interp(fld, x=np.array([10.0]), method='linear')
        self.assertTrue(np.isnan(result[0]))

    def test_identity_same_grid(self):
        # dst_grid == source grid by default: convert should return same fld
        fld = np.sin(self.g.x)
        result = self.g.convert(fld)
        np.testing.assert_array_equal(result, fld)


class TestGrid1DDistance(unittest.TestCase):

    def test_non_cyclic_distance(self):
        g = Grid1D.regular_grid(0.0, 10.0, 1.0, cyclic=False)
        x = np.array([0.0, 3.0, 7.0])
        d = g.distance(ref_x=5.0, x=x)
        np.testing.assert_allclose(d, np.abs(x - 5.0))

    def test_cyclic_distance_wraps(self):
        # domain [0,10), Lx = 10*1=10
        g = Grid1D.regular_grid(0.0, 10.0, 1.0, cyclic=True)
        # distance from 0.5 to 9.5 should be 1.0 (via wrap), not 9.0
        d = g.distance(ref_x=0.5, x=np.array([9.5]))
        np.testing.assert_allclose(d, [1.0], atol=1e-10)


class TestGrid1DEquality(unittest.TestCase):

    def test_same_grid_equal(self):
        g1 = Grid1D.regular_grid(0.0, 5.0, 1.0)
        g2 = Grid1D.regular_grid(0.0, 5.0, 1.0)
        self.assertEqual(g1, g2)

    def test_different_grid_not_equal(self):
        g1 = Grid1D.regular_grid(0.0, 5.0, 1.0)
        g2 = Grid1D.regular_grid(0.0, 10.0, 1.0)
        self.assertNotEqual(g1, g2)


if __name__ == '__main__':
    unittest.main()
