import numpy as np
import unittest
from NEDAS.utils.spatial_operation import (
    gradx, grady, gradx2, grady2, laplacian,
    coarsen_field, sharpen_field,
)


class TestGradx(unittest.TestCase):

    def test_linear_field_non_cyclic(self):
        nx, ny = 10, 5
        dx = 1.0
        fld = np.tile(np.arange(nx, dtype=float), (ny, 1))
        gx = gradx(fld, dx)
        # interior and right edge: df/dx = 1
        np.testing.assert_allclose(gx[:, 1:-1], 1.0, atol=1e-12)
        np.testing.assert_allclose(gx[:, 0], 1.0, atol=1e-12)
        np.testing.assert_allclose(gx[:, -1], 1.0, atol=1e-12)

    def test_constant_field_zero_gradient(self):
        fld = np.ones((4, 6))
        np.testing.assert_allclose(gradx(fld, 1.0), 0.0, atol=1e-12)

    def test_cyclic_sin_field(self):
        nx = 32
        dx = 2 * np.pi / nx
        x = np.arange(nx) * dx
        fld = np.sin(x)[np.newaxis, :]
        gx = gradx(fld, dx, cyclic_dim='x')
        np.testing.assert_allclose(gx[0], np.cos(x), atol=1e-2)

    def test_output_shape_matches_input(self):
        fld = np.random.default_rng(0).random((3, 7, 11))
        self.assertEqual(gradx(fld, 1.0).shape, fld.shape)


class TestGrady(unittest.TestCase):

    def test_linear_field_non_cyclic(self):
        ny, nx = 10, 5
        dy = 1.0
        fld = np.tile(np.arange(ny, dtype=float)[:, np.newaxis], (1, nx))
        gy = grady(fld, dy)
        np.testing.assert_allclose(gy[1:-1, :], 1.0, atol=1e-12)

    def test_cyclic_cos_field(self):
        ny = 32
        dy = 2 * np.pi / ny
        y = np.arange(ny) * dy
        fld = np.cos(y)[:, np.newaxis]
        gy = grady(fld, dy, cyclic_dim='y')
        np.testing.assert_allclose(gy[:, 0], -np.sin(y), atol=1e-2)


class TestLaplacian(unittest.TestCase):

    def test_quadratic_interior(self):
        # f = x^2 + y^2, Laplacian = 2+2 = 4 at interior points
        # Skip [0:2, :] and [:, 0:2] boundaries where one-sided differences propagate
        n = 20
        dx = dy = 1.0
        x, y = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
        fld = x**2 + y**2
        lap = laplacian(fld, dx, dy)
        np.testing.assert_allclose(lap[2:-2, 2:-2], 4.0, atol=1e-10)

    def test_constant_field_zero_laplacian(self):
        fld = np.ones((8, 8))
        np.testing.assert_allclose(laplacian(fld, 1.0, 1.0), 0.0, atol=1e-12)


class TestGradx2(unittest.TestCase):

    def test_quadratic_second_derivative(self):
        # f = x^2 → d^2f/dx^2 = 2 at deep interior points
        # col=1 is adjacent to the one-sided boundary at col=0, so skip it
        n = 20
        fld = np.tile((np.arange(n, dtype=float)**2)[np.newaxis, :], (5, 1))
        g2 = gradx2(fld, 1.0)
        np.testing.assert_allclose(g2[:, 2:-2], 2.0, atol=1e-10)


class TestCoarsenField(unittest.TestCase):

    def test_uniform_field_unchanged(self):
        fld = np.ones((4, 4))
        result = coarsen_field(fld, 0, 1)
        np.testing.assert_allclose(result, np.ones((2, 2)))

    def test_averaging_correctness(self):
        # 2x2 blocks with known values
        fld = np.array([[1., 2., 3., 4.],
                        [5., 6., 7., 8.],
                        [9., 10., 11., 12.],
                        [13., 14., 15., 16.]])
        result = coarsen_field(fld, 0, 1)
        expected = np.array([[(1+2+5+6)/4, (3+4+7+8)/4],
                              [(9+10+13+14)/4, (11+12+15+16)/4]])
        np.testing.assert_allclose(result, expected)

    def test_no_change_same_level(self):
        fld = np.random.default_rng(1).random((4, 4))
        result = coarsen_field(fld, 2, 2)
        np.testing.assert_array_equal(result, fld)

    def test_shape_after_coarsen(self):
        fld = np.ones((8, 8))
        result = coarsen_field(fld, 0, 2)
        self.assertEqual(result.shape, (2, 2))


class TestSharpenField(unittest.TestCase):

    def test_no_change_same_level(self):
        fld = np.random.default_rng(2).random((4, 4))
        result = sharpen_field(fld, 1, 1)
        np.testing.assert_array_equal(result, fld)

    def test_shape_after_sharpen(self):
        fld = np.ones((2, 2))
        result = sharpen_field(fld, 2, 1)
        self.assertEqual(result.shape, (4, 4))

    def test_uniform_field_stays_uniform(self):
        fld = np.ones((2, 2)) * 5.0
        result = sharpen_field(fld, 2, 1)
        np.testing.assert_allclose(result, 5.0)

    def test_coarsen_then_sharpen_roundtrip(self):
        # coarsen then sharpen should recover the shape, not necessarily exact values
        fld = np.random.default_rng(3).random((4, 4))
        coarsened = coarsen_field(fld, 0, 1)
        sharpened = sharpen_field(coarsened, 2, 1)
        self.assertEqual(sharpened.shape, fld.shape)


if __name__ == '__main__':
    unittest.main()
