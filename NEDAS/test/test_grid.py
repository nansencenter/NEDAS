import numpy as np
import unittest
from pyproj import Proj
from NEDAS.grid import Grid

class TestGrid(unittest.TestCase):
    def test_regular_grid_creation(self):
        proj = Proj('+proj=stere')
        xstart, xend, ystart, yend = 0, 100, 0, 100
        dx = 1
        grid = Grid.regular_grid(proj, xstart, xend, ystart, yend, dx)
        self.assertEqual(grid.nx, 100)
        self.assertEqual(grid.ny, 100)
        self.assertAlmostEqual(grid.dx, 1.0)
        self.assertAlmostEqual(grid.dy, 1.0)

    def test_random_grid_creation(self):
        proj = Proj('+proj=stere')
        xstart, xend, ystart, yend = 0, 100, 0, 100
        npoints = 1000
        grid = Grid.random_grid(proj, xstart, xend, ystart, yend, npoints)
        self.assertEqual(grid.x.size, npoints)

    def test_find_index_regular(self):
        proj = Proj('+proj=stere')
        xstart, xend, ystart, yend = 0, 100, 0, 100
        dx = 1
        grid = Grid.regular_grid(proj, xstart, xend, ystart, yend, dx)

        ##case 1: find one of the grid point
        x, y = 10, 10
        inside, _, vertices, in_coords, nearest = grid.find_index(x, y)
        self.assertTrue(inside[0])
        self.assertTrue((vertices[0] == np.array([1010, 1011, 1111, 1110])).all())
        self.assertTrue((in_coords[0] == np.array([0., 0.])).all())
        self.assertEqual(nearest[0], 1010)

        ##case 2: find a point inside the grid
        x, y = 10.8, 10.2
        inside, _, vertices, in_coords, nearest = grid.find_index(x, y)
        self.assertTrue(inside[0])
        self.assertTrue((vertices[0] == np.array([1010, 1011, 1111, 1110])).all())
        self.assertAlmostEqual(in_coords[0,0], 0.8)
        self.assertAlmostEqual(in_coords[0,1], 0.2)
        self.assertEqual(nearest[0], 1011)

        ##case 3: find a point outside the grid
        x, y = -1, -1
        inside, _, _, _, _ = grid.find_index(x, y)
        self.assertFalse(inside[0])

    def test_find_index_irregular(self):
        proj = Proj('+proj=stere')
        x, y = np.meshgrid(np.arange(10), np.arange(10))
        grid = Grid(proj, x.flatten(), y.flatten(), regular=False)

        ##case 1: find a point inside the irregular mesh
        x, y = 1.2, 2.7
        inside, indices, vertices, in_coords, nearest = grid.find_index(x, y)
        self.assertTrue(inside[0])
        self.assertEqual(indices[0], 22)
        self.assertTrue((vertices[0] == np.array([31, 21, 32])).all())
        self.assertAlmostEqual(in_coords[0,0], 0.5)
        self.assertAlmostEqual(in_coords[0,1], 0.3)
        self.assertAlmostEqual(in_coords[0,2], 0.2)
        self.assertEqual(nearest[0], 31)

    def test_cyclic_dim(self):
        proj = Proj('+proj=stere')
        xstart, xend, ystart, yend = 0, 100, 0, 100
        dx = 1
        grid = Grid.regular_grid(proj, xstart, xend, ystart, yend, dx, cyclic_dim='xy')

        ##a point outside grid boundary will still be found "inside"
        x, y = -1, -1
        inside, _, vertices, in_coords, nearest = grid.find_index(x, y)
        self.assertTrue(inside[0])
        self.assertTrue((vertices[0] == np.array([9999, 9900, 0, 99])).all())
        self.assertTrue((in_coords[0] == np.array([0., 0.])).all())
        self.assertEqual(nearest[0], 9999)

    def test_pole_dim(self):
        grid1 = Grid.regular_grid(Proj('+proj=longlat'), -180, 181, 0, 91, 1, cyclic_dim='x')  ##lon lat grid
        grid2 = Grid.regular_grid(Proj('+proj=stere +lat_0=90'), -1e6, 1e6, -1e6, 1e6, 1e4, centered=True)  ##polar stereographic grid

        ##a vector field with all zonal winds
        u = np.ones(grid1.x.shape)
        v = np.zeros(grid1.x.shape)
        vfld1 = np.array([u, v])

        ##case 1: pole_dim is not set, there will be nan after rotating vectors
        grid1.set_destination_grid(grid2)
        vfld2 = grid1.convert(vfld1, is_vector=True)
        self.assertTrue(np.isnan(vfld2[0, 100, 100]))

        ##case 2: when pole_dim is set, the void will be filled, so nan is gone
        grid1.pole_dim='y'
        grid1.pole_index=(-1,)
        grid1.set_destination_grid(grid2)
        vfld2 = grid1.convert(vfld1, is_vector=True)
        self.assertFalse(np.isnan(vfld2).any())

    def test_rotate_vector(self):
        grid1 = Grid.regular_grid(Proj('+proj=stere +lat_0=90 +lon_0=0'), -1e6, 1e6, -1e6, 1e6, 1e4)
        grid2 = Grid.regular_grid(Proj('+proj=stere +lat_0=90 +lon_0=90'), -1e6, 1e6, -1e6, 1e6, 1e4)
        u = np.ones(grid1.x.shape)
        v = np.zeros(grid1.x.shape)
        vfld1 = np.array([u, v])

        ##at domain center, vfld1 is (1, 0), after rotating to grid2, vfld2 should be (0, -1)
        grid1.set_destination_grid(grid2)
        vfld2 = grid1.convert(vfld1, is_vector=True)
        self.assertAlmostEqual(vfld2[0, 100, 100], 0.0)
        self.assertAlmostEqual(vfld2[1, 100, 100], -1.0)

if __name__ == '__main__':
    unittest.main()

