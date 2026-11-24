import unittest
import numpy as np
from pyproj import Proj
from NEDAS.grid import Grid
from NEDAS.assim_tools.assimilators.serial import BoxBins


class TestBoxBins(unittest.TestCase):
    """The candidates hold every point within r, on plain and cyclic grids, near the edges too."""

    def check(self, cyclic_dim):
        grid = Grid.regular_grid(Proj('+proj=stere'), 0, 1000, 0, 600, 10, cyclic_dim=cyclic_dim)
        rng = np.random.default_rng(0)
        x, y = rng.uniform(0, 1000, 3000), rng.uniform(0, 600, 3000)
        x[:5] = np.nan      # invalid points are never candidates
        bins = BoxBins(grid, x, y, 150)
        for r in (30, 150, 400, 2000):
            for rx, ry in rng.uniform((0, 0), (1000, 600), (50, 2)):
                cand = bins.candidates(rx, ry, r)
                near = np.where(grid.distance(rx, x, ry, y) <= r)[0]
                self.assertTrue(set(near) <= set(cand), (cyclic_dim, r, rx, ry))
                self.assertEqual(len(cand), len(set(cand)))

    def test_plain(self):
        self.check(None)

    def test_cyclic(self):
        self.check('xy')

    def test_no_box_gives_all(self):
        grid = Grid.regular_grid(Proj('+proj=stere'), 0, 100, 0, 100, 10)
        x = np.arange(5.)
        self.assertEqual(list(BoxBins(grid, x, x, np.inf).candidates(0, 0, 10)), list(range(5)))
        self.assertEqual(list(BoxBins(grid, x, x, 10).candidates(0, 0, np.inf)), list(range(5)))


if __name__ == '__main__':
    unittest.main()
