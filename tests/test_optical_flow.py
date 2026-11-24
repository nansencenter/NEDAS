import numpy as np
import unittest
from NEDAS.grid import Grid
from NEDAS.utils.random_perturb import random_field_gaussian
from NEDAS.utils.spatial_operation import warp
from NEDAS.utils.optical_flow import OpticalFlow


class TestOpticalFlowConstantShift(unittest.TestCase):
    """Every backend recovers a known uniform shift with the same sign convention."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        cls.grid = Grid.regular_grid(None, 0, 100, 0, 100, 1, centered=True, cyclic_dim='xy')
        cls.img1 = random_field_gaussian(100, 100, 1, 10)
        shape = cls.grid.x.shape
        cls.img2 = warp(cls.grid, cls.img1, np.full(shape, 3.0), np.full(shape, -2.0))

    def check(self, method, **kwargs):
        flow = OpticalFlow(method, **kwargs)(self.grid, self.img1, self.img2)
        inner = (slice(20, 80), slice(20, 80))
        self.assertAlmostEqual(flow[0][inner].mean(), 3.0, delta=0.3)
        self.assertAlmostEqual(flow[1][inner].mean(), -2.0, delta=0.3)

    def test_dis(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest('opencv not installed')
        self.check('DIS')

    def test_raft(self):
        try:
            import torchvision  # noqa: F401
        except ImportError:
            self.skipTest('torchvision not installed')
        self.check('RAFT')
        self.check('RAFT', normalize='uint8')


if __name__ == '__main__':
    unittest.main()
