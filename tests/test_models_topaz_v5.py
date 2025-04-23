import unittest
import numpy as np
from datetime import datetime

from NEDAS.models.topaz.v5 import Model
from NEDAS.models.topaz.confmap import ConformalMapping


class TestTopaz5Model(unittest.TestCase):
    def test_model_class_init_grid(self):
        model = Model()
        self.assertEqual(model.grid.nx, model.idm)
        self.assertEqual(model.grid.ny, model.jdm)
        self.assertIsInstance(model.grid.proj, ConformalMapping)

