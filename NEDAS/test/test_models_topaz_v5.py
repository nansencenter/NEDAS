import unittest
import numpy as np
from datetime import datetime

from models.topaz.v5 import Model
from models.topaz.confmap import ConformalMapping


class TestTopaz5Model(unittest.TestCase):
    def test_model_class_init_grid(self):
        model = Model()
        self.assertEqual(model.grid.nx, model.idm)
        self.assertEqual(model.grid.ny, model.jdm)
        self.assertIsInstance(model.grid.proj, ConformalMapping)
        
    def test_model_get_filename(self):
        model = Model()
        path = '/path/to/work/dir'
        member = 0
        time = datetime(2021, 9, 6, 12)
        vname = 'ocean_temp'
        fname = model.filename(path=path, name=vname, time=time, member=member)
        self.assertEqual(fname, '/path/to/work/dir/restart.2021_249_12_0000_mem001.a')
