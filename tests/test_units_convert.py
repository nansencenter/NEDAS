import numpy as np
import unittest
from NEDAS.utils.conversion import units_convert

class TestUnitsConvert(unittest.TestCase):

    def test_units_convert_scalar(self):
        self.assertEqual(units_convert(100, 1, 1.), 0.01)
        self.assertEqual(units_convert('km', 'm', 1.), 1000.)
        self.assertEqual(units_convert('h', 's', 1.), 3600.)
        self.assertEqual(units_convert('m/s', 'km/h', 1.), 3.6)
        self.assertEqual(units_convert('kg', 'g', 1.), 1000.)
        self.assertEqual(units_convert('m/s', 'm/s', 1.), 1.)
        self.assertEqual(units_convert('C', 'K', 0.), 273.15)

    def test_units_convert_field(self):
        shape = (100, 100)
        self.assertEqual(np.mean(units_convert(100, 1, np.ones(shape))), 0.01)
        self.assertAlmostEqual(np.mean(units_convert('km', 'm', np.ones(shape))), 1000.)


if __name__ == '__main__':
    unittest.main()
