import unittest
from NEDAS.grid import Grid
from NEDAS.core.perturb import PerturbField

class TestPerturbParseConfig(unittest.TestCase):
    """ Test if the input kwargs dict get parsed correctly into perturb_opts"""
    def setUp(self):
        self.grid = Grid.regular_grid(None, 1, 100, 1, 100, 1)

    def test_parse_single_variable_single_scale(self):
        input_dict = {
            'variable': 'atmos_surf_temp',
            'type': 'gaussian',
            'amp': 1.0,
            'hcorr': 2e6,
            'tcorr': 48,
        }
        p = PerturbField(**input_dict, grid=self.grid)
        self.assertEqual(p.perturb_type, 'gaussian')
        self.assertEqual(p.other_opts, [])
        self.assertEqual(p.params['atmos_surf_temp']['nscale'], 1)
        self.assertEqual(p.params['atmos_surf_temp']['amp'], [1.0])
        self.assertEqual(p.params['atmos_surf_temp']['hcorr'], [2e6])
        self.assertEqual(p.params['atmos_surf_temp']['tcorr'], [48])

    def test_parse_multiple_variables_single_scale(self):
        input_dict = {
            'variable': ['atmos_surf_press', 'atmos_surf_velocity'],
            'type': 'gaussian,press_wind_relate',
            'amp': [300.0, 2.0],
            'hcorr': [2e6, 2e6],
            'tcorr': [48, 48],
        }
        p = PerturbField(**input_dict, grid=self.grid)
        self.assertEqual(p.perturb_type, 'gaussian')
        self.assertIn('press_wind_relate', p.other_opts)
        self.assertEqual(p.params['atmos_surf_velocity']['nscale'], 1)
        self.assertEqual(p.params['atmos_surf_velocity']['amp'], [2.0])
        self.assertEqual(p.params['atmos_surf_velocity']['hcorr'], [2e6])
        self.assertEqual(p.params['atmos_surf_velocity']['tcorr'], [48])

    def test_raise_error_when_param_size_mismatch1(self):
        input_dict = {
            'variable': ['atmos_surf_press', 'atmos_surf_velocity'],
            'type': 'gaussian,press_wind_relate',
            'amp': 300.0,
            'hcorr': 2e6,
            'tcorr': 48,
        }
        with self.assertRaises((ValueError,)):
            PerturbField(**input_dict, grid=self.grid)

    def test_raise_error_when_param_size_mismatch2(self):
        input_dict = {
            'variable': ['atmos_surf_press', 'atmos_surf_velocity'],
            'type': 'gaussian',
            'amp': [100.0, 300.0, 200.0],
            'hcorr': [2e6, 2e6, 2e6],
            'tcorr': [48, 48, 48],
        }
        with self.assertRaises((ValueError,)):
            PerturbField(**input_dict, grid=self.grid)

    def test_parse_single_variable_multi_scale(self):
        input_dict = {
            'variable': 'atmos_surf_temp',
            'type': 'gaussian',
            'amp': [1.0, 0.5],
            'hcorr': [2e6, 1e5],
            'tcorr': [48, 6],
        }
        p = PerturbField(**input_dict, grid=self.grid)
        self.assertEqual(p.perturb_type, 'gaussian')
        self.assertEqual(p.other_opts, [])
        self.assertEqual(p.params['atmos_surf_temp']['nscale'], 2)
        self.assertEqual(p.params['atmos_surf_temp']['amp'], [1.0, 0.5])
        self.assertEqual(p.params['atmos_surf_temp']['hcorr'], [2e6, 1e5])
        self.assertEqual(p.params['atmos_surf_temp']['tcorr'], [48, 6])

    def test_parse_multi_variable_multi_scale(self):
        input_dict = {
            'variable': ['atmos_surf_temp', 'atmos_surf_velocity'],
            'type': 'gaussian',
            'amp': [[1.0, 0.5], [1, 2]],
            'hcorr': [[2e6, 1e5], [2e6, 1e5]],
            'tcorr': [[120, 48], [120, 48]],
        }
        p = PerturbField(**input_dict, grid=self.grid)
        self.assertEqual(p.perturb_type, 'gaussian')
        self.assertEqual(p.params['atmos_surf_temp']['nscale'], 2)
        self.assertEqual(p.params['atmos_surf_temp']['amp'], [1.0, 0.5])
        self.assertEqual(p.params['atmos_surf_temp']['hcorr'], [2e6, 1e5])
        self.assertEqual(p.params['atmos_surf_temp']['tcorr'], [120, 48])
        self.assertEqual(p.params['atmos_surf_velocity']['nscale'], 2)
        self.assertEqual(p.params['atmos_surf_velocity']['amp'], [1, 2])
        self.assertEqual(p.params['atmos_surf_velocity']['hcorr'], [2e6, 1e5])
        self.assertEqual(p.params['atmos_surf_velocity']['tcorr'], [120, 48])

if __name__ == '__main__':
    unittest.main()
