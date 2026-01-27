import unittest
from NEDAS.utils.random_perturb import random_perturb, parse_perturb_opts

class TestRandomPerturb(unittest.TestCase):
    """ Test if the input kwargs dict get parsed correctly into perturb_opts"""

    def test_parse_single_variable_single_scale(self):
        input_dict = {
            'variable': 'atmos_surf_temp',
            'type': 'gaussian',
            'amp': 1.0,
            'hcorr': 2e6,
            'tcorr': 48,
        }
        p_type, other_opts, params = parse_perturb_opts(**input_dict)
        self.assertEqual(p_type, 'gaussian')
        self.assertEqual(other_opts, [])
        self.assertEqual(params['atmos_surf_temp']['nscale'], 1)
        self.assertEqual(params['atmos_surf_temp']['amp'], [1.0])
        self.assertEqual(params['atmos_surf_temp']['hcorr'], [2e6])
        self.assertEqual(params['atmos_surf_temp']['tcorr'], [48])

    def test_parse_multiple_variables_single_scale(self):
        input_dict = {
            'variable': ['atmos_surf_press', 'atmos_surf_velocity'],
            'type': 'gaussian,press_wind_relate',
            'amp': [300.0, 2.0],
            'hcorr': [2e6, 2e6],
            'tcorr': [48, 48],
        }
        p_type, other_opts, params = parse_perturb_opts(**input_dict)
        self.assertEqual(p_type, 'gaussian')
        self.assertIn('press_wind_relate', other_opts)
        self.assertEqual(params['atmos_surf_velocity']['nscale'], 1)
        self.assertEqual(params['atmos_surf_velocity']['amp'], [2.0])
        self.assertEqual(params['atmos_surf_velocity']['hcorr'], [2e6])
        self.assertEqual(params['atmos_surf_velocity']['tcorr'], [48])

    def test_raise_error_when_param_size_mismatch1(self):
        input_dict = {
            'variable': ['atmos_surf_press', 'atmos_surf_velocity'],
            'type': 'gaussian,press_wind_relate',
            'amp': 300.0,
            'hcorr': 2e6,
            'tcorr': 48,
        }
        with self.assertRaises((ValueError,)):
            parse_perturb_opts(**input_dict)

    def test_raise_error_when_param_size_mismatch2(self):
        input_dict = {
            'variable': ['atmos_surf_press', 'atmos_surf_velocity'],
            'type': 'gaussian',
            'amp': [100.0, 300.0, 200.0],
            'hcorr': [2e6, 2e6, 2e6],
            'tcorr': [48, 48, 48],
        }
        with self.assertRaises((ValueError,)):
            parse_perturb_opts(**input_dict)

    def test_parse_single_variable_multi_scale(self):
        input_dict = {
            'variable': 'atmos_surf_temp',
            'type': 'gaussian',
            'amp': [1.0, 0.5],
            'hcorr': [2e6, 1e5],
            'tcorr': [48, 6],
        }
        p_type, other_opts, params = parse_perturb_opts(**input_dict)
        self.assertEqual(p_type, 'gaussian')
        self.assertEqual(other_opts, [])
        self.assertEqual(params['atmos_surf_temp']['nscale'], 2)
        self.assertEqual(params['atmos_surf_temp']['amp'], [1.0, 0.5])
        self.assertEqual(params['atmos_surf_temp']['hcorr'], [2e6, 1e5])
        self.assertEqual(params['atmos_surf_temp']['tcorr'], [48, 6])

    def test_parse_multi_variable_multi_scale(self):
        input_dict = {
            'variable': ['atmos_surf_temp', 'atmos_surf_velocity'],
            'type': 'gaussian',
            'amp': [[1.0, 0.5], [1, 2]],
            'hcorr': [[2e6, 1e5], [2e6, 1e5]],
            'tcorr': [[120, 48], [120, 48]],
        }
        p_type, other_opts, params = parse_perturb_opts(**input_dict)
        self.assertEqual(p_type, 'gaussian')
        self.assertEqual(params['atmos_surf_temp']['nscale'], 2)
        self.assertEqual(params['atmos_surf_temp']['amp'], [1.0, 0.5])
        self.assertEqual(params['atmos_surf_temp']['hcorr'], [2e6, 1e5])
        self.assertEqual(params['atmos_surf_temp']['tcorr'], [120, 48])
        self.assertEqual(params['atmos_surf_velocity']['nscale'], 2)
        self.assertEqual(params['atmos_surf_velocity']['amp'], [1, 2])
        self.assertEqual(params['atmos_surf_velocity']['hcorr'], [2e6, 1e5])
        self.assertEqual(params['atmos_surf_velocity']['tcorr'], [120, 48])
