"""
Checks on the DART interface (DART's filter_main with its I/O in NEDAS memory).

The full check is a cycled run against the native EAKF, which is identical on Lorenz-96;
these only catch a DART version whose filter_mod no longer takes the patch, and a library
missing its C entry points. Set NEDAS_DART_DIR to the DART checkout.
"""
import os
import ctypes
import unittest
from NEDAS.assim_tools.assimilators.DART import patch_filter_mod
from NEDAS.assim_tools.assimilators.DART.core import default_lib_path

DART = os.environ.get('NEDAS_DART_DIR', os.path.expanduser('~/code/DART'))
FILTER_MOD = os.path.join(DART, 'assimilation_code/modules/assimilation/filter_mod.f90')


@unittest.skipUnless(os.path.exists(FILTER_MOD), f"no DART checkout at {DART}")
class TestPatch(unittest.TestCase):
    def test_patch_applies(self):
        with open(FILTER_MOD) as f:
            out = patch_filter_mod.patch(f.read())
        for hook in ('nedas_read_state', 'nedas_write_state', 'nedas_read_obs_seq',
                     'nedas_obs_ens_distrib_state', 'nedas_write_obs_seq'):
            self.assertIn(f'call {hook}(', out)
        self.assertNotIn('call read_state(', out)

    def test_patch_refuses_unknown_source(self):
        with self.assertRaises(RuntimeError):
            patch_filter_mod.patch('module filter_mod\nend module filter_mod\n')


@unittest.skipUnless(os.path.exists(default_lib_path()), "libdartfilter.so not built")
class TestLibrary(unittest.TestCase):
    def test_entry_points(self):
        lib = ctypes.CDLL(default_lib_path())
        for name in ('dart_filter_set_state', 'dart_filter_set_obs', 'dart_filter_set_posterior',
                     'dart_filter_set_periodic', 'dart_filter_set_write_obs_seq', 'dart_filter_run'):
            self.assertTrue(hasattr(lib, name), name)


if __name__ == '__main__':
    unittest.main()
