"""
Scalar (SSPE) parameter persistence across process boundaries (offline mode).

Offline-mode runs execute each step and each ensemble member in a separate
process with no shared memory, so Model.write_param/read_param must round-trip
through an on-disk store (work_dir/param/<model>/_memXXX.json). Online mode
keeps params in process memory and must not create those files.

Regression test for the offline SSPE no-op fix (2026-08-18).
"""
import os
import unittest
import tempfile
import shutil
from typing import cast
from NEDAS.config import Config
from NEDAS.models.lorenz96.lorenz96_model import Lorenz96Model

CONFIG_FILE = os.path.join(os.path.dirname(__file__), '..', 'examples', 'lorenz96', 'config.yml')


def build_offline_config(work_dir, nens=5):
    config = Config(config_file=CONFIG_FILE, quiet=True)
    config.io_mode = 'offline'
    config.nens = nens
    config.work_dir = work_dir
    assert config.model_def is not None
    config.model_def['lorenz96']['ens_init_dir'] = os.path.join(work_dir, 'init_ens')
    config.model_def['lorenz96']['truth_dir'] = os.path.join(work_dir, 'truth')
    config.model_def['lorenz96']['F_std'] = 1.5
    return config


class TestScalarParamPersistence(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='nedas_sspe_')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_offline_param_roundtrips_across_contexts(self):
        """A param written in one context (process) is read by a fresh context."""
        from NEDAS import get_scheme

        work_dir = os.path.join(self.tmpdir, 'work')
        config = build_offline_config(work_dir)

        model = cast(Lorenz96Model, get_scheme(config).c.models['lorenz96'])
        self.assertEqual(model.io_mode, 'offline')
        model.write_param(8.7, name='F', member=3)
        model.write_param(6.4, name='F', member=0)

        # fresh context = a later subprocess; memory is empty, file must carry values
        fresh_model = get_scheme(config).c.models['lorenz96']
        self.assertEqual(fresh_model.read_param(name='F', member=3), 8.7)
        self.assertEqual(fresh_model.read_param(name='F', member=0), 6.4)
        # members with no written value fall back to the model default
        self.assertEqual(fresh_model.read_param(name='F', member=1), model.F)

        # file lives in the documented store location
        store = os.path.join(work_dir, 'param', 'lorenz96')
        self.assertTrue(os.path.exists(os.path.join(store, '_mem004.json')))

    def test_online_param_stays_in_memory_no_files(self):
        """Online mode is unchanged: memory-backed, no on-disk store."""
        from NEDAS import get_scheme

        work_dir = os.path.join(self.tmpdir, 'work')
        config = build_offline_config(work_dir)
        config.io_mode = 'online'

        model = get_scheme(config).c.models['lorenz96']
        model.write_param(9.9, name='F', member=2)
        self.assertEqual(model.read_param(name='F', member=2), 9.9)

        store = os.path.join(work_dir, 'param', 'lorenz96')
        self.assertFalse(os.path.exists(store), 'online mode should not write param files')


if __name__ == '__main__':
    unittest.main()
