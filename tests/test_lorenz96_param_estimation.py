"""
Test simultaneous state and parameter estimation (SSPE) for the Lorenz-96 model.

The ensemble of the forcing parameter F is initialized from N(F_true, F_std).
After several DA cycles the ensemble spread in F should decrease (observations
constrain F indirectly through the state-parameter covariance).
"""
import os
import sys
import unittest
import tempfile
import shutil
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

CONFIG_FILE = os.path.join(os.path.dirname(__file__), '..', 'examples', 'lorenz96', 'config.yml')


def run_sspe_scheme(tmpdir, nens=20, F_std=1.5, hroi=12, ncycles=3):
    """Run the lorenz96 DA scheme with F as a scalar state variable.

    Returns (F_prior_ensemble, F_post_ensemble) after all cycles.
    """
    from NEDAS.config import Config
    from NEDAS import get_scheme
    from NEDAS.utils.conversion import dt1h

    config = Config(config_file=CONFIG_FILE, quiet=True)
    config.nens = nens
    config.obs_def[0]['hroi'] = hroi

    # redirect work dir to temp directory so there's no cached data
    config.work_dir = tmpdir
    config.model_def['lorenz96']['ens_init_dir'] = os.path.join(tmpdir, 'init_ens')
    config.model_def['lorenz96']['truth_dir']    = os.path.join(tmpdir, 'truth')
    # set F_std so generate_init_ensemble perturbs F per member
    config.model_def['lorenz96']['F_std'] = F_std

    # add F as a scalar state variable
    config.state_def = list(config.state_def) + [
        {'name': 'F', 'model_src': 'lorenz96', 'var_type': 'scalar', 'err_type': 'normal'}
    ]

    # shorten the cycling window for speed
    cycle_period = 24  # hours (one L96 model unit = 120 h, 24h = 0.2 model units)
    from datetime import datetime, timezone, timedelta
    t0 = config.time_analysis_start
    config.time_analysis_end = t0 + timedelta(hours=cycle_period * ncycles)
    config.time_end = config.time_analysis_end + timedelta(hours=cycle_period)

    scheme = get_scheme(config)
    model = scheme.c.models['lorenz96']

    scheme()

    F_post = np.array([model.read_param(name='F', member=m) for m in range(nens)])
    return F_post


class TestLorenz96ParameterEstimation(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='nedas_sspe_test_')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_scalar_param_infrastructure(self):
        """StateInfo registers F as a scalar; scalars appear in pack data."""
        from NEDAS.config import Config
        from NEDAS import get_scheme
        import tempfile, shutil

        td = tempfile.mkdtemp(prefix='nedas_sspe_infra_')
        try:
            config = Config(config_file=CONFIG_FILE, quiet=True)
            config.nens = 5
            config.work_dir = td
            config.model_def['lorenz96']['ens_init_dir'] = os.path.join(td, 'init_ens')
            config.model_def['lorenz96']['truth_dir']    = os.path.join(td, 'truth')
            config.model_def['lorenz96']['F_std'] = 0.5
            config.state_def = list(config.state_def) + [
                {'name': 'F', 'model_src': 'lorenz96', 'var_type': 'scalar', 'err_type': 'normal'}
            ]

            scheme = get_scheme(config)
            c = scheme.c

            # StateInfo should have one scalar registered
            from NEDAS.core.state import State
            from NEDAS.core.state_info import StateInfo
            info = StateInfo(c)
            self.assertEqual(len(info.scalars), 1, "Expected 1 scalar (F) in StateInfo")
            rec = info.scalars[0]
            self.assertEqual(rec.name, 'F')
            self.assertEqual(rec.model_src, 'lorenz96')

            # generate truth + init ensemble so model memory is populated
            scheme.prepare_truth()
            scheme.prepare_init_ensemble()

            # all members should have distinct F values
            model = c.models['lorenz96']
            F_vals = [model.read_param(name='F', member=m) for m in range(config.nens)]
            self.assertGreater(np.std(F_vals), 0.0,
                               "Expected F ensemble spread > 0 after generate_init_ensemble with F_std=0.5")
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_F_ensemble_updated_after_da(self):
        """F ensemble posterior differs from the uniform-F prior."""
        from NEDAS.config import Config
        from NEDAS import get_scheme

        config = Config(config_file=CONFIG_FILE, quiet=True)
        config.nens = 20
        config.work_dir = self.tmpdir
        config.model_def['lorenz96']['ens_init_dir'] = os.path.join(self.tmpdir, 'init_ens')
        config.model_def['lorenz96']['truth_dir']    = os.path.join(self.tmpdir, 'truth')
        config.model_def['lorenz96']['F_std'] = 1.5  # initial F_m ~ N(8, 1.5)
        config.obs_def[0]['hroi'] = 12

        config.state_def = list(config.state_def) + [
            {'name': 'F', 'model_src': 'lorenz96', 'var_type': 'scalar', 'err_type': 'normal'}
        ]

        # 3 DA cycles
        from datetime import timedelta
        config.time_analysis_end = config.time_analysis_start + timedelta(hours=24 * 3)
        config.time_end = config.time_analysis_end + timedelta(hours=24)

        scheme = get_scheme(config)
        model = scheme.c.models['lorenz96']

        # collect initial F before any DA
        scheme.prepare_truth()
        scheme.prepare_init_ensemble()
        F_init = np.array([model.read_param(name='F', member=m)
                           for m in range(config.nens)])

        # run the DA cycles
        scheme.run_all()

        F_post = np.array([model.read_param(name='F', member=m)
                           for m in range(config.nens)])

        # posterior F should differ from initial (DA updated the ensemble)
        F_changed = np.sum(np.abs(F_post - F_init) > 1e-10)
        self.assertGreater(F_changed, 0,
                           "No F member was updated after DA — SSPE did not modify F")

        # ensemble F should remain in a physically reasonable range
        F_post_mean = float(np.mean(F_post))
        self.assertGreater(F_post_mean, 2.0, f"F mean {F_post_mean:.2f} suspiciously low")
        self.assertLess(F_post_mean, 20.0, f"F mean {F_post_mean:.2f} suspiciously high")


if __name__ == '__main__':
    unittest.main()
