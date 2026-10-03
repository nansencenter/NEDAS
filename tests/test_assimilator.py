import unittest
import importlib
from NEDAS.core import Context
from NEDAS.assim_tools.assimilators import registry, get_assimilator
from NEDAS.assim_tools.covariance import Covariance

class TestAnalysisScheme(unittest.TestCase):
    def setUp(self):
        self.c = Context()


    def test_assimilator_init(self):
        for assimilator_name in registry.keys():
            self.c.config.assimilator_def = {}
            self.c.config.assimilator_def['type'] = assimilator_name
            module = importlib.import_module('NEDAS.assim_tools.assimilators.'+assimilator_name)
            assimilator = get_assimilator(self.c)
            self.assertIsInstance(assimilator, getattr(module, registry[assimilator_name]))

    def test_check_capabilities(self):
        # the pure ensemble covariance is supported by every assimilator
        for assimilator_name in registry.keys():
            self.c.config.assimilator_def = {'type': assimilator_name}
            get_assimilator(self.c).check_capabilities(self.c)

        # a hybrid covariance: ETKF, EAKF and TopazDEnKF support it, QCEF lists exactly the unsupported settings
        self.c.covariance = Covariance(self.c.nens, beta=0.5, nens_static=5, hybrid_perturbation=True)
        for assimilator_name in ('ETKF', 'EAKF', 'TopazDEnKF'):
            self.c.config.assimilator_def = {'type': assimilator_name}
            get_assimilator(self.c).check_capabilities(self.c)
        self.c.config.assimilator_def = {'type': 'QCEF'}
        with self.assertRaises(NotImplementedError) as err:
            get_assimilator(self.c).check_capabilities(self.c)
        for name in ('nens_static', 'hybrid_perturbation'):
            self.assertIn(name, str(err.exception))

    def test_raise_exception_when_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.c.config.assimilator_def = {}
            self.c.config.assimilator_def['type'] = 'foo'
            get_assimilator(self.c)

    def test_assimilation_algorithm_implemented(self):
        for assimilator_name in registry.keys():
            self.c.config.assimilator_def = {}
            self.c.config.assimilator_def['type'] = assimilator_name
            assimilator = get_assimilator(self.c)

            self.assertTrue(hasattr(assimilator, 'assimilation_algorithm'), "Method 'assimilation_algorithm' not found")

            method = getattr(assimilator, 'assimilation_algorithm')
            self.assertTrue(callable(method), "'assimilation_algorithm' is not callable")

if __name__ == '__main__':
    unittest.main()


class TestSerialBatchEquivalence(unittest.TestCase):
    """
    The serial and the batch analysis agree exactly when nothing is localized, and differ
    once something is.

    The first half is a correctness statement worth pinning: with a linear observation
    operator and no localization, assimilating observations one at a time and assimilating
    them simultaneously are the same deterministic square-root analysis, so the two
    strategies must land on the same posterior mean. If that ever breaks, one of the two
    implementations has drifted.

    The second half pins the gap as expected rather than as a defect. It has two causes, and
    neither is a choice of localization weight: the EAKF tapers the regression of the
    observation increment onto the state (the Kalman gain) while the ETKF tapers R, which is
    a different operation at any power; and a sequence of tapered single-observation updates
    is not one tapered simultaneous update (Nerger, 2015). Both ETKF settings of
    loc_weight_sqrt are checked, so neither can be mistaken for closing the gap.
    """
    NENS, NOBS = 60, 4

    def analyses(self, weights):
        import numpy as np
        from NEDAS.assim_tools.assimilators.EAKF.core import obs_increment_eakf, update_ensemble
        from NEDAS.assim_tools.assimilators.ETKF.core import (
            ensemble_transform_weights, apply_ensemble_transform)

        nens, nobs = self.NENS, self.NOBS
        rng = np.random.default_rng(11)
        prior = rng.normal(0, 1, (nens, 1))
        obs_prior = rng.normal(0, 1, (nens, nobs))
        obs = rng.normal(0, 1, nobs)
        err = np.full(nobs, 0.7)
        no_static_state, no_static_obs = np.zeros((0, 1)), np.zeros((0, nobs))
        fac = 1.0 / np.sqrt(nens - 1)

        # serial: one observation at a time, with the obs priors kept consistent as the
        # serial loop does, so the comparison is against the real algorithm
        X, Y = prior.copy(), obs_prior.copy()
        for j in range(nobs):
            incr = obs_increment_eakf(Y[:, j], np.zeros(0), obs[j], err[j], 1.0, 0.0, False)
            X = update_ensemble(X, no_static_state, Y[:, j], np.zeros(0), incr,
                                np.full((1,), weights[j]), None, 1.0, 0.0, False)
            for k in range(nobs):
                if k != j:
                    Y[:, k] = update_ensemble(Y[:, k][:, None], np.zeros((0, 1)), Y[:, j],
                                              np.zeros(0), incr, np.array([weights[j]]),
                                              None, 1.0, 0.0, False)[:, 0]
        serial = X.mean()

        batch = {}
        for use_sqrt in (False, True):
            lfactor = np.sqrt(weights) if use_sqrt else weights
            wt, _ = ensemble_transform_weights(obs, err, obs_prior.copy(), no_static_obs,
                                               lfactor, np.eye(nens), False, fac, 0.0, False)
            batch[use_sqrt] = apply_ensemble_transform(prior.copy()[:, 0], wt).mean()
        return serial, batch

    def test_agree_without_localization(self):
        import numpy as np
        serial, batch = self.analyses(np.ones(self.NOBS))
        for use_sqrt, value in batch.items():
            self.assertAlmostEqual(serial, value, places=10,
                                   msg=f'serial vs batch (loc_weight_sqrt={use_sqrt}) at w=1')

    def test_differ_with_localization(self):
        import numpy as np
        serial, batch = self.analyses(np.array([1.0, 0.6, 0.3, 0.1]))
        for use_sqrt, value in batch.items():
            self.assertGreater(abs(serial - value), 1e-4,
                               msg=f'loc_weight_sqrt={use_sqrt} unexpectedly reproduces the '
                                   'serial analysis under localization')


class TestSerialOverrideSignatures(unittest.TestCase):
    """serial.py calls update_local_state/obs with a fixed argument list; every override must
    accept it."""

    def test_overrides_match_base(self):
        import inspect
        from NEDAS.assim_tools.assimilators.serial import SerialAssimilator
        from NEDAS.assim_tools.assimilators.EAKF.core import EAKFAssimilator
        for name in ('update_local_state', 'update_local_obs'):
            want = list(inspect.signature(getattr(SerialAssimilator, name)).parameters)
            # QCEF is left out: it is an unfinished stub whose overrides and kernel calls predate
            # this interface, and is not a working assimilator to hold to it yet
            for cls in (EAKFAssimilator,):
                got = list(inspect.signature(getattr(cls, name)).parameters)
                self.assertEqual(got, want, f"{cls.__name__}.{name}")
