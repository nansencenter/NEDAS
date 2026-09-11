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
