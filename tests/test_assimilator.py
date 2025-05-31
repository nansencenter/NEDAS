import unittest
import importlib
from NEDAS.config import Config
from NEDAS.assim_tools.assimilators import registry, get_assimilator

class TestAnalysisScheme(unittest.TestCase):
    def test_assimilator_init(self):
        c = Config()
        for assimilator_name in registry.keys():
            c.assimilator_def['type'] = assimilator_name
            module = importlib.import_module('NEDAS.assim_tools.assimilators.'+assimilator_name)
            assimilator = get_assimilator(c)
            self.assertIsInstance(assimilator, getattr(module, registry[assimilator_name]))

    def test_raise_exception_when_not_implemented(self):
        c = Config()
        with self.assertRaises(NotImplementedError):
            c.assimilator_def['type'] = 'foo'
            get_assimilator(c)

    def test_assimilation_algorithm_implemented(self):
        c = Config()
        for assimilator_name in registry.keys():
            c.assimilator_def['type'] = assimilator_name
            assimilator = get_assimilator(c)

            self.assertTrue(hasattr(assimilator, 'assimilation_algorithm'), "Method 'assimilation_algorithm' not found")

            method = getattr(assimilator, 'assimilation_algorithm')
            self.assertTrue(callable(method), "'assimilation_algorithm' is not callable")

if __name__ == '__main__':
    unittest.main()
