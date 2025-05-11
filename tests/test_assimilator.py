import unittest
import importlib
from NEDAS.config import Config
from NEDAS.assim_tools.assimilators import registry, get_assimilator_class

class TestAnalysisScheme(unittest.TestCase):
    def test_assimilator_init(self):
        c = Config()
        for assimilator_name in registry.keys():
            c.assimilator_def['type'] = assimilator_name
            module = importlib.import_module('NEDAS.assim_tools.assimilators.'+assimilator_name)
            Assimilator = get_assimilator_class(assimilator_name)
            assimilator = Assimilator(c)
            self.assertIsInstance(assimilator, getattr(module, registry[assimilator_name]))

    def test_raise_exception_when_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            get_assimilator_class('foo')

    def test_assimilation_algorithm_implemented(self):
        c = Config()
        for assimilator_name in registry.keys():
            c.assimilator_def['type'] = assimilator_name
            Assimilator = get_assimilator_class(assimilator_name)
            assimilator = Assimilator(c)

            self.assertTrue(hasattr(assimilator, 'assimilation_algorithm'), "Method 'assimilation_algorithm' not found")

            method = getattr(assimilator, 'assimilation_algorithm')
            self.assertTrue(callable(method), "'assimilation_algorithm' is not callable")

if __name__ == '__main__':
    unittest.main()
