import unittest
import importlib
from NEDAS.config import Config
from NEDAS.core import Context
from NEDAS.io_backends import get_io_backend
from NEDAS.assim_tools.assimilators import registry, get_assimilator

class TestAnalysisScheme(unittest.TestCase):
    def setUp(self):
        cf = Config()
        self.c = Context(cf)
        self.c.io = get_io_backend(self.c)

    def test_assimilator_init(self):
        for assimilator_name in registry.keys():
            self.c.config.assimilator_def = {}
            self.c.config.assimilator_def['type'] = assimilator_name
            module = importlib.import_module('NEDAS.assim_tools.assimilators.'+assimilator_name)
            assimilator = get_assimilator(self.c)
            self.assertIsInstance(assimilator, getattr(module, registry[assimilator_name]))

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
