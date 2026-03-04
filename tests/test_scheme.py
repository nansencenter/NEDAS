import unittest
from NEDAS.config import Config
from NEDAS.schemes import get_scheme

class TestAnalysisScheme(unittest.TestCase):
    def test_raise_exception_when_not_implemented(self):        
        with self.assertRaises(NotImplementedError):
            cf = Config(scheme='foo')
            get_scheme(cf)

if __name__ == '__main__':
    unittest.main()
