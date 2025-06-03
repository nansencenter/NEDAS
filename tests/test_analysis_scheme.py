import unittest
from NEDAS.config import Config
from NEDAS.schemes.get_analysis_scheme import get_analysis_scheme

class TestAnalysisScheme(unittest.TestCase):
    def test_raise_exception_when_not_implemented(self):        
        with self.assertRaises(NotImplementedError):
            c = Config(analysis_scheme='foo')
            get_analysis_scheme(c)

if __name__ == '__main__':
    unittest.main()
