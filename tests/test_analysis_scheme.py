import unittest
from NEDAS.config import Config
from NEDAS.schemes.get_analysis_scheme import get_analysis_scheme
from NEDAS.schemes.base import AnalysisScheme

class TestAnalysisScheme(unittest.TestCase):
    def test_analysis_scheme_init(self):
        c = Config()
        scheme = get_analysis_scheme(c.analysis_scheme)
        self.assertIsInstance(scheme, AnalysisScheme)

    def test_raise_exception_when_not_implemented(self):        
        with self.assertRaises(NotImplementedError):
            get_analysis_scheme('foo')

if __name__ == '__main__':
    unittest.main()
