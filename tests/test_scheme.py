import unittest
from NEDAS.schemes import get_scheme

class TestAnalysisScheme(unittest.TestCase):
    def test_raise_exception_when_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            get_scheme(scheme='foo')

if __name__ == '__main__':
    unittest.main()
