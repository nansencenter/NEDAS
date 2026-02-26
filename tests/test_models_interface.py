import unittest
from NEDAS.core import Model
from NEDAS.core.model import registry, get_model_class

class TestModelInterface(unittest.TestCase):
    def test_model_class_init(self):
        for model_name in registry.keys():
            ModelClass = get_model_class(model_name)
            model = ModelClass()
            self.assertIsInstance(model, Model)

if __name__ == '__main__':
    unittest.main()
