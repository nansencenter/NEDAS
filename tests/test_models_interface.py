import unittest
from NEDAS.models import registry, get_model_class, Model

class TestModelInterface(unittest.TestCase):
    def test_model_class_inheritance(self):
        for model_name in registry.keys():
            ModelClass = get_model_class(model_name)
            self.assertTrue(issubclass(ModelClass, Model))

if __name__ == '__main__':
    unittest.main()
