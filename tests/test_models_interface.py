import unittest
from NEDAS.core import Model, Context
from NEDAS.models import registry, get_model_class

class TestModelInterface(unittest.TestCase):
    def setUp(self):
        self.c = Context()

    def test_model_class_init(self):
        for model_name in registry.keys():
            ModelClass = get_model_class(model_name)
            model = ModelClass(runtime=self.c.rt)
            self.assertIsInstance(model, Model)

if __name__ == '__main__':
    unittest.main()
