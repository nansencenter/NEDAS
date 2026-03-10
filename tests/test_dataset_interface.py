import unittest
from NEDAS.core import Dataset, Context
from NEDAS.datasets import registry, get_dataset_class

class TestDatasetInterface(unittest.TestCase):
    def test_dataset_class_init(self):
        c = Context()
        for dataset_name in registry.keys():
            DatasetClass = get_dataset_class(dataset_name)
            dataset = DatasetClass(grid=c.grid, mask=c.grid.mask)
            self.assertIsInstance(dataset, Dataset)

if __name__ == '__main__':
    unittest.main()
