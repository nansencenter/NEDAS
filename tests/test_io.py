import os
import unittest
from datetime import datetime
from NEDAS.core import Context
from NEDAS.io_backends.offline import OfflineIO
from NEDAS.io_backends.online import OnlineIO

class TestOfflineIO(unittest.TestCase):

    def setUp(self):
        time = datetime(2023, 1, 1)
        self.c = Context(work_dir='.', time=time, io_mode='offline')

    def test_io_instance(self):
        self.assertIsInstance(self.c.io, OfflineIO)

    def test_binfile_name(self):
        cwd = os.getcwd()
        assert isinstance(self.c.io, OfflineIO)
        binfile = os.path.join(cwd, 'cycle', '202301010000', 'analysis', 'fields_prior.bin')
        self.assertEqual(self.c.io.binfile_name(self.c, 'prior'), binfile)

class TestOnlineIO(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
