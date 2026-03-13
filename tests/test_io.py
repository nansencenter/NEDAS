import unittest
from datetime import datetime
from NEDAS.core import Context
from NEDAS.io_backends.offline import OfflineIO
from NEDAS.io_backends.online import OnlineIO

class TestOfflineIO(unittest.TestCase):

    def test_filenames(self):
        dir_def = {'analysis_dir': './{time:%Y%m%d}'}
        time = datetime(2023, 1, 1)
        c = Context(work_dir='./', time=time, directories=dir_def, io_mode='offline')
        assert isinstance(c.io, OfflineIO)
        self.assertIsInstance(c.io, OfflineIO)
        self.assertEqual(c.io.binfile_name(c, 'prior'), './20230101/fields_prior.bin')

class TestOnlineIO(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
