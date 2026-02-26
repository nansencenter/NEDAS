import unittest
import os
from datetime import datetime
from NEDAS.config import Config
from NEDAS.core import Context
from NEDAS.io_backends.file_io import FileIO

class TestFileIO(unittest.TestCase):

    def test_directory_names(self):
        dir_def = {'cycle_dir': '{work_dir}/cycle/{time:%Y%m%d}',
                   'analysis_dir': '{work_dir}/cycle/{time:%Y%m%d}/analysis',
                   'forecast_dir': '{work_dir}/cycle/{time:%Y%m%d}/{model_name}',}
        time = datetime(2023, 1, 1)
        cwd = os.getcwd()
        cf = Config(work_dir='test', time=time, directories=dir_def)
        c = Context(cf)
        io = FileIO(c)
        self.assertEqual(io.cycle_dir(time), os.path.join(cwd, 'test', 'cycle', '20230101'))
        self.assertEqual(io.analysis_dir(time), os.path.join(cwd, 'test', 'cycle', '20230101', 'analysis'))
        self.assertEqual(io.forecast_dir(time, 'qg'), os.path.join(cwd, 'test', 'cycle', '20230101', 'qg'))

    def test_filenames(self):
        dir_def = {'analysis_dir': './{time:%Y%m%d}'}
        time = datetime(2023, 1, 1)
        cwd = os.getcwd()
        cf = Config(work_dir='./', time=time, directories=dir_def)
        c = Context(cf)
        io = FileIO(c)
        self.assertEqual(io.prior_file, '.')

if __name__ == '__main__':
    unittest.main()
