import unittest
import os
from datetime import datetime
from NEDAS.core import Context

class TestFileSystem(unittest.TestCase):

    def test_directory_names(self):
        dir_def = {'cycle_dir': '{work_dir}/cycle/{time:%Y%m%d}',
                   'analysis_dir': '{work_dir}/cycle/{time:%Y%m%d}/analysis',
                   'forecast_dir': '{work_dir}/cycle/{time:%Y%m%d}/{model_name}',}
        time = datetime(2023, 1, 1)
        cwd = os.getcwd()
        c = Context(work_dir='test', time=time, directories=dir_def)
        self.assertEqual(c.fs.cycle_dir(time), os.path.join(cwd, 'test', 'cycle', '20230101'))
        self.assertEqual(c.fs.analysis_dir(time), os.path.join(cwd, 'test', 'cycle', '20230101', 'analysis'))
        self.assertEqual(c.fs.forecast_dir(time, 'qg'), os.path.join(cwd, 'test', 'cycle', '20230101', 'qg'))

if __name__ == '__main__':
    unittest.main()
