import unittest
import os
import tempfile
from datetime import datetime, timezone
from NEDAS.config import Config

class TestConfig(unittest.TestCase):
    def test_convert_work_dir_to_abspath(self):
        c = Config()
        self.assertEqual(c.work_dir, os.path.abspath(c.work_dir))

    def test_nedas_root_available(self):
        c = Config()
        self.assertTrue(os.path.exists(c.nedas_root))

    def test_config_time_variable_type(self):
        c = Config()
        self.assertIsInstance(c.time, datetime)
        self.assertIsInstance(c.time_start, datetime)
        self.assertIsInstance(c.time_end, datetime)
        self.assertIsInstance(c.time_analysis_start, datetime)
        self.assertIsInstance(c.time_analysis_end, datetime)
        self.assertIsInstance(c.prev_time, datetime)
        self.assertIsInstance(c.next_time, datetime)

    def test_prev_next_time_arithmetic(self):
        tzinfo = timezone.utc
        time_start = datetime(2022, 1, 1, tzinfo=tzinfo)
        time = datetime(2023, 1, 1, tzinfo=tzinfo)
        cycle_period = 24
        c = Config(time_start=time_start, time=time, cycle_period=cycle_period)
        self.assertEqual(c.prev_time, datetime(2022, 12, 31, tzinfo=tzinfo))
        self.assertEqual(c.next_time, datetime(2023, 1, 2, tzinfo=tzinfo))

    def test_raise_exception_if_time_type_error(self):
        with self.assertRaises(TypeError):
            c = Config()
            c.time='2023-01-01'

    def test_argparse_time_str(self):
        tzinfo = timezone.utc
        c = Config(time='2001-01-01 00:00:00')
        self.assertEqual(c.time, datetime(2001,1,1, tzinfo=tzinfo))
        c = Config(time='20010101000000')
        self.assertEqual(c.time, datetime(2001,1,1, tzinfo=tzinfo))
        c = Config(time='2001-01-01T00:00:00Z')
        self.assertEqual(c.time, datetime(2001,1,1,tzinfo=tzinfo))

    def test_directory_names(self):
        dir_def = {'cycle_dir': '{work_dir}/cycle/{time:%Y%m%d}',
                   'analysis_dir': '{work_dir}/cycle/{time:%Y%m%d}/analysis',
                   'forecast_dir': '{work_dir}/cycle/{time:%Y%m%d}/{model_name}',}
        time = datetime(2023, 1, 1)
        cwd = os.getcwd()
        c = Config(work_dir='test', time=time, directories=dir_def)
        self.assertEqual(c.cycle_dir(time), os.path.join(cwd, 'test', 'cycle', '20230101'))
        self.assertEqual(c.analysis_dir(time), os.path.join(cwd, 'test', 'cycle', '20230101', 'analysis'))
        self.assertEqual(c.forecast_dir(time, 'qg'), os.path.join(cwd, 'test', 'cycle', '20230101', 'qg'))

    def test_nproc_mem_divides_nproc(self):
        with self.assertRaises(ValueError):
            c = Config(nproc=10, nproc_mem=3)

    def test_nproc_mem_nproc_rec_arithmetic(self):
        c = Config(nproc=10, nproc_mem=2)
        self.assertEqual(c.nproc_mem, 2)
        self.assertEqual(c.nproc_rec, 5)

    def test_yaml_file_dump_and_load(self):
        c = Config()

        with tempfile.NamedTemporaryFile(prefix='config', suffix='.yml', delete=False) as tmp_file:
            c.dump_yaml(tmp_file.name)
            c_tmp = Config(config_file=tmp_file.name)
            self.assertEqual(c.config_dict, c_tmp.config_dict)

if __name__ == '__main__':
    unittest.main()
