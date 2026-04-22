import unittest
import os
import tempfile
from datetime import datetime, timezone
from NEDAS.config import Config

class TestConfig(unittest.TestCase):
    def test_convert_work_dir_to_abspath(self):
        cf = Config()
        self.assertEqual(cf.work_dir, os.path.abspath(cf.work_dir))

    def test_nedas_root_available(self):
        cf = Config()
        self.assertTrue(os.path.exists(cf.nedas_root))

    def test_config_time_variable_type(self):
        cf = Config()
        self.assertIsInstance(cf.time, datetime)
        self.assertIsInstance(cf.time_start, datetime)
        self.assertIsInstance(cf.time_end, datetime)
        self.assertIsInstance(cf.time_analysis_start, datetime)
        self.assertIsInstance(cf.time_analysis_end, datetime)

    def test_raise_exception_if_time_parse_error(self):
        with self.assertRaises(ValueError):
            Config(time='abcdefg')

    def test_argparse_time_str(self):
        tzinfo = timezone.utc
        cf = Config(time='2001-01-01 00:00:00')
        self.assertEqual(cf.time, datetime(2001,1,1, tzinfo=tzinfo))
        cf = Config(time='20010101000000')
        self.assertEqual(cf.time, datetime(2001,1,1, tzinfo=tzinfo))
        cf = Config(time='2001-01-01T00:00:00Z')
        self.assertEqual(cf.time, datetime(2001,1,1,tzinfo=tzinfo))

    def test_nproc_mem_divides_nproc(self):
        with self.assertRaises(ValueError):
            cf = Config(nproc=10, nproc_mem=3)

    def test_nproc_mem_nproc_rec_arithmetic(self):
        cf = Config(nproc=10, nproc_mem=2)
        self.assertEqual(cf.nproc_mem, 2)
        self.assertEqual(cf.nproc_rec, 5)

    def test_yaml_file_dump_and_load(self):
        cf = Config()

        with tempfile.NamedTemporaryFile(prefix='config', suffix='.yml', delete=False) as tmp_file:
            cf.dump_yaml(tmp_file.name)
            cf_tmp = Config(config_file=tmp_file.name)
            self.assertEqual(cf.__dict__, cf_tmp.__dict__)

if __name__ == '__main__':
    unittest.main()
