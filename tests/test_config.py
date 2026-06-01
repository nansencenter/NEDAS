import unittest
import os
import tempfile
import yaml
from datetime import datetime, timezone
from NEDAS.config import Config
from NEDAS.config.parse_config import str2bool, parse_config


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

    def test_raise_exception_if_time_parse_error(self):
        with self.assertRaises(ValueError):
            Config(time='abcdefg')

    def test_argparse_time_str(self):
        tzinfo = timezone.utc
        self.assertEqual(Config(time='2001-01-01 00:00:00').time, datetime(2001, 1, 1, tzinfo=tzinfo))
        self.assertEqual(Config(time='20010101000000').time,      datetime(2001, 1, 1, tzinfo=tzinfo))
        self.assertEqual(Config(time='2001-01-01T00:00:00Z').time, datetime(2001, 1, 1, tzinfo=tzinfo))

    def test_nproc_mem_must_divide_nproc(self):
        with self.assertRaises(ValueError):
            Config(nproc=10, nproc_mem=3)

    def test_nproc_mem_nproc_rec_arithmetic(self):
        c = Config(nproc=10, nproc_mem=2)
        self.assertEqual(c.nproc_mem, 2)
        self.assertEqual(c.nproc_rec, 5)

    def test_yaml_file_dump_and_load(self):
        c = Config()
        with tempfile.NamedTemporaryFile(prefix='config', suffix='.yml', delete=False) as f:
            c.dump_yaml(f.name)
            c2 = Config(config_file=f.name)
            self.assertEqual(c.__dict__, c2.__dict__)


class TestStr2Bool(unittest.TestCase):

    def test_true_variants(self):
        for s in ['y', 'yes', 'on', 't', 'true', '.true.', 'Y', 'YES', 'True', 'TRUE']:
            with self.subTest(s=s):
                self.assertEqual(str2bool(s), 1)

    def test_false_variants(self):
        for s in ['n', 'no', 'off', 'f', 'false', '.false.', 'N', 'NO', 'False', 'FALSE']:
            with self.subTest(s=s):
                self.assertEqual(str2bool(s), 0)

    def test_invalid_raises_value_error(self):
        for s in ['maybe', '1', '0', '', 'nope']:
            with self.subTest(s=s):
                with self.assertRaises(ValueError):
                    str2bool(s)


class TestParseConfig(unittest.TestCase):

    def _nedas_root(self):
        import NEDAS
        return os.path.dirname(NEDAS.__file__)

    def test_returns_dict(self):
        self.assertIsInstance(parse_config(code_dir=self._nedas_root()), dict)

    def test_kwarg_overrides_default(self):
        self.assertEqual(parse_config(code_dir=self._nedas_root(), nens=77)['nens'], 77)

    def test_yaml_file_overrides_default(self):
        with tempfile.NamedTemporaryFile(suffix='.yml', mode='w', delete=False) as f:
            yaml.dump({'nens': 55}, f)
            tmp = f.name
        try:
            self.assertEqual(parse_config(code_dir=self._nedas_root(), config_file=tmp)['nens'], 55)
        finally:
            os.unlink(tmp)

    def test_kwarg_overrides_yaml_file(self):
        with tempfile.NamedTemporaryFile(suffix='.yml', mode='w', delete=False) as f:
            yaml.dump({'nens': 55}, f)
            tmp = f.name
        try:
            self.assertEqual(
                parse_config(code_dir=self._nedas_root(), config_file=tmp, nens=99)['nens'], 99)
        finally:
            os.unlink(tmp)


if __name__ == '__main__':
    unittest.main()
