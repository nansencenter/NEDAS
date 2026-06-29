import os
import numpy as np
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
        self.assertEqual(self.c.io.state_binfile_name(self.c, 'prior'), binfile)

class _MockContext:
    """Minimal context for OnlineIO tests — no MPI, no models, no IO setup."""
    pass


class TestOnlineIO(unittest.TestCase):

    def setUp(self):
        self.io = OnlineIO()
        self.c = _MockContext()
        self.c.state = _MockContext()

    def test_io_instance(self):
        self.assertIsInstance(self.io, OnlineIO)

    def test_write_then_read_field(self):
        fld = np.array([1.0, 2.0, 3.0])
        self.io.write_field(fld, self.c, 'prior', rec_id=0, mem_id=0)
        result = self.io.read_field(self.c, 'prior', rec_id=0, mem_id=0)
        np.testing.assert_array_equal(result, fld)

    def test_write_multiple_members(self):
        for mem_id in range(3):
            fld = np.full(5, float(mem_id))
            self.io.write_field(fld, self.c, 'prior', rec_id=0, mem_id=mem_id)
        for mem_id in range(3):
            result = self.io.read_field(self.c, 'prior', rec_id=0, mem_id=mem_id)
            np.testing.assert_allclose(result, float(mem_id))

    def test_write_multiple_records(self):
        for rec_id in range(4):
            fld = np.full(3, float(rec_id) * 10)
            self.io.write_field(fld, self.c, 'post', rec_id=rec_id, mem_id=0)
        for rec_id in range(4):
            result = self.io.read_field(self.c, 'post', rec_id=rec_id, mem_id=0)
            np.testing.assert_allclose(result, float(rec_id) * 10)

    def test_call_method_passes_tag(self):
        received_kwargs = {}

        def dummy_method(*args, **kwargs):
            received_kwargs.update(kwargs)
            return 'ok'

        result = self.io.call_method(self.c, 'prior', dummy_method, 'arg1')
        self.assertEqual(result, 'ok')
        self.assertEqual(received_kwargs.get('tag'), 'prior')

    def test_invalid_tag_raises(self):
        fld = np.zeros(3)
        with self.assertRaises(Exception):
            self.io.write_field(fld, self.c, 'invalid_tag', rec_id=0, mem_id=0)


if __name__ == '__main__':
    unittest.main()
