import unittest
import os
from datetime import datetime, timezone
from NEDAS.core import Context

class TestContext(unittest.TestCase):

    def setUp(self):
        self.c = Context()

    def test_mem_list_init(self):
        self.assertIsNotNone(self.c.mem_list)
        self.assertIsInstance(self.c.mem_list, dict)
        self.assertIsInstance(self.c.mem_list[self.c.pid_mem], list)

    def test_prev_next_time_variable_type(self):
        self.assertIsInstance(self.c.prev_time, datetime)
        self.assertIsInstance(self.c.next_time, datetime)

    def test_prev_next_time_arithmetic(self):
        time_start = datetime(2022, 1, 1)
        time = datetime(2023, 1, 1)
        cycle_period = 24
        c = Context(time_start=time_start, time=time, cycle_period=cycle_period)
        self.assertEqual(c.prev_time, datetime(2022, 12, 31, tzinfo=timezone.utc))
        self.assertEqual(c.next_time, datetime(2023, 1, 2, tzinfo=timezone.utc))

if __name__ == '__main__':
    unittest.main()
