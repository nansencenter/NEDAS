import unittest
import os
from datetime import datetime, timezone
from NEDAS.config import Config
from NEDAS.core import Coordinator

class TestCoordinator(unittest.TestCase):

    def test_prev_next_time_variable_type(self):
        cf = Config()
        c = Coordinator(cf)
        self.assertIsInstance(c.prev_time, datetime)
        self.assertIsInstance(c.next_time, datetime)

    def test_prev_next_time_arithmetic(self):
        time_start = datetime(2022, 1, 1)
        time = datetime(2023, 1, 1)
        cycle_period = 24
        cf = Config(time_start=time_start, time=time, cycle_period=cycle_period)
        c = Coordinator(cf)
        self.assertEqual(c.prev_time, datetime(2022, 12, 31, tzinfo=timezone.utc))
        self.assertEqual(c.next_time, datetime(2023, 1, 2, tzinfo=timezone.utc))

if __name__ == '__main__':
    unittest.main()
