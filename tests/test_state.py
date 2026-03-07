import unittest
import numpy as np
from NEDAS.config import Config
from NEDAS.core.state_info import StateInfo
from NEDAS.core.state import State
from NEDAS.core.context import Context
from NEDAS.core.types import FieldRecord

class TestStateClass(unittest.TestCase):
    """Unit tests for the State class"""

    def setUp(self):
        config = Config()
        self.c = Context(config)
        self.c.state = State(self.c)

    def test_state_init(self):
        self.assertIsNotNone(self.c.state.info)
        self.assertIsNotNone(self.c.mem_list)
        self.assertIsNotNone(self.c.state.rec_list)

    def test_state_info_check(self):
        self.assertIsInstance(self.c.state.info, StateInfo)
        self.assertIsInstance(self.c.state.info.fields, dict)

    def test_task_lists_check(self):
        self.assertIsInstance(self.c.mem_list, dict)
        self.assertIsInstance(self.c.mem_list[self.c.pid_mem], list)
        self.assertIsInstance(self.c.state.rec_list, dict)
        self.assertIsInstance(self.c.state.rec_list[self.c.pid_rec], list)


if __name__ == '__main__':
    unittest.main()