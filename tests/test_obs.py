import unittest
import numpy as np
from NEDAS.config import Config
from NEDAS.core.state import State
from NEDAS.core.obs_info import ObsInfo
from NEDAS.core.obs import Obs
from NEDAS.core.context import Context

class TestObsClass(unittest.TestCase):
    """Test suite for Obs class"""

    def setUp(self):
        """Set up test fixtures"""
        config = Config()
        self.c = Context(config)
        self.c.state = State(self.c)
        self.c.obs = Obs(self.c)

    def test_obs_init(self):
        self.assertIsNotNone(self.c.obs.info)
        self.assertIsNotNone(self.c.obs.obs_rec_list)

    def test_state_info_check(self):
        self.assertIsInstance(self.c.obs.info, ObsInfo)
        self.assertIsInstance(self.c.obs.info.records, dict)

    def test_task_lists_check(self):
        self.assertIsInstance(self.c.obs.obs_rec_list, dict)
        self.assertIsInstance(self.c.obs.obs_rec_list[self.c.pid_rec], list)

if __name__ == '__main__':
    unittest.main()