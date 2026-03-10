import unittest
import numpy as np
from NEDAS.core.state import State
from NEDAS.core.obs_info import ObsInfo
from NEDAS.core.obs import Obs
from NEDAS.core.context import Context

class TestObsClassInit(unittest.TestCase):
    """Test suite for Obs class"""

    def setUp(self):
        """Set up test fixtures"""
        self.c = Context()
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

    def test_collect_obs_seq(self):
        ...

    def test_state_to_obs(self):
        ...

class TestVerticalInterp(unittest.TestCase):
    """Test vertical_interp method for Obs class"""

    def setUp(self):
        self.c = Context()
        self.c.state = State(self.c)
        self.c.obs = Obs(self.c)

        # Setup common testing variables
        # k    i                  fz    z    dz
        # sfc     --------------- 10     0
        #         -  -  -  -  -  =10        } 1
        # 1    0  --------------- 15     1
        #         -  -  -  -  -  =20        } 1
        # 2    1  --------------- 25     2
        #         -  -  -  -  -  =30        } 1
        # 3    2  --------------- 30     3
        self.nobs = 5
        self.levels = [1, 2, 3]    # layer index
        self.dz = [1.0, 1.0, 1.0]  # layer thickness
        self.z = [1.0, 2.0, 3.0]   # z coords at each layer
        self.fz = [10., 20., 30.]  # field value at each layer
        self.obs_z = np.array([0.1, 0.5, 1.0, 2.2, 3.0])
        self.seq = np.zeros(self.nobs)

    def test_first_level(self):
        """Test i=0: Constant f from 0 to dz/2 (not including)."""
        i = 0
        f = np.ones(self.nobs) * self.fz[i]  # field value is 10.0
        z = np.ones(self.nobs) * self.dz[i]  # z(0) thickness is 1 so dz is 1.0

        # At i=0, code looks for obs_z between [0, 0.5*dz)
        seq_out, fp, zp, dzp = self.c.obs.vertical_interp(
            self.seq.copy(), self.levels[i], self.levels, f, None, z, None, None, self.obs_z
        )

        self.assertEqual(seq_out[0], 10.0) # obs_z[0]=0.1 is in range. It should be 10.0
        self.assertEqual(seq_out[1], 0.0) # out of range
        self.assertEqual(seq_out[2], 0.0)
        self.assertEqual(fp[0], 10.0)  # check if prev layer is returned correctly
        self.assertEqual(zp[0], 1.0)
        self.assertEqual(dzp[0], 1.0)

    def test_middle_level_linear(self):
        """Test i>0: Linear interpolation between layers."""
        i = 1
        # Setup previous layer (i=0)
        fp = np.ones(self.nobs) * self.fz[i-1]
        zp = np.ones(self.nobs) * self.z[i-1]
        dzp = np.ones(self.nobs) * self.dz[i-1] # z_fp = 1.0 - 0.5 = 0.5

        # Current layer (i=1)
        f = np.ones(self.nobs) * self.fz[i]
        z = np.ones(self.nobs) * self.z[i]  # dz = 2.0 - 1.0 = 1.0. z_f = 1.0 + 0.5 = 1.5

        # Range is [0.5, 1.5). Midpoint is 1.0. 
        # At obs_z=1.0, value should be 15.0 (avg of 10 and 20)
        seq_out, fp, zp, dzp = self.c.obs.vertical_interp(
            self.seq.copy(), self.levels[i], self.levels, f, fp, z, zp, dzp, self.obs_z
        )

        # prev level values shouldn't be changed
        self.assertEqual(seq_out[0], 0.0)  # prev layer shouldn't be changed
        self.assertEqual(seq_out[1], 10.0) # Exactly at z_fp
        self.assertEqual(seq_out[2], 15.0) # Midpoint
        self.assertEqual(seq_out[3], 0.0)  # Exactly at z_f, out of range
        self.assertEqual(seq_out[4], 0.0)  # out of range
        self.assertEqual(fp[0], 20) # check if prev layer is returned correctly
        self.assertEqual(zp[0], 2.0)
        self.assertEqual(dzp[0], 1.0)

    def test_last_level(self):
        """Test i=last: Constant f from z-dz/2 to z."""
        i = 2
        # setup previous layer (i=1)
        fp = np.ones(self.nobs) * self.fz[i-1]
        zp = np.ones(self.nobs) * self.z[i-1]
        dzp = np.ones(self.nobs) * self.dz[i-1] # z_fp = 2.0 - 0.5 = 1.5

        # current layer (i=2)
        f = np.ones(self.nobs) * self.fz[i]
        z = np.ones(self.nobs) * self.z[i] # dz = 1.0. Range: [2.0 - 0.5, 2.0] = [1.5, 2.0]

        seq_out, _, _, _ = self.c.obs.vertical_interp(
            self.seq.copy(), self.levels[i], self.levels, f, fp, z, zp, dzp, self.obs_z
        )

        self.assertEqual(seq_out[0], 0.0)  # prev layers shouldn't be changed
        self.assertEqual(seq_out[1], 0.0)
        self.assertEqual(seq_out[2], 0.0)
        self.assertAlmostEqual(seq_out[3], 27.0) # obs_z[3]=1.5 is in range
        self.assertEqual(seq_out[4], 30.0) # obs_z[4]=2.0 is in range

    def test_collapsed_layer(self):
        """Test logic when z == zp to avoid division by zero."""
        # alter the data to have level 2 collapsed
        self.dz = [1.0, 0.0, 1.0]  # layer thickness
        self.z = [1.0, 1.0, 2.0]   # z coords at each layer
        self.fz = [10., 10., 20.]  # field value at each layer

        i = 1
        # setup previous layer (i=1)
        fp = np.ones(self.nobs) * self.fz[i-1]  # 10.
        zp = np.ones(self.nobs) * self.z[i-1]   # 1.0
        dzp = np.ones(self.nobs) * self.dz[i-1] # dzp=1.0, z_fp = 1.0 - 0.5 = 0.5

        # current layer (i=2)
        f = np.ones(self.nobs) * self.fz[i]  # 10.
        z = np.ones(self.nobs) * self.z[i]   # dz = 0, z = zp = 1.0, range is [0.5, 1.0)

        # Function should handle this via the 'collapsed' mask
        seq_out, _, _, _ = self.c.obs.vertical_interp(
            self.seq.copy(), self.levels[i], self.levels, f, fp, z, zp, dzp, self.obs_z
        )

        self.assertEqual(seq_out[0], 0.0)  # obs_z[0]=0.1
        self.assertEqual(seq_out[1], 10.0)  # obs_z[1]=0.5  this is within range
        self.assertEqual(seq_out[2], 0.0)  # obs_z[2]=1.0
        self.assertEqual(seq_out[3], 0.0)  # obs_z[3]=2.2
        self.assertEqual(seq_out[4], 0.0)  # obs_z[4]=3.0

if __name__ == '__main__':
    unittest.main()
