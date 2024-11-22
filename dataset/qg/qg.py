import numpy as np
import os
from ..dataset_config import DatasetConfig

class Dataset(DatasetConfig):
    def __init__(self, config_file=None, parse_args=False, **kwargs):
        super().__init__(config_file, parse_args, **kwargs)

        self.variables = {
            'velocity': {'dtype':'float', 'is_vector':True, 'z_units':'*', 'units':'*'},
            'streamfunc': {'dtype':'float', 'is_vector':False, 'z_units':'*', 'units':'*'},
            'temperature': {'dtype':'float', 'is_vector':False, 'z_units':'*', 'units':'*'},
            'vorticity': {'dtype':'float', 'is_vector':False, 'z_units':'*', 'units':'*'},
        }

        self.operator = {}

    def random_network(self, **kwargs):
        super().parse_kwargs(**kwargs)

        if self.nobs is None:
            self.nobs = 1000

        obs_y = np.random.uniform(self.grid.ymin, self.grid.ymax, self.nobs)
        obs_x = np.random.uniform(self.grid.xmin, self.grid.xmax, self.nobs)

        # obs_z = np.random.uniform(0, 1, self.nobs)
        obs_z = np.zeros(self.nobs)

        obs_seq = {'obs': np.full(self.nobs, np.nan),
                't': np.full(self.nobs, self.time),
                'z': obs_z,
                'y': obs_y,
                'x': obs_x,
                'err_std': np.ones(self.nobs) * self.err['std']
                }
        return obs_seq

