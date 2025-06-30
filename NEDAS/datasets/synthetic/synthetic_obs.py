import os
import numpy as np
from datetime import datetime, timedelta, timezone
from NEDAS.grid import Grid
from NEDAS.models import get_model_class
from NEDAS.datasets import Dataset

class SyntheticObs(Dataset):

    def __init__(self, config_file=None, parse_args=False, **kwargs):
        super().__init__(config_file, parse_args, **kwargs)

        if 'model_src' not in kwargs:
            raise KeyError("synthetic obs expecting 'model_src' in init kwargs")
        Model = get_model_class(kwargs['model_src'])
        model = Model()

        self.variables = {}
        for vname, vrec in model.variables.items():
            self.variables[vname] = {}
            for key in ['dtype', 'is_vector', 'units']:
                self.variables[vname][key] = model.variables[vname][key]
            self.variables[vname]['z_units'] = 'm'

        self.grid = model.grid

    def random_network(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)

        if kwargs['nobs'] is None:
            nobs = 1000
        else:
            nobs = kwargs['nobs']

        grid = kwargs['grid']

        obs_x = []
        obs_y = []
        while len(obs_x) < nobs:
            y = np.random.uniform(grid.ymin, grid.ymax)
            x = np.random.uniform(grid.xmin, grid.xmax)
            valid = grid.interp(grid.mask.astype(int), x, y)
            if valid == 0:
                obs_x.append(x)
                obs_y.append(y)

        obs_x = np.array(obs_x)
        obs_y = np.array(obs_y)
        obs_z = np.zeros(nobs)

        obs_seq = {'obs': np.full(nobs, np.nan),
                't': np.full(nobs, kwargs['time']),
                'z': obs_z,
                'y': obs_y,
                'x': obs_x,
                'err_std': np.ones(nobs) * kwargs['err']['std']
                }
        return obs_seq

    def read_obs(self, **kwargs):
        raise NotImplementedError("read_obs is not implemented for synthetic observations")

