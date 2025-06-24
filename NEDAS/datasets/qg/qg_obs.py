import numpy as np
from NEDAS.datasets import Dataset

class QGObs(Dataset):
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
        kwargs = super().parse_kwargs(**kwargs)

        if kwargs['nobs'] is None:
            nobs = 1000
        else:
            nobs = kwargs['nobs']

        grid = kwargs['grid']

        obs_y = np.random.uniform(grid.ymin, grid.ymax, nobs)
        obs_x = np.random.uniform(grid.xmin, grid.xmax, nobs)

        # obs_z = np.random.uniform(0, 1, self.nobs)
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
        raise NotImplementedError('read_obs: not implemented for qg model since only using synthetic obs')
