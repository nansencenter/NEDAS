import numpy as np
from NEDAS.models.lorenz96 import Lorenz96Model
from NEDAS.datasets import Dataset

class Lorenz96Obs(Dataset):

    def __init__(self, config_file=None, parse_args=False, **kwargs):
        super().__init__(config_file, parse_args, **kwargs)

        self.variables = {
            'state': {'dtype':'float', 'is_vector':False, 'z_units':'*', 'units':'*'},
            }

    def random_network(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        assert isinstance(kwargs['model'], Lorenz96Model), 'random_network: ERROR: model must be an instance of models.lorenz96.Model'
        grid = kwargs['grid']

        nobs = kwargs['nobs']
        obs_x = np.random.uniform(grid.xmin, grid.xmax, nobs)

        obs_seq = {'obs': np.full(nobs, np.nan),
                't': np.full(nobs, kwargs['time']),
                'z': np.zeros(nobs),
                'y': np.zeros(nobs),
                'x': obs_x,
                'err_std': np.ones(nobs) * kwargs['err']['std'],
                }

        return obs_seq

    def read_obs(self):
        raise NotImplementedError('Only synthetic obs will be used.')