import numpy as np
import importlib
from ..dataset_config import DatasetConfig

class Dataset(DatasetConfig):

    def __init__(self, config_file=None, parse_args=False, **kwargs):
        super().__init__(config_file, parse_args, **kwargs)

        self.variables = {
            'state': {'dtype':'float', 'is_vector':False, 'z_units':'*', 'units':'*'},
            }

    def random_network(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)

        nobs = kwargs['nobs']
        obs_x = np.random.uniform(kwargs['grid'].xmin, kwargs['grid'].xmax, nobs)

        obs_seq = {'obs': np.full(nobs, np.nan),
                't': np.full(nobs, kwargs['time']),
                'z': np.zeros(nobs),
                'y': np.zeros(nobs),
                'x': obs_x,
                'err_std': np.ones(nobs) * kwargs['err']['std'],
                }

        return obs_seq

