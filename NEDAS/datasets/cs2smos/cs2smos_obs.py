import numpy as np
import pyproj
from NEDAS.grid import Grid
from NEDAS.datasets import Dataset

class Cs2SmosObs(Dataset):
    def __init__(self, config_file=None, parse_args=False, **kwargs):
        super().__init__(config_file, parse_args, **kwargs)

        self.variables = {'seaice_thick': {'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'m'}, }

        proj = pyproj.Proj(self.proj)
        x, y = np.meshgrid(np.arange(self.xstart, self.xend, self.dx), np.arange(self.ystart, self.yend, self.dy))
        self.grid = Grid(proj, x, y)

    def filename(self, **kwargs):
        pass

    def read_obs(self, **kwargs):
        raise NotImplementedError
        ##TODO: implement real CS2SMOS dataset
        kwargs = super().parse_kwargs(**kwargs)
        grid = kwargs['grid']
        mask = kwargs['mask']

        obs_seq = {'obs':[], 't':[], 'z':[], 'y':[], 'x':[], 'err_std':[], }

        return obs_seq

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
            if valid[0] == 0:
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
