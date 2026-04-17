import numpy as np
from NEDAS.core import Dataset

class SyntheticObs(Dataset):
    nobs: int|None
    obs_position: str
    obs_x: list[float]
    obs_y: list[float]|None
    obs_z: list[float]|None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if 'model_src' in kwargs:
            model = self.c.models[kwargs['model_src']]
            for vname, vrec in model.variables.items():
                self.variables[vname] = model.variables[vname]
            self.grid = model.grid

    def generate_obs_network(self, **kwargs):
        kwargs = super().parse_kwargs(kwargs)

        if self.nobs:
            nobs = self.nobs
        if kwargs['nobs'] is None:
            nobs = 1000
        else:
            nobs = kwargs['nobs']

        grid = kwargs['grid']

        if self.obs_position == 'random':
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

        elif self.obs_position == 'prescribed':
            obs_x = np.array(self.obs_x)
            obs_y = np.array(self.obs_y) if self.obs_y else np.zeros(nobs)
            obs_z = np.array(self.obs_z) if self.obs_z else np.zeros(nobs)

        else:
            raise ValueError(f"SyntheticObs: failed to create obs network using position = {self.obs_position}")

        obs_seq = {
            'obs': np.full(nobs, np.nan),
            't': np.full(nobs, kwargs['time']),
            'z': obs_z,
            'y': obs_y,
            'x': obs_x,
            'err_std': np.ones(nobs) * kwargs['err']['std']
        }
        return obs_seq

    def read_obs(self, **kwargs):
        raise NotImplementedError("read_obs is not implemented for synthetic observations")

