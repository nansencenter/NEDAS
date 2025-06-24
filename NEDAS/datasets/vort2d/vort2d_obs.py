import numpy as np
from NEDAS.models.vort2d import Vort2DModel
from NEDAS.datasets import Dataset

class Vort2DObs(Dataset):
    def __init__(self, config_file=None, parse_args=False, **kwargs):
        super().__init__(config_file, parse_args, **kwargs)

        self.variables = {
            'velocity': {'dtype':'float', 'is_vector':True, 'z_units':'m', 'units':'m/s'},
            'vortex_position': {'dtype':'float', 'is_vector':True, 'z_units':'m', 'units':'m'},
            'vortex_intensity': {'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'m/s'},
            'vortex_size':  {'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'m'},
            }

        self.obs_operator = {
            'vortex_position': self.get_vortex_position,
            'vortex_intensity': self.get_vortex_intensity,
            'vortex_size': self.get_vortex_size,
            }

    def random_network(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        name = kwargs['name']
        model = kwargs['model']
        assert isinstance(model, Vort2DModel), 'random_network: ERROR: model must be an instance of Vort2DModel'
        grid = kwargs['grid']

        ##get truth vortex position, some network is vortex-following
        velocity = self.get_velocity(**{**kwargs, 'path': model.truth_dir})
        ##diagnose the vortex position on grid
        i, j = self.vortex_position(velocity[0,...], velocity[1,...])
        true_center_x, true_center_y = grid.x[j,i], grid.y[j,i]

        if 'network_type' in kwargs:
            network_type = kwargs['network_type']
        else:
            network_type = 'global'

        if name == 'velocity':

            nobs = kwargs['nobs']
            if network_type == 'global':
                if nobs is None:
                    nobs = 1000
                y = np.random.uniform(grid.ymin, grid.ymax, nobs)
                x = np.random.uniform(grid.xmin, grid.xmax, nobs)

            elif network_type == 'targeted':
                if nobs is None:
                    nobs = 800   ##note: number of obs in entire domain
                                 ##later only obs within range will be kept
                obs_range = 180000  ##observed range from vortex center, m
                y = np.random.uniform(grid.ymin, grid.ymax, nobs)
                x = np.random.uniform(grid.xmin, grid.xmax, nobs)

                dist = np.hypot(x - true_center_x, y - true_center_y)
                ind = np.where(dist <= obs_range)
                x = x[ind]
                y = y[ind]
                nobs = x.size

            else:
                raise ValueError('unknown network type: '+network_type)

            obs_seq = {'obs': np.full(nobs, np.nan),
                    't': np.full(nobs, kwargs['time']),
                    'z': np.zeros(nobs),
                    'y': y,
                    'x': x,
                    'err_std': np.ones(nobs) * kwargs['err']['std']
                    }

        elif name == 'vortex_position':
            obs_seq = {'obs': np.array([[np.nan, np.nan]]),
                    't': np.array([kwargs['time']]),
                    'z': np.array([0]),
                    'y': np.array([true_center_y]),
                    'x': np.array([true_center_x]),
                    'err_std': np.array([kwargs['err']['std']])
                    }

        elif name in ['vortex_intensity', 'vortex_size']:
            obs_seq = {'obs': np.array([np.nan]),
                    't': np.array([kwargs['time']]),
                    'z': np.array([0]),
                    'y': np.array([true_center_y]),
                    'x': np.array([true_center_x]),
                    'err_std': np.array([kwargs['err']['std']])
                    }

        else:
            raise ValueError('unknown obs variable: '+name)

        return obs_seq

    ###utility functions for obs diagnostics
    def vortex_position(self, u, v):
        ny, nx = u.shape

        ##compute vorticity
        zeta = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1) - np.roll(u, -1, axis=0) + np.roll(u, 1, axis=0)) / 2.0

        ##search for max vorticity
        zmax = -999
        center_x, center_y = -1, -1
        buff = 6
        for j in range(buff, ny-buff):
            for i in range(buff, nx-buff):
                z = np.sum(zeta[j-buff:j+buff, i-buff:i+buff])
                if z > zmax:
                    zmax = z
                    center_i, center_j = i, j

        return center_i, center_j

    def vortex_intensity(self, u, v):
        return np.max(np.hypot(u, v))

    def vortex_size(self, u, v, center_i, center_j):
        wind = np.hypot(u, v)
        ny, nx = wind.shape

        nr = 30
        wind_min = 15
        wind_rad = np.zeros(nr)
        count_rad = np.zeros(nr)
        for j in range(-nr, nr+1):
            for i in range(-nr, nr+1):
                r = int(np.sqrt(i**2+j**2))
                if r < nr:
                    wind_rad[r] += wind[int(center_j+j)%ny, int(center_i+i)%nx]
                    count_rad[r] += 1
        wind_rad = wind_rad/count_rad

        if np.max(wind_rad)<wind_min or np.where(wind_rad>=wind_min)[0].size==0:
            Rsize = -1
        else:
            i1 = np.where(wind_rad>=wind_min)[0][-1] ###last point with wind > 35knot
            if i1==nr-1:
                Rsize = i1
            else:
                Rsize = i1 + (wind_rad[i1] - wind_min) / (wind_rad[i1] - wind_rad[i1+1])

        return Rsize

    def get_velocity(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        model = kwargs['model']
        assert isinstance(model, Vort2DModel), 'get_velocity: ERROR: model must be an instance of Vort2DModel'
        grid = kwargs['grid']
        ##read the velocity field from truth
        model_velocity = model.read_var(**{**kwargs, 'name':'velocity'})
        ##convert velocity to target grid
        model.grid.set_destination_grid(grid)
        velocity = model.grid.convert(model_velocity, is_vector=True)
        return velocity

    def get_vortex_position(self, **kwargs):
        velocity = self.get_velocity(**kwargs)
        grid = kwargs['model'].grid
        center_i, center_j = self.vortex_position(velocity[0,...], velocity[1,...])
        obs_seq = np.zeros((2, 1), dtype='float')
        obs_seq[0,0] = grid.x[center_j, center_i]
        obs_seq[1,0] = grid.y[center_j, center_i]
        return obs_seq

    def get_vortex_intensity(self, **kwargs):
        velocity = self.get_velocity(**kwargs)
        Vmax = self.vortex_intensity(velocity[0,...], velocity[1,...])
        return np.array([Vmax])

    def get_vortex_size(self, **kwargs):
        velocity = self.get_velocity(**kwargs)
        dx = kwargs['model'].grid.dx
        center_i, center_j = self.vortex_position(velocity[0,...], velocity[1,...])
        Rsize = self.vortex_size(velocity[0,...], velocity[1,...], center_i, center_j)
        Rsize = Rsize * dx
        return np.array([Rsize])

    def read_obs(self):
        raise NotImplementedError("read_obs is not implemented for vort2d, since only using synthetic obs.")