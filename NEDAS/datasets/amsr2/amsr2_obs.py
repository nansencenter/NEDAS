import os
import numpy as np
from datetime import datetime, timedelta
import pyproj
import netCDF4
from NEDAS.grid import Grid
from NEDAS.datasets import Dataset
import .rtm_amsr_fcts as rtm_amsr

class AMSR2Obs(Dataset):

    def __init__(self, config_file=None, parse_args=False, **kwargs):
        super().__init__(config_file, parse_args, **kwargs)

        self.variables = {
            'tb19h': {'name':'tb19h', 'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'K'},
            'tb19v': {'name':'tb19v', 'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'K'},
            'tb37h': {'name':'tb37h', 'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'K'},
            'tb37v': {'name':'tb37v', 'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'K'},
            }

        ##obs is in NorthPolarStereo projection:
        self.proj = pyproj.Proj(self.proj4)
        self.grid = Grid.regular_grid(self.proj, self.xstart, self.xend, self.ystart, self.yend, self.dx)

        self.obs_operator = {
        #    'tb19h': self.get_simulated_tb19h,
        }

    def parse_kwargs(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        assert kwargs['time'] is not None, 'AMSR2Obs.read_obs: time is not defined'
        assert kwargs['obs_window_max'] is not None, 'AMSR2Obs.read_obs: obs_window_max is not defined'
        assert kwargs['obs_window_min'] is not None, 'AMSR2Obs.read_obs: obs_window_min is not defined'
        return kwargs

    def filename(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        t = kwargs['time']
        return os.path.join(self.dataset_dir, f"{t:%Y}", f"tc_amsr-gw1_topaz5-6p25km_{t:%Y%m%d}12.nc")

    def read_obs(self, **kwargs):
        """read obs from AMSR2 dataset"""
        kwargs = super().parse_kwargs(**kwargs)
        t = kwargs['time']
        obs_name = kwargs['name']
        native_name = self.variables[obs_name]['name']

        is_vector = self.variables[obs_name]['is_vector']

        obs_err_std = self.obs_err_std
        if 'err' in kwargs:
            obs_err_std = kwargs['err']['std']

        ##target grid for obs_seq
        grid = kwargs['grid']

        ##note: x,y are obs location on grid.proj (in meters)
        ##      x0,y0,x1,y1 are position on rgps_proj (in kilometers),
        ##      triangles: velocity is defined on nodes and deform on elements
        ##      record: trajectory id, we process one record at a time
        obs_seq = {'obs':[], 'err_std':[], 't':[], 'z':[], 'y':[], 'x':[],}

        with netCDF4.Dataset(self.filename(**kwargs), 'r') as f:
            ##read the tc_amsr-gw1_topaz5-6p25km_*.nc files for the observed tb data
            ##flip y direction (the files have decreasing y coords)
            tmp = f[native_name][0,::-1,:]
            dat = tmp.data
            dat[tmp.mask] = np.nan

        ##convert to target grid
        self.grid.set_destination_grid(grid)
        dat1 = self.grid.convert(dat, is_vector=is_vector, coarse_grain=True)
        mask1 = np.isnan(dat1)
        ones = np.ones(np.sum(~mask1))

        ##build obs sequence
        obs_seq['obs'].append(dat1[~mask1])
        obs_seq['err_std'].append(ones*obs_err_std)
        obs_seq['t'].append(np.full(ones.size, t))
        obs_seq['z'].append(np.full(ones.size, 0.0))
        obs_seq['y'].append(grid.y[~mask1])
        obs_seq['x'].append(grid.x[~mask1])

        ##convert from list to np.array
        ##raw data are kept in list format
        for key in ('obs', 'err_std', 't', 'y', 'x', 'z'):
            obs_seq[key] = np.array(obs_seq[key])

        if is_vector:
            obs_seq['obs'] = obs_seq['obs'].T

        return obs_seq

    def random_network(self):
        raise NotImplementedError

    def _get_simulated_tb(self, obs_name, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)

        obs_seq = np.full(nobs, np.nan)

        return obs_seq

    def get_simulated_tb19h(self, **kwargs):
        return _get_simulated_tb('tb19h', **kwargs)

