import os
import glob
import numpy as np
from datetime import datetime, timedelta
import pyproj
from NEDAS.grid import Grid
from NEDAS.datasets import Dataset

class AMSR2Obs(Dataset):

    def __init__(self, config_file=None, parse_args=False, **kwargs):
        super().__init__(config_file, parse_args, **kwargs)

        self.variables = {
            'tb19h': {'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'K'},
            'tb19v': {'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'K'},
            'tb37h': {'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'K'},
            'tb37v': {'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'K'},
            }

        ##obs is in NorthPolarStereo projection:
        self.proj = pyproj.Proj(self.proj4)
        self.grid = None

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

        if 'obs_window_min' in kwargs and 'obs_window_max' in kwargs:
            d_range = [kwargs['obs_window_min'], kwargs['obs_window_max']]
        else:
            d_range = [0]

        file_list = []
        # search = os.path.join(kwargs['path'], '&&&')
        # for result in glob.glob(search):
        #     ss = result.split('_')
        #     t1 = datetime.strptime(ss[1], '%Y-%m-%d')
        #     t2 = datetime.strptime(ss[2], '%Y-%m-%d')
        #     for d in d_range:
        #         t_ = t + d * timedelta(hours=1)
        #         if t_ >= t1 and t_ <= t2 and result not in file_list:
        #             file_list.append(result)

        # assert len(file_list)>0, 'no matching files found'

        return file_list

    def read_obs(self, **kwargs):
        """read obs from AMSR2 dataset"""
        kwargs = super().parse_kwargs(**kwargs)
        obs_name = kwargs['name']
        d0_out = kwargs['time'] + timedelta(hours=1) * kwargs['obs_window_min']
        d1_out = kwargs['time'] + timedelta(hours=1) * kwargs['obs_window_max']

        ##target grid for obs_seq
        grid = kwargs['grid']

        ##note: x,y are obs location on grid.proj (in meters)
        ##      x0,y0,x1,y1 are position on rgps_proj (in kilometers),
        ##      triangles: velocity is defined on nodes and deform on elements
        ##      record: trajectory id, we process one record at a time
        obs_seq = {'obs':[], 'err_std':[], 't':[], 'z':[], 'y':[], 'x':[],
                   'x0':[], 'y0':[], 'triangles':[], 'record':[], 'index':[]}

        rec = 0
        for file_name in self.filename(**kwargs):
            obs_seq['obs'].append(1)
            rec += 1

        ##convert from list to np.array
        ##raw data are kept in list format
        for key in ('obs', 'err_std', 't', 'y', 'x', 'z'):
            obs_seq[key] = np.array(obs_seq[key])

        if self.variables[obs_name]['is_vector']:
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

