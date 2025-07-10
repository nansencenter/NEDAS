import os
import glob
import numpy as np
from datetime import datetime, timedelta
import pyproj
import netCDF4
import pandas as pd
from NEDAS.grid import Grid
from NEDAS.utils.conversion import units_convert
from NEDAS.datasets import Dataset
from .rtm_amsr_fcts import simulated_tb_v03

class AMSR2Obs(Dataset):
    proj4: str
    xstart: float
    xend: float
    ystart: float
    yend: float
    dx: float
    obs_err_std: float
    dataset_dir: str    
    coefficients_file: str
    ow_bias_file: str

    def __init__(self, config_file=None, parse_args=False, **kwargs):
        super().__init__(config_file, parse_args, **kwargs)

        self.channels = ['tb19h', 'tb19v', 'tb37h', 'tb37v']
        self.variables = {
            'dal': {'name': 'dal_SICCI3LF_corrSICCI3LF', 'dtype': 'float', 'is_vector': False, 'z_units': 'm', 'units': 100},
            'eia': {'name': 'eia', 'dtype': 'float', 'is_vector': False, 'z_units': 'm', 'units': 'deg'},
            'sic': {'name': 'ct_SICCI3LF_corrSICCI3LF', 'dtype': 'float', 'is_vector': False, 'z_units': 'm', 'units': 100},
            'owf': {'name': 'owf_SICCI3LF_corrSICCI3LF', 'dtype': 'float', 'is_vector': False, 'z_units': 'm', 'units': 1},
            'wind_speed': {'name': 'wind_speed@tb37', 'dtype': 'float', 'is_vector': False, 'z_units': 'm', 'units': 'm/s'},
            'water_vapor': {'name': 'tcwv@tb37', 'dtype': 'float', 'is_vector': False, 'z_units': 'm', 'units': 'g/kg'},
            'liquid_water': {'name': 'tclw@tb37', 'dtype': 'float', 'is_vector': False, 'z_units': 'm', 'units': 'g/kg'},
            'air_temp': {'name': 'air_temp@tb37', 'dtype': 'float', 'is_vector': False, 'z_units': 'm', 'units': 'K'},
        }
        for ch in self.channels:
            self.variables[ch] = {'name': ch, 'dtype': 'float', 'is_vector': False, 'z_units': 'm', 'units': 'K'}

        ##obs is in NorthPolarStereo projection:
        self.proj = pyproj.Proj(self.proj4)
        self.grid = Grid.regular_grid(self.proj, self.xstart, self.xend, self.ystart, self.yend, self.dx)

        self.coeffs = None
        self.ow_bias = None

        self.obs_operator = {}
        for ch in self.channels:
            self.obs_operator[ch] = self._get_obs_operator(ch)

    def parse_kwargs(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        assert kwargs['time'] is not None, 'AMSR2Obs.read_obs: time is not defined'
        assert kwargs['obs_window_max'] is not None, 'AMSR2Obs.read_obs: obs_window_max is not defined'
        assert kwargs['obs_window_min'] is not None, 'AMSR2Obs.read_obs: obs_window_min is not defined'
        return kwargs

    def filename(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        time = kwargs['time']
        obs_window_min = kwargs['obs_window_min']
        obs_window_max = kwargs['obs_window_max']

        search = None
        if time is None:
            search = os.path.join(self.dataset_dir, "????", f"tc_amsr-gw1_topaz5-6p25km_????????.nc")
            file_list = glob.glob(search)

        else:
            if obs_window_min is not None and obs_window_max is not None:
                d_range = np.arange(obs_window_min, obs_window_max)
            else:
                d_range = [0]

            file_list = []
            for d in d_range:
                t = time + d * timedelta(hours=1)
                search = os.path.join(self.dataset_dir, f"{t:%Y}", f"tc_amsr-gw1_topaz5-6p25km_{t:%Y%m%d}12.nc")
                for result in glob.glob(search):
                    if result not in file_list:
                        file_list.append(result)

        if search is None or len(file_list) == 0:
            raise RuntimeError(f'no matching files found: {search}')
        return file_list
    
    def read_obs(self, **kwargs):
        """read obs from AMSR2 dataset"""
        kwargs = super().parse_kwargs(**kwargs)
        time = kwargs['time']
        obs_name = kwargs['name']
        native_name = self.variables[obs_name]['name']
        native_units = self.variables[obs_name]['units']

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
        obs_seq: dict = {'obs':[], 'err_std':[], 't':[], 'z':[], 'y':[], 'x':[]}

        for file in self.filename(**kwargs):
            with netCDF4.Dataset(file, 'r') as f:
                ##read the tc_amsr-gw1_topaz5-6p25km_*.nc files for the observed tb data
                ##flip y direction (the files have decreasing y coords)
                tmp = f[native_name][0,::-1,:]
                dtime = f['dtime'][0,::-1,:].data
                dat = tmp.data
                dat[tmp.mask] = np.nan

            ##convert to target grid
            self.grid.set_destination_grid(grid)
            dat1 = self.grid.convert(dat, is_vector=is_vector, coarse_grain=True)
            dtime1 = self.grid.convert(dtime, is_vector=False, coarse_grain=True)
            if not isinstance(dat1, np.ndarray) or not isinstance(dtime1, np.ndarray):
                raise ValueError(f"AMSR2Obs.read_obs: conversion to target grid failed for {obs_name}")
            mask1 = np.isnan(dat1)
            ones = np.ones(np.sum(~mask1))

            ##build obs sequence
            obs_seq['obs'].append(dat1[~mask1])
            obs_seq['err_std'].append(ones*obs_err_std)
            obs_seq['t'].append([time + timedelta(seconds=1)*d for d in dtime1[~mask1]])
            obs_seq['z'].append(np.full(ones.size, 0.0))
            obs_seq['y'].append(grid.y[~mask1])
            obs_seq['x'].append(grid.x[~mask1])

        ##convert from list to np.array
        ##raw data are kept in list format
        for key in ('obs', 'err_std', 't', 'y', 'x', 'z'):
            obs_seq[key] = np.array(obs_seq[key])

        if is_vector:
            obs_seq['obs'] = obs_seq['obs'].T

        #convert units if necessary
        if 'units' in kwargs and kwargs['units'] != native_units:
            obs_seq['obs'] = units_convert(native_units, kwargs['units'], obs_seq['obs'])
            obs_seq['err_std'] = units_convert(native_units, kwargs['units'], obs_seq['err_std'])

        return obs_seq

    def random_network(self):
        raise NotImplementedError

    def _load_coeffs_biases(self, date):
        if self.coeffs is None:
            df = pd.read_csv(self.coefficients_file)

            index = df[df['date'] == float(date)].index.tolist()
            if len(index) > 4 : # One date for each channel
                return 'The date appears more than once in the csv file, check!'

            m_a1, m_a2, m_c = ([] for _ in range(3))
            self.coeffs = dict()
            for ich, ch in enumerate(self.channels) :
                self.coeffs[ch[2::]] = dict()
                m_a1.append(df.iloc[index[0] + ich]['a1'])
                self.coeffs[ch[2::]]['a1'] = df.iloc[index[0] + ich]['a1']
                m_a2.append(df.iloc[index[0] + ich]['a2'])
                self.coeffs[ch[2::]]['a2'] = df.iloc[index[0] + ich]['a2']
                m_c.append(df.iloc[index[0] + ich]['c'])
                self.coeffs[ch[2::]]['c'] = df.iloc[index[0] + ich]['c']

        if self.ow_bias is None:
            df = pd.read_csv(self.ow_bias_file)

            self.ow_bias = dict()
            for ch in self.channels:
                self.ow_bias[ch] = df[df['Month'] == int(date[4:6])][f'Mean_Value_{ch[2::]}'].values[0]

    def _simulated_tb(self, channel, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        model = kwargs['model']
        grid = kwargs['grid']
        x, y = kwargs['x'], kwargs['y']

        #load coefficients and biases from csv files of the given date
        self._load_coeffs_biases(kwargs['time'].strftime('%Y%m%d'))
        if not isinstance(self.ow_bias, dict):
            raise ValueError("AMSR2Obs._simulated_tb: error reading ow_bias, check the ow_bias_file")
        ow_bias = self.ow_bias[channel]

        #load atmospheric data
        ##should come from model.read_var, for convenience I read the copy from observation files for now
        wind = self.read_obs(**{**kwargs, 'name': 'wind_speed', 'units': 'm/s'})['obs']
        vapor = self.read_obs(**{**kwargs, 'name': 'water_vapor', 'units': 'g/kg'})['obs']
        liquid = self.read_obs(**{**kwargs, 'name': 'liquid_water', 'units': 'g/kg'})['obs']
        airtemp = self.read_obs(**{**kwargs, 'name': 'air_temp', 'units': 'K'})['obs']

        #sea ice concentration from model
        sic_from_model = model.read_var(**{**kwargs, 'name': 'seaice_conc', 'units': 1})
        model.grid.set_destination_grid(grid)
        sic_on_grid = model.grid.convert(sic_from_model)
        sic = grid.interp(sic_on_grid, x, y)

        ##these properties come from the observation files
        dal = self.read_obs(**{**kwargs, 'name': 'dal', 'units': 1})['obs']
        eia = self.read_obs(**{**kwargs, 'name': 'eia', 'units': 'deg'})['obs']

        tb, _ = simulated_tb_v03(vapor, wind, liquid, airtemp,
                                 sic, eia, channel[2::],
                                 ow_bias = ow_bias,
                                 dict_coeffs = self.coeffs,
                                 opt_em = 2, dal = dal)
        return tb

    def _get_obs_operator(self, channel):
        def obs_operator(**kwargs):
            return self._simulated_tb(channel, **kwargs)
        return obs_operator
