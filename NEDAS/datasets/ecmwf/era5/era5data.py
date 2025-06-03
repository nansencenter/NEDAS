"""Interface to ERA5 dataset stored 6 hourly in netCDF formats"""

import os
import numpy as np
from datetime import datetime, timedelta, timezone
from pyproj import Proj
import netCDF4
from NEDAS.grid import Grid
from NEDAS.utils.conversion import units_convert
from NEDAS.datasets import Dataset
from NEDAS.datasets.ecmwf import atmos_utils

class ERA5Data(Dataset):
    def __init__(self, config_file=None, parse_args=False, **kwargs):
        super().__init__(config_file, parse_args, **kwargs)

        ##variable dictionary for ERA5 naming convention
        self.variables = {
            'atmos_surf_velocity': {'name':('10U', '10V'), 'is_vector':True, 'units':'m/s'},
            'atmos_surf_temp': {'name':'2T', 'is_vector':False, 'units':'K'},
            'atmos_surf_dewpoint': {'name':'2D', 'is_vector':False, 'units':'K'},
            'atmos_surf_press': {'name':'MSL', 'is_vector':False, 'units':'Pa'},
            'atmos_precip':        {'name':'TP', 'is_vector':False, 'units':'m/s'},
            'atmos_down_longwave': {'name':'STRD', 'is_vector':False, 'units':'W/m2'},
            'atmos_down_shortwave': {'name':'SSRD', 'is_vector':False, 'units':'W/m2'},
            'atmos_surf_vapor_mix': {'getter':self.get_vapmix, 'is_vector':False, 'units':'kg/kg'},
            }

        self.grid = None

    ###format filename
    def filename(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        assert kwargs['time'] is not None, 'please specify time'
        assert kwargs['name'] is not None, 'please specify variable name'
        year = '{:04d}'.format(kwargs['time'].year)
        rec = self.variables[kwargs['name']]
        if rec['is_vector']:
            files = []
            for name in rec['name']:
                file = os.path.join(kwargs['path'], f"6h.{name}_{year}.nc")
                assert os.path.exists(file), 'file not found: '+file
                files.append(file)
            return files
        else:
            file = os.path.join(kwargs['path'], f"6h.{rec['name']}_{year}.nc")
            assert os.path.exists(file), 'file not found: '+file
            files = [file]
        return files

    def read_grid(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        fname = self.filename(**kwargs)[0]
        with netCDF4.Dataset(fname) as f:
            lon = f['longitude'][:].data
            lat = f['latitude'][:].data
            x, y = np.meshgrid(lon, lat)
        self.grid = Grid(Proj('+proj=longlat'), x, y, cyclic_dim='x', pole_dim='y', pole_index=(0,))

    ##find the nearest index in data for the given t
    def find_time_index(self, time_series, time):
        t_ = (time - datetime(1900,1,1,tzinfo=timezone.utc)) / timedelta(hours=1)
        ind = np.abs(time_series - t_).argmin()
        return ind

    def read_var(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        name = kwargs['name']
        assert name is not None, 'please specify which variable (name=?) to get'
        time = kwargs['time']
        k = kwargs.get('k', 0)
        rec = self.variables[name]

        if time is None:
            t_index = 0

        if 'name' in rec:
            files = self.filename(**kwargs)
            if rec['is_vector']:
                var = []
                for i,file in enumerate(files):
                    with netCDF4.Dataset(file) as f:
                        t_in_file = f['time'][:].data
                        t_index = self.find_time_index(t_in_file, time)
                        dat = f.variables[rec['name'][i]][t_index, ...]
                        tmp = dat.data
                        tmp[dat.mask] = np.nan
                        var.append(tmp)
                var = np.array(var)
            else:
                with netCDF4.Dataset(files[0]) as f:
                    t_in_file = f['time'][:].data
                    t_index = self.find_time_index(t_in_file, time)
                    dat = f.variables[rec['name']][t_index, ...]
                    var = dat.data
                    var[dat.mask] = np.nan
            ##convert units if necessary
            if rec['name'] in ('TP', 'STRD', 'SSRD'):
                ##need to convert the fluxes to per second units (they are per 6 hours in the dataset)
                var /= 3600. * 6

        else:
            var = rec['getter'](**kwargs)

        var = units_convert(rec['units'], kwargs['units'], var)
        return var

    def get_vapmix(self, **kwargs):
        dewpoint = self.read_var(**{**kwargs, 'name':'atmos_surf_dewpoint', 'units':'K'})
        e_sat = atmos_utils.satvap(dewpoint)
        press = self.read_var(**{**kwargs, 'name':'atmos_surf_press', 'units':'Pa'})
        vapmix = atmos_utils.vapmix(e_sat, press)
        return vapmix

    def read_obs(self):
        raise NotImplementedError