import os
from pyproj import Proj
import numpy as np
from NEDAS.grid import Grid
from NEDAS.utils.conversion import units_convert, s2t, dt1h
from NEDAS.datasets.ecmwf import atmos_utils
from NEDAS.datasets import Dataset

from NEDAS.datasets.ecmwf.era5 import ERA5Data
era5 = ERA5Data()

class EcmwfForecastData(Dataset):
    def __init__(self, config_file=None, parse_args=False, **kwargs):
        super().__init__(config_file, parse_args, **kwargs)
        self.variables = {
            'atmos_surf_velocity': {'name':("10 metre U wind component", "10 metre V wind component"), 'is_vector':True, 'units':'m/s'},
            'atmos_surf_temp': {'name':"2 metre temperature", 'is_vector':False, 'units':'K'},
            'atmos_surf_dewpoint':  {'name':"2 metre dewpoint temperature", 'is_vector':False, 'units':'K'},
            'atmos_surf_press': {'name':"Mean sea level pressure", 'is_vector':False, 'units':'Pa'},
            'atmos_cloud_cover':  {'name':"Total cloud cover", 'is_vector':False, 'units':1},
            'atmos_precip': {'name':"Total precipitation", 'is_vector':False, 'units':'m/s'},
            'atmos_down_longwave': {'name':"Surface long-wave (thermal) radiation downwards", 'is_vector':False, 'units':'W/m2'},
            'atmos_down_shortwave': {'name':"Surface short-wave (solar) radiation downwards", 'is_vector':False, 'units':'W/m2'},
            'atmos_surf_vapor_mix': {'getter':self.get_vapmix, 'is_vector':False, 'units':'kg/kg'},
            }
        self.files = {}
        self.lookup = {}
        self.grid = None
        if isinstance(self.time_start, str):
            self.time_start = s2t(self.time_start)

    ###format filename
    def filename(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        t = self.time_start
        file = os.path.join(kwargs['path'], f"{t:%Y-%m}", f"fc_{t:%Y-%m-%d}.grb")
        assert os.path.exists(file), f"file {file} does not exist"
        return file

    def open_file(self, fname):
        if fname not in self.files:
            try:
                import pygrib
            except ImportError:
                raise RuntimeError("pygrib package is required to open ecmwf forecast data files.")
            print(f"opening file {fname}")
            self.files[fname] = pygrib.open(fname)
            self.get_record_id_lookup(fname)

    def close_file(self, fname):
        self.files[fname].close()

    def read_grid(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        fname = self.filename(**kwargs)
        self.open_file(fname)
        lat, lon = self.files[fname].message(1).latlons()
        self.grid = Grid(Proj('+proj=longlat'), lon, lat, cyclic_dim='x', pole_dim='y', pole_index=(0,))

    def get_record_id_lookup(self, fname):
        self.lookup[fname] = {}
        grbs = self.files[fname]
        for i in range(grbs.messages):
            grb = grbs.message(i+1)
            variable_name = grb.name
            forecast_hours = int(grb.stepRange)
            start_date = grb.analDate
            member = grb.perturbationNumber
            key = (variable_name, start_date, forecast_hours, member)
            self.lookup[fname][key] = i+1

    def read_data_from_grb(self, fname, time, member, vname, units):
        forecast_hours = int((time - self.time_start) / dt1h)
        ##build search key
        key = (vname, self.time_start, forecast_hours, member)
        ##look up the message id
        rec_id = self.lookup[fname][key]
        ##read the message into var
        grb = self.files[fname].message(rec_id)
        var = grb.values
        return var

    def read_var(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        name = kwargs['name']
        assert name is not None, 'please specify which variable (name=?) to get'
        time = kwargs['time']
        k = kwargs.get('k', 0)
        member = kwargs['member']
        assert member is not None, 'please specify which member (member=?) to get'
        rec = self.variables[name]

        if 'name' in rec:
            fname = self.filename(**kwargs)
            self.open_file(fname)

            if rec['is_vector']:
                var1 = self.read_data_from_grb(fname, time, member, rec['name'][0], rec['units'])
                var2 = self.read_data_from_grb(fname, time, member, rec['name'][1], rec['units'])
                var = np.array([var1, var2])
            else:
                var = self.read_data_from_grb(fname, time, member, rec['name'], rec['units'])

            ##some variables are accumulated over the dt_hours, convert them to fluxes
            if name in ('atmos_precip', 'atmos_down_shortwave', 'atmos_down_longwave'):
                if time == self.time_start:
                    ##first time step is zeros for these variables
                    ##to ensure continuity in time, get the variable snapshot from ERA5 instead
                    era5.read_grid(name=name, time=time)
                    tmp = era5.read_var(name=name, time=time)
                    ##convert to grid
                    era5.grid.set_destination_grid(self.grid)
                    var = era5.grid.convert(tmp)
                else:
                    prev_time = time - dt1h * self.dt_hours
                    var -= self.read_data_from_grb(fname, prev_time, member, rec['name'], rec['units'])
                    ##need to convert the fluxes to per second units
                    var /= 3600. * self.dt_hours
                var[np.where(var<0.)] = 0. ##ensure positive definite
        else:
            var = rec['getter'](**kwargs)

        ##convert units if necessary
        var = units_convert(rec['units'], kwargs['units'], var)
        return var

    def get_vapmix(self, **kwargs):
        dewpoint = self.read_var(**{**kwargs, 'name':'atmos_surf_dewpoint', 'units':'K'})
        press = self.read_var(**{**kwargs, 'name':'atmos_surf_press', 'units':'Pa'})
        e = atmos_utils.satvap(dewpoint)
        vapmix = atmos_utils.vapmix(e, press)
        return vapmix

    def read_obs(self):
        raise NotImplementedError
