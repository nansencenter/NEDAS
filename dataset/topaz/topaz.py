import numpy as np
import os
import glob
from utils.conversion import dt1h
from models.topaz.model_grid import get_topaz_grid
from models.topaz.time_format import datetojul
from .uf_data import read_uf_data
from ..dataset_config import DatasetConfig

class TopazPrepData(DatasetConfig):
    """
    Observations already preprocessed by TOPAZ EnKF Prep_Routines, saved in unformatted binary format
    """
    def __init__(self, config_file=None, parse_args=False, **kwargs):
        super().__init__(config_file, parse_args, **kwargs)

        self.variables = {
            'ocean_temp': {'name':'TEM', 'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'K'},
            'ocean_saln': {'name':'SAL', 'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'psu'},
            'ocean_surf_temp': {'name':'SST', 'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'K'},
            'ocean_surf_height': {'name':'TSLA', 'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'m'},
            'seaice_conc': {'name':'ICEC', 'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'%'},
            'seaice_thick': {'name':'HICE', 'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'m'},
            'seaice_drift': {'name':'IDRFT', 'dtype':'float', 'is_vector':True, 'z_units':'m', 'units':'km'},
            }
        self.z_units = 'm'

        ##set default grid if not specified
        if self.grid is None:
            ##use native grid
            grid_info_file = os.path.join(self.basedir, 'topo', 'grid.info')
            self.grid = get_topaz_grid(grid_info_file)
        if self.mask is None:
            self.mask = np.full(self.grid.x.shape, False)
        if self.z_coords is None:
            self.z_coords = np.zeros(self.grid.x.shape)

    def filename(self, **kwargs):
        kwargs = super().filename(**kwargs)

        vname = self.variables[self.name]['name']
        dirname = vname
        if self.name == 'seaice_drift':
            native_name += '1'   ##i=1, 2, 3, 4, 5: 2day drift in km with starting day = current day -i-1

        file_list = []
        if self.time is not None:
            if 'obs_window_min' in kwargs and 'obs_window_max' in kwargs:
                d_range = np.arange(kwargs['obs_window_min'], kwargs['obs_window_max'])
            else:
                d_range = [0]
            for d in d_range:
                t = kwargs['time'] + d * dt1h
                time_str = "{:5d}".format(int(datetojul(t)))
                search = os.path.join(self.path, dirname, 'obs_'+native_name+'_'+time_str+'.uf')
                for result in glob.glob(search):
                    if result not in file_list:
                        file_list.append(result)
        else:
            time_str = '?????'
            search = os.path.join(self.path, dirname, 'obs_'+native_name+'_'+time_str+'.uf')
            file_list = glob.glob(search)

        assert len(file_list)>0, 'no matching files found: '+search
        return file_list

    def read_obs(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        name = kwargs['name']

        is_vector = self.variables[name]['is_vector']
        obs_seq = {'obs':[], 't':[], 'z':[], 'y':[], 'x':[], 'err_std':[], }

        for file in self.filename(**kwargs):
            data = read_uf_data(file)

            if is_vector:
                nobs = len(data) // 2
            else:
                nobs = len(data)

            for i in range(nobs):
                if is_vector:
                    obs = [data[i][0], data[nobs+i][0]]
                else:
                    obs = data[i][0]
                obs_seq['obs'].append(obs)
                obs_seq['t'].append(kwargs['time'])
                obs_seq['z'].append(-data[i][5])  ##depth
                lon, lat = data[i][3], data[i][4]
                x, y = self.grid.proj(lon, lat)
                obs_seq['y'].append(y)
                obs_seq['x'].append(x)
                obs_seq['err_std'].append(np.sqrt(data[i][1]))

        for key in obs_seq.keys():
            obs_seq[key] = np.array(obs_seq[key])

        if is_vector:
            obs_seq['obs'] = obs_seq['obs'].T  ##make vector dimension [2,nobs]
        return obs_seq

