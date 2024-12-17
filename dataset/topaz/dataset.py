import numpy as np
import os
import glob
from utils.conversion import dt1h
from models.topaz.model_grid import get_topaz_grid
from models.topaz.time_format import datetojul
from .uf_data import read_uf_data
from ..dataset_config import DatasetConfig

class Dataset(DatasetConfig):
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

    def filename(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        name = kwargs['name']
        time = kwargs['time']
        path = kwargs['path']

        vname = self.variables[name]['name']
        dirname = vname
        native_name = vname
        if name == 'seaice_drift':
            native_name += '1'   ##i=1, 2, 3, 4, 5: 2day drift in km with starting day = current day -i-1

        file_list = []
        if time is not None:
            time_str = "{:5d}".format(int(datetojul(time)))
        else:
            time_str = '?????'
        search = os.path.join(path, dirname, 'obs_'+native_name+'_'+time_str+'.uf')
        file_list = glob.glob(search)

        assert len(file_list)>0, 'no matching files found: '+search
        return file_list

    def read_obs(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        name = kwargs['name']
        model_grid = kwargs['grid']

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
                x, y = model_grid.proj(lon, lat)
                obs_seq['y'].append(y)
                obs_seq['x'].append(x)
                obs_seq['err_std'].append(np.sqrt(data[i][1]))

        for key in obs_seq.keys():
            obs_seq[key] = np.array(obs_seq[key])

        if is_vector:
            obs_seq['obs'] = obs_seq['obs'].T  ##make vector dimension [2,nobs]
        return obs_seq

