import os
import glob
import numpy as np
from NEDAS.utils.conversion import dt1h
from NEDAS.models.topaz.time_format import datetojul
from NEDAS.datasets import Dataset
from .uf_data import read_uf_data

class TopazPrepObs(Dataset):
    """
    Observations already preprocessed by TOPAZ EnKF Prep_Routines, saved in unformatted binary format
    """
    def __init__(self, config_file=None, parse_args=False, **kwargs):
        super().__init__(config_file, parse_args, **kwargs)

        self.variables = {
            'ocean_temp': {'name':'TEM', 'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'C'},
            'ocean_saln': {'name':'SAL', 'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'psu'},
            'ocean_surf_temp': {'name':'SST', 'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'C'},
            'ocean_surf_height_anomaly': {'name':'TSLA', 'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'m'},
            'seaice_conc': {'name':'ICEC', 'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':1},
            'seaice_thick': {'name':'HICE', 'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'m'},
            'seaice_drift': {'name':'IDRFT', 'dtype':'float', 'is_vector':True, 'z_units':'m', 'units':'km'},
            }

        self.obs_operator = {
            'seaice_drift': self.get_seaice_drift,
            }

    def filename(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        name = kwargs['name']
        time = kwargs['time']
        path = kwargs['path']

        vname = self.variables[name]['name']
        dirname = vname
        native_name = vname
        if name == 'seaice_drift':
            native_name += '?'   ##i=1, 2, 3, 4, 5: 2day drift in km with starting day = current day -i-1

        file_list = []
        if time is not None:
            time_str = "{:5d}".format(int(datetojul(time)))
        else:
            time_str = '?????'
        search = os.path.join(path, dirname, 'obs_'+native_name+'_'+time_str+'.uf')
        file_list = glob.glob(search)

        assert len(file_list)>0, 'no matching files found: '+search
        return file_list

    def random_network(self, **kwargs):
        raise NotImplementedError('random_network: ERROR: random_network not implemented for topaz dataset')

    def read_obs(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        name = kwargs['name']
        time = kwargs['time']
        grid = kwargs['grid']
        model = kwargs.get('model')
        if self.vthin:
            ##check if model.z is available for vertical thinning
            if model is None or not hasattr(model, 'z') or model.z is None:
                print("WARNING: topaz.dataset: model.z levels are not provided, setting vthin to False")
                self.vthin = False

        is_vector = self.variables[name]['is_vector']
        obs_seq = {'obs':[], 't':[], 'z':[], 'y':[], 'x':[], 'err_std':[],
                   'ipiv':[], 'jpiv':[], 'ns':[], 'in_coords':[], 'stat':[], 'i_orig_grid':[], 'j_orig_grid':[], 'h':[], 'date':[]}

        try:
            fname = self.filename(**kwargs)
        except AssertionError:
            ##just return empty obs_seq if no matching files found
            for key in obs_seq.keys():
                obs_seq[key] = np.array([])
            return obs_seq

        for file in fname:
            data = read_uf_data(file)

            if is_vector:
                nobs = len(data) // 2
            else:
                nobs = len(data)

            # if ice drift obs, get the time lag from file name
            obsType = os.path.basename(file).split('_')[1]
            if obsType[:5] == 'IDRFT':
                lag_days = int(obsType[5]) + 1
                t = time - dt1h * 24 * lag_days   ##start time of the drift vectors
            else:
                t = time

            for i in range(nobs):
                if is_vector:
                    obs = [data[i][0], data[nobs+i][0]]
                else:
                    obs = data[i][0]
                obs_seq['obs'].append(obs)
                obs_seq['t'].append(t)
                obs_seq['z'].append(-data[i][5])  ##depth
                lon, lat = data[i][3], data[i][4]
                x, y = grid.proj(lon, lat)
                obs_seq['y'].append(y)
                obs_seq['x'].append(x)
                obs_seq['err_std'].append(np.sqrt(data[i][1]))

                ##additional info
                obs_seq['ipiv'].append(data[i][6])
                obs_seq['jpiv'].append(data[i][7])
                obs_seq['ns'].append(data[i][8])
                obs_seq['in_coords'].append([data[i][9], data[i][10], data[i][11], data[i][12]])
                obs_seq['stat'].append(data[i][13])
                obs_seq['i_orig_grid'].append(data[i][14])
                obs_seq['j_orig_grid'].append(data[i][15])
                obs_seq['h'].append(data[i][16])
                obs_seq['date'].append(data[i][17])

        if self.vthin and name in ['ocean_temp', 'ocean_saln']:
            ##thin observation profiles vertically by picking those closest to model levels only
            mask = np.full(len(obs_seq['obs']), False)
            

        for key in obs_seq.keys():
            obs_seq[key] = np.array(obs_seq[key])

        if is_vector:
            obs_seq['obs'] = obs_seq['obs'].T  ##make vector dimension [2,nobs]
        return obs_seq

    def get_seaice_drift(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        grid = kwargs['grid']  ##output grid
        path = kwargs['path']
        member = kwargs['member']
        model = kwargs['model']  ##model object
        model.grid.set_destination_grid(grid)

        start_x = kwargs['x']
        start_y = kwargs['y']
        start_t = kwargs['t']
        x = start_x.copy()
        y = start_y.copy()
        obs_seq = np.zeros((2,)+x.shape)

        ##go through each of the 5 start times
        for uniq_start_t in np.unique(start_t):
            ind = np.where(start_t==uniq_start_t)

            for day in range(2):
                t = uniq_start_t + day * dt1h * 24

                ##get daily mean seaice velocity
                model_vel = model.read_var(path=path, name='seaice_velocity_daily', time=t, member=member, units='km/day')
                ##convert to grid
                vel = model.grid.convert(model_vel, is_vector=True)
                ##get velocity at x, y position
                u = grid.interp(vel[0,...], x[ind], y[ind])  ##km over 1 day
                v = grid.interp(vel[1,...], x[ind], y[ind])

                ##increment the x,y components
                x[ind] += u * 1000.
                y[ind] += v * 1000.

        obs_seq[0, :] = (x - start_x) / 1000.
        obs_seq[1, :] = (y - start_y) / 1000.

        return obs_seq

